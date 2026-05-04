#!/usr/bin/env python3
"""Serve the local LLM study studio and its control API."""

from __future__ import annotations

import json
import os
import re
import signal
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingTCPServer
from typing import Any
from urllib.parse import urlparse

from studio.config import DEFAULT_CHAT_INSTRUCTION, JSONL_SCHEMAS, STUDIO_METHODS


PORT = 8765
ROOT = Path(__file__).resolve().parents[1]
VISUALIZER_DIR = ROOT / "visualizer"
RUNTIME_DIR = VISUALIZER_DIR / "runtime"
CONTROL_PATH = RUNTIME_DIR / "control.json"
CATALOG_PATH = VISUALIZER_DIR / "traces" / "catalog.json"
LIVE_TRACE = "visualizer/traces/live.json"
STUDIO_CUSTOM_DATA_DIR = RUNTIME_DIR / "custom-data"
STUDIO_RUNS_DIR = RUNTIME_DIR / "studio-runs"
STUDIO_PROFILE_PATH = RUNTIME_DIR / "studio-profile.json"
STUDIO_DATA_DIR = VISUALIZER_DIR / "studio" / "data"
EXTRA_ARG_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
RESERVED_EXTRA_ARGS = {
    "method",
    "model-name",
    "model_name",
    "data",
    "trace",
    "trace-delay",
    "trace_delay",
    "output-dir",
    "output_dir",
    "adapter-dir",
    "adapter_dir",
    "generation-dir",
    "generation_dir",
    "report",
    "index",
    "metrics",
    "config",
    "eval-prompts",
    "eval_prompts",
    "max-steps",
    "max-length",
    "max-new-tokens",
    "learning-rate",
    "rank",
    "alpha",
    "dropout",
    "per-device-train-batch-size",
    "per-device-eval-batch-size",
    "per-device-batch-size",
    "gradient-accumulation-steps",
    "warmup-steps",
    "weight-decay",
    "lr-scheduler-type",
    "target-modules",
    "beta",
    "device",
    "seq-length",
    "target-module-count",
    "quick",
    "skip-hf-metadata",
    "local-files-only",
    "no-gradient-checkpointing",
}

QUICK_ARGS = {
    "04-trainer": ["--max-steps", "1"],
    "05-lora": ["--max-steps", "1"],
    "06-peft-lora": ["--max-steps", "1"],
    "07-sft-baseline": ["--max-steps", "1", "--max-new-tokens", "16"],
    "08-dpo-preference": ["--quick", "--max-steps", "1", "--max-new-tokens", "16"],
    "09-rlhf-reward": ["--quick"],
    "10-qlora-engineering": ["--quick"],
}
MAX_STEPS_LESSONS = {
    "04-trainer",
    "05-lora",
    "06-peft-lora",
    "07-sft-baseline",
    "08-dpo-preference",
}
MAX_NEW_TOKEN_LESSONS = {"07-sft-baseline", "08-dpo-preference"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return fallback


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def load_catalog() -> list[dict[str, Any]]:
    payload = read_json(CATALOG_PATH, {"courses": []})
    return [course for course in payload.get("courses", []) if course.get("lesson_id")]


def project_path(repo_path: str) -> Path:
    candidate = (ROOT / repo_path).resolve()
    if ROOT not in candidate.parents and candidate != ROOT:
        raise ValueError(f"Path must stay inside this repo: {repo_path}")
    return candidate


def require_existing_repo_file(repo_path: str) -> Path:
    path = project_path(repo_path)
    if not path.exists() or not path.is_file():
        raise ValueError(f"Dataset file not found: {repo_path}")
    return path


def require_existing_studio_data_file(repo_path: str) -> Path:
    path = require_existing_repo_file(repo_path)
    allowed_roots = [STUDIO_DATA_DIR.resolve(), STUDIO_CUSTOM_DATA_DIR.resolve()]
    if not any(path == root or root in path.parents for root in allowed_roots):
        raise ValueError("Studio datasets must be under visualizer/studio/data or visualizer/runtime/custom-data.")
    return path


def studio_method_key(payload: dict[str, Any]) -> str:
    method = str(payload.get("method") or "sft-lora")
    if method not in STUDIO_METHODS:
        raise ValueError(f"Unknown training method: {method}")
    return method


def studio_method(payload: dict[str, Any]) -> dict[str, Any]:
    return STUDIO_METHODS[studio_method_key(payload)]


def repo_relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def read_studio_profile() -> dict[str, Any]:
    profile = read_json(STUDIO_PROFILE_PATH, {})
    if not isinstance(profile, dict):
        return {}
    adapter_dir = str(profile.get("adapter_dir") or "").strip()
    if adapter_dir:
        try:
            adapter_path = project_path(adapter_dir)
            profile["adapter_available"] = adapter_path.exists() and (adapter_path / "adapter_config.json").exists()
            profile["adapter_config_file"] = (
                repo_relative(adapter_path / "adapter_config.json") if profile["adapter_available"] else None
            )
        except ValueError:
            profile["adapter_available"] = False
            profile["adapter_config_file"] = None
    return profile


def validate_jsonl_schema(text: str, schema_name: str) -> dict[str, Any]:
    schema = JSONL_SCHEMAS.get(schema_name)
    if not schema:
        raise ValueError(f"No JSONL schema for method: {schema_name}")
    rows: list[dict[str, Any]] = []
    required = schema["required"]
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number}: each JSONL row must be an object")
        missing = [
            key for key in required
            if not isinstance(row.get(key), str) or (key != "input" and not row.get(key, "").strip())
        ]
        if missing:
            raise ValueError(f"line {line_number}: missing string fields: {', '.join(missing)}")
        rows.append(row)
    if not rows:
        raise ValueError("Dataset JSONL is empty.")
    if len(rows) < int(schema["min_rows"]):
        raise ValueError(f"Dataset needs at least {schema['min_rows']} rows.")
    return {
        "rows": len(rows),
        "fields": sorted({key for row in rows for key in row}),
    }


def validate_sft_jsonl(text: str) -> dict[str, Any]:
    return validate_jsonl_schema(text, "sft")


def resolve_studio_data(payload: dict[str, Any], method: dict[str, Any], run_dir: Path) -> tuple[str, dict[str, Any]]:
    schema = str(method["schema"])
    if schema == "none":
        return "", {"mode": "none", "rows": 0, "fields": []}
    run_data_dir = run_dir / "data"
    run_data_dir.mkdir(parents=True, exist_ok=True)
    mode = str(payload.get("dataset_mode") or "paste")
    if mode == "studioExample":
        repo_path = str(method["default_data"])
        path = require_existing_studio_data_file(repo_path)
        summary = validate_jsonl_schema(path.read_text(encoding="utf-8"), schema)
        data_path = run_data_dir / path.name
        shutil.copyfile(path, data_path)
        return repo_relative(data_path), {"mode": mode, "source": repo_path, "written": repo_relative(data_path), **summary}
    if mode == "studioPath":
        repo_path = str(payload.get("data_path") or "").strip()
        if not repo_path:
            raise ValueError("data_path is required when dataset_mode=studioPath")
        path = require_existing_studio_data_file(repo_path)
        summary = validate_jsonl_schema(path.read_text(encoding="utf-8"), schema)
        data_path = run_data_dir / path.name
        shutil.copyfile(path, data_path)
        return repo_relative(data_path), {"mode": mode, "source": repo_relative(path), "written": repo_relative(data_path), **summary}
    if mode in {"paste", "upload"}:
        text = str(payload.get("custom_dataset_text") or payload.get("upload_dataset_text") or "")
        summary = validate_jsonl_schema(text, schema)
        suffix = Path(str(payload.get("source_filename") or "")).suffix or ".jsonl"
        data_path = run_data_dir / f"{schema}{suffix}"
        data_path.write_text("\n".join(line.strip() for line in text.splitlines() if line.strip()) + "\n", encoding="utf-8")
        summary_payload = {
            "mode": mode,
            "written": repo_relative(data_path),
            "source_filename": str(payload.get("source_filename") or "").strip(),
            **summary,
        }
        return repo_relative(data_path), summary_payload
    raise ValueError(f"Unknown dataset_mode: {mode}")


def preview_studio_data(payload: dict[str, Any]) -> dict[str, Any]:
    method = studio_method(payload)
    schema = str(method["schema"])
    if schema == "none":
        return {"mode": "none", "rows": 0, "fields": []}
    mode = str(payload.get("dataset_mode") or "paste")
    if mode == "studioExample":
        repo_path = str(method["default_data"])
        path = require_existing_studio_data_file(repo_path)
        return {"mode": mode, "data_path": repo_path, **validate_jsonl_schema(path.read_text(encoding="utf-8"), schema)}
    if mode == "studioPath":
        repo_path = str(payload.get("data_path") or "").strip()
        path = require_existing_studio_data_file(repo_path)
        return {"mode": mode, "data_path": repo_relative(path), **validate_jsonl_schema(path.read_text(encoding="utf-8"), schema)}
    if mode in {"paste", "upload"}:
        text = str(payload.get("custom_dataset_text") or payload.get("upload_dataset_text") or "")
        return {"mode": mode, **validate_jsonl_schema(text, schema)}
    raise ValueError(f"Unknown dataset_mode: {mode}")


def add_numeric_arg(command: list[str], payload: dict[str, Any], key: str, flag: str, cast: type = int) -> None:
    value = payload.get(key)
    if value in (None, ""):
        return
    command.extend([flag, str(cast(value))])


def add_string_arg(command: list[str], payload: dict[str, Any], key: str, flag: str) -> None:
    value = str(payload.get(key) or "").strip()
    if value:
        command.extend([flag, value])


def add_bool_arg(command: list[str], payload: dict[str, Any], key: str, flag: str) -> None:
    if bool(payload.get(key)):
        command.append(flag)


def normalize_extra_args_config(payload: dict[str, Any]) -> dict[str, Any]:
    raw = payload.get("extra_args_json")
    if raw in (None, ""):
        return {}
    if isinstance(raw, str):
        try:
            config = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"extra_args_json invalid JSON: {exc.msg}") from exc
    else:
        config = raw
    if not isinstance(config, dict) or isinstance(config, list):
        raise ValueError("extra_args_json must be a JSON object.")
    normalized: dict[str, Any] = {}
    for key, value in config.items():
        normalized_key = str(key).strip()
        cli_key = normalized_key.replace("_", "-")
        if not EXTRA_ARG_NAME_RE.match(cli_key):
            raise ValueError(f"extra arg name is invalid: {normalized_key}")
        if cli_key in RESERVED_EXTRA_ARGS:
            raise ValueError(f"extra arg is managed by Studio and cannot be overridden: {normalized_key}")
        if isinstance(value, list):
            normalized[cli_key] = ",".join(str(item) for item in value)
        elif isinstance(value, dict):
            normalized[cli_key] = json.dumps(value, ensure_ascii=False)
        elif value is None or isinstance(value, (str, int, float, bool)):
            normalized[cli_key] = value
        else:
            normalized[cli_key] = str(value)
    return normalized


def lesson_script(lesson_id: str) -> Path:
    return ROOT / "lessons" / lesson_id / "run.py"


def project_python() -> str:
    venv_python = ROOT / ".venv" / "bin" / "python"
    return str(venv_python) if venv_python.exists() else sys.executable


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def log_tail(path: Path, limit: int = 8000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    return text[-limit:]


def initial_control() -> dict[str, Any]:
    return {
        "mode": "running",
        "step_token": 0,
        "stop_requested": False,
        "updated_at": now_iso(),
    }


class JobManager:
    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.job: dict[str, Any] | None = None
        self.process: subprocess.Popen[str] | None = None
        write_json(CONTROL_PATH, initial_control())

    def state(self) -> dict[str, Any]:
        with self.lock:
            self._refresh_locked()
            job = dict(self.job) if self.job else None
            if job and job.get("log_path"):
                job["log_tail"] = log_tail(Path(job["log_path"]))
            return {
                "job": job,
                "control": read_json(CONTROL_PATH, initial_control()),
                "catalog": load_catalog(),
                "studio_profile": read_studio_profile(),
            }

    def start_lesson(self, lesson_id: str, options: dict[str, Any]) -> dict[str, Any]:
        catalog_ids = {course["lesson_id"] for course in load_catalog()}
        if lesson_id not in catalog_ids:
            raise ValueError(f"Unknown lesson_id: {lesson_id}")
        script = lesson_script(lesson_id)
        if not script.exists():
            raise ValueError(f"Missing lesson script: {script.relative_to(ROOT)}")

        command = [
            project_python(),
            str(script.relative_to(ROOT)),
            "--trace",
            LIVE_TRACE,
            "--trace-delay",
            str(float(options.get("trace_delay") or 0.0)),
        ]
        if options.get("quick"):
            command.extend(QUICK_ARGS.get(lesson_id, []))
        else:
            max_steps = options.get("max_steps")
            if max_steps and lesson_id in MAX_STEPS_LESSONS:
                command.extend(["--max-steps", str(int(max_steps))])
            max_new_tokens = options.get("max_new_tokens")
            if max_new_tokens and lesson_id in MAX_NEW_TOKEN_LESSONS:
                command.extend(["--max-new-tokens", str(int(max_new_tokens))])

        return self._start(command, "lesson", lesson_id, options)

    def start_studio_sft(self, options: dict[str, Any]) -> dict[str, Any]:
        method_key = studio_method_key(options)
        method = studio_method(options)
        model_name = str(options.get("model_name") or "auto").strip()
        if not model_name:
            raise ValueError("model_name is required")
        run_dir = STUDIO_RUNS_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{method_key}_{uuid.uuid4().hex[:8]}"
        output_dir = run_dir / "trainer"
        adapter_dir = run_dir / "adapter"
        generation_dir = run_dir / "generations"
        report_path = run_dir / "report.md"
        index_path = run_dir / "index.html"
        trace_path = run_dir / "trace.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        data_path, data_summary = resolve_studio_data(options, method, run_dir)
        extra_args = normalize_extra_args_config(options)
        extra_args_path = run_dir / "extra-args.json"
        if extra_args:
            write_json(extra_args_path, extra_args)

        command = [
            project_python(),
            "visualizer/studio/run.py",
            "--method",
            method_key,
            "--trace",
            repo_relative(trace_path),
            "--trace-delay",
            str(float(options.get("trace_delay") or 0.0)),
            "--model-name",
            model_name,
        ]
        if method["schema"] != "none":
            command.extend(["--data", data_path])

        studio_id = str(method["lesson_id"])
        engine_kind = str(method["engine_kind"])
        metrics_path = output_dir / "metrics.json" if engine_kind == "dpo" else run_dir / "metrics.json"
        metrics_display = repo_relative(metrics_path) if engine_kind in {"sft", "dpo"} else ""
        command.extend(
            [
                "--output-dir",
                repo_relative(output_dir),
                "--adapter-dir",
                repo_relative(adapter_dir),
                "--generation-dir",
                repo_relative(generation_dir),
                "--report",
                repo_relative(report_path),
                "--index",
                repo_relative(index_path),
                "--metrics",
                repo_relative(metrics_path),
            ]
        )
        if engine_kind != "qlora":
            add_numeric_arg(command, options, "max_steps", "--max-steps", int)
            add_numeric_arg(command, options, "max_length", "--max-length", int)
            add_numeric_arg(command, options, "learning_rate", "--learning-rate", float)
            add_numeric_arg(command, options, "alpha", "--alpha", int)
            add_numeric_arg(command, options, "dropout", "--dropout", float)

        if engine_kind in {"sft", "dpo"}:
            add_numeric_arg(command, options, "max_new_tokens", "--max-new-tokens", int)
        add_numeric_arg(command, options, "rank", "--rank", int)
        if engine_kind in {"sft", "peft"}:
            add_numeric_arg(command, options, "per_device_train_batch_size", "--per-device-train-batch-size", int)
            add_numeric_arg(command, options, "per_device_eval_batch_size", "--per-device-eval-batch-size", int)
            add_numeric_arg(command, options, "gradient_accumulation_steps", "--gradient-accumulation-steps", int)
            add_numeric_arg(command, options, "warmup_steps", "--warmup-steps", int)
            add_numeric_arg(command, options, "weight_decay", "--weight-decay", float)
            add_string_arg(command, options, "lr_scheduler_type", "--lr-scheduler-type")
            add_string_arg(command, options, "target_modules", "--target-modules")
        if engine_kind == "dpo":
            add_numeric_arg(command, options, "beta", "--beta", float)
            add_numeric_arg(command, options, "weight_decay", "--weight-decay", float)
            add_string_arg(command, options, "device", "--device")
            add_string_arg(command, options, "target_modules", "--target-modules")
            add_bool_arg(command, options, "quick", "--quick")
        if engine_kind == "qlora":
            add_numeric_arg(command, options, "seq_length", "--seq-length", int)
            add_numeric_arg(command, options, "per_device_batch_size", "--per-device-batch-size", int)
            add_numeric_arg(command, options, "gradient_accumulation_steps", "--gradient-accumulation-steps", int)
            add_numeric_arg(command, options, "target_module_count", "--target-module-count", int)
            add_bool_arg(command, options, "quick", "--quick")
            add_bool_arg(command, options, "skip_hf_metadata", "--skip-hf-metadata")
            add_bool_arg(command, options, "local_files_only", "--local-files-only")
            add_bool_arg(command, options, "no_gradient_checkpointing", "--no-gradient-checkpointing")
        if extra_args:
            command.extend(["--extra-args-json", json.dumps(extra_args, ensure_ascii=False)])

        studio_options = {
            **options,
            "method": method_key,
            "data_path": data_path,
            "data_summary": data_summary,
            "lesson_id": studio_id,
            "engine_kind": engine_kind,
            "run_dir": repo_relative(run_dir),
            "output_dir": repo_relative(output_dir),
            "adapter_dir": repo_relative(adapter_dir),
            "generation_dir": repo_relative(generation_dir),
            "report": repo_relative(report_path),
            "index": repo_relative(index_path),
            "metrics": metrics_display,
            "trace": repo_relative(trace_path),
            "extra_args": extra_args,
            "extra_args_path": repo_relative(extra_args_path) if extra_args else "",
        }
        state = self._start(command, str(method["job_type"]), studio_id, studio_options)
        profile = {
            "updated_at": now_iso(),
            "status": state.get("job", {}).get("status") if state.get("job") else "running",
            "job_id": state.get("job", {}).get("id") if state.get("job") else None,
            "method": method_key,
            "lesson_id": studio_id,
            "engine_kind": engine_kind,
            "model_name": model_name,
            "data_path": data_path,
            "data_summary": data_summary,
            "run_dir": repo_relative(run_dir),
            "output_dir": repo_relative(output_dir),
            "adapter_dir": repo_relative(adapter_dir),
            "generation_dir": repo_relative(generation_dir),
            "report": repo_relative(report_path),
            "index": repo_relative(index_path),
            "metrics": metrics_display,
            "trace": repo_relative(trace_path),
            "extra_args": extra_args,
            "extra_args_path": repo_relative(extra_args_path) if extra_args else "",
            "adapter_expected": method.get("artifact_kind") == "adapter",
            "source": "studio",
        }
        write_json(STUDIO_PROFILE_PATH, profile)
        return self.state()

    def start_all(self, options: dict[str, Any]) -> dict[str, Any]:
        command = [
            project_python(),
            "lessons/run_all.py",
            "--from-lesson",
            str(options.get("from_lesson") or "01"),
            "--to-lesson",
            str(options.get("to_lesson") or "10"),
            "--trace-delay",
            str(float(options.get("trace_delay") or 0.0)),
        ]
        if options.get("quick"):
            command.append("--quick")
        return self._start(command, "course-set", "all", options)

    def control(self, action: str) -> dict[str, Any]:
        with self.lock:
            payload = read_json(CONTROL_PATH, initial_control())
            action = action.lower()
            if action == "pause":
                payload["mode"] = "paused"
                payload["stop_requested"] = False
            elif action == "resume":
                payload["mode"] = "running"
                payload["stop_requested"] = False
            elif action == "step":
                payload["mode"] = "step"
                payload["step_token"] = int(payload.get("step_token") or 0) + 1
                payload["stop_requested"] = False
            elif action == "stop":
                payload["stop_requested"] = True
                payload["mode"] = "running"
                if self.job:
                    self.job["stop_requested"] = True
                self._terminate_locked()
            else:
                raise ValueError(f"Unknown control action: {action}")
            payload["updated_at"] = now_iso()
            write_json(CONTROL_PATH, payload)
            return self.state()

    def _start(
        self,
        command: list[str],
        job_type: str,
        lesson_id: str,
        options: dict[str, Any],
    ) -> dict[str, Any]:
        with self.lock:
            self._refresh_locked()
            if self.process and self.process.poll() is None:
                raise RuntimeError("A lesson job is already running.")

            write_json(CONTROL_PATH, initial_control())
            job_id = uuid.uuid4().hex[:12]
            log_path = RUNTIME_DIR / f"{job_id}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(f"$ {command_text(command)}\n\n", encoding="utf-8")

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["LLM_STUDY_CONTROL_PATH"] = str(CONTROL_PATH)
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            self.process = process
            self.job = {
                "id": job_id,
                "type": job_type,
                "lesson_id": lesson_id,
                "status": "running",
                "pid": process.pid,
                "command": command,
                "command_text": command_text(command),
                "options": options,
                "log_path": str(log_path),
                "started_at": now_iso(),
                "updated_at": now_iso(),
                "returncode": None,
            }
            thread = threading.Thread(target=self._pump_output, args=(process, log_path, job_id), daemon=True)
            thread.start()
            return self.state()

    def _pump_output(self, process: subprocess.Popen[str], log_path: Path, job_id: str) -> None:
        with log_path.open("a", encoding="utf-8") as log_file:
            assert process.stdout is not None
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
            returncode = process.wait()

        with self.lock:
            if not self.job or self.job.get("id") != job_id:
                return
            self.job["returncode"] = returncode
            self.job["status"] = "done" if returncode == 0 else "stopped" if self.job.get("stop_requested") else "failed"
            self.job["updated_at"] = now_iso()
            if str(self.job.get("type") or "").startswith("studio"):
                profile = read_json(STUDIO_PROFILE_PATH, {})
                if isinstance(profile, dict) and profile.get("job_id") == job_id:
                    profile["status"] = self.job["status"]
                    profile["returncode"] = returncode
                    profile["finished_at"] = self.job["updated_at"]
                    profile["updated_at"] = self.job["updated_at"]
                    write_json(STUDIO_PROFILE_PATH, profile)
            control = read_json(CONTROL_PATH, initial_control())
            control["mode"] = "running"
            control["stop_requested"] = False
            control["updated_at"] = now_iso()
            write_json(CONTROL_PATH, control)

    def _refresh_locked(self) -> None:
        if not self.job or not self.process:
            return
        returncode = self.process.poll()
        if returncode is None:
            self.job["status"] = read_json(CONTROL_PATH, initial_control()).get("mode", "running")
            self.job["updated_at"] = now_iso()
            return
        if self.job.get("returncode") is None:
            self.job["returncode"] = returncode
            self.job["status"] = "done" if returncode == 0 else "stopped" if self.job.get("stop_requested") else "failed"
            self.job["updated_at"] = now_iso()

    def _terminate_locked(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        self.process.terminate()
        deadline = time.time() + 4
        while time.time() < deadline:
            if self.process.poll() is not None:
                break
            time.sleep(0.1)
        if self.process.poll() is None:
            self.process.kill()


MANAGER = JobManager()


class ReusableThreadingTCPServer(ThreadingTCPServer):
    allow_reuse_address = True


class StudioHandler(SimpleHTTPRequestHandler):
    server_version = "LLMStudyStudio/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            self.send_json(HTTPStatus.OK, MANAGER.state())
            return
        if parsed.path == "/api/catalog":
            self.send_json(HTTPStatus.OK, {"courses": load_catalog()})
            return
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = self.read_body_json()
            if parsed.path == "/api/run":
                state = MANAGER.start_lesson(str(payload.get("lesson_id") or ""), payload)
                self.send_json(HTTPStatus.OK, state)
                return
            if parsed.path == "/api/run-all":
                state = MANAGER.start_all(payload)
                self.send_json(HTTPStatus.OK, state)
                return
            if parsed.path == "/api/studio/run":
                state = MANAGER.start_studio_sft(payload)
                self.send_json(HTTPStatus.OK, state)
                return
            if parsed.path == "/api/studio/validate-data":
                self.send_json(HTTPStatus.OK, preview_studio_data(payload))
                return
            if parsed.path == "/api/control":
                state = MANAGER.control(str(payload.get("action") or ""))
                self.send_json(HTTPStatus.OK, state)
                return
            if parsed.path == "/api/chat":
                self.send_json(HTTPStatus.OK, run_chat(payload))
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown API endpoint")
        except RuntimeError as exc:
            self.send_json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except ValueError as exc:
            self.send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except subprocess.TimeoutExpired as exc:
            self.send_json(HTTPStatus.GATEWAY_TIMEOUT, {"error": f"chat timed out: {exc}"})
        except Exception as exc:  # noqa: BLE001
            self.send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def read_body_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8"))

    def send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_chat(payload: dict[str, Any]) -> dict[str, Any]:
    profile = read_studio_profile()
    lesson_id = str(payload.get("lesson_id") or profile.get("lesson_id") or "07-sft-baseline")
    user_input = str(payload.get("input") or "").strip()
    if not user_input:
        raise ValueError("Chat input is required.")
    mode = str(payload.get("mode") or "both")
    if mode not in {"base", "adapter", "both", "triple"}:
        raise ValueError(f"Unknown chat mode: {mode}")
    model_name = str(payload.get("model_name") or profile.get("model_name") or "auto").strip()
    if not model_name:
        raise ValueError("model_name is required.")
    profile_adapter = profile.get("adapter_dir") if profile.get("adapter_expected") else ""
    source = str(payload.get("source") or "")
    studio_source = source.startswith("studio")
    adapter_dir = str(payload.get("adapter_dir") or profile_adapter or "").strip()
    if not adapter_dir and not studio_source:
        adapter_dir = "lessons/07-sft-baseline/outputs/adapter"
    if not adapter_dir and studio_source and mode in {"adapter", "both", "triple"}:
        raise ValueError("adapter_dir is required for Studio adapter comparison. Run Studio training or choose Base only.")
    if not adapter_dir:
        adapter_dir = "visualizer/runtime/no-adapter"
    if adapter_dir:
        adapter_dir = repo_relative(project_path(adapter_dir))
    instruction = str(payload.get("instruction") or DEFAULT_CHAT_INSTRUCTION).strip()
    if not instruction:
        raise ValueError("instruction is required.")
    max_new_tokens = int(payload.get("max_new_tokens") or 96)
    command = [
        project_python(),
        "training/engines/chat_compare.py",
        "--input",
        user_input,
        "--mode",
        mode,
        "--model-name",
        model_name,
        "--adapter-dir",
        adapter_dir,
        "--instruction",
        instruction,
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    started = time.time()
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=int(payload.get("timeout_seconds") or 900),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "chat compare failed").strip())
    data = json.loads(result.stdout)
    data["lesson_id"] = lesson_id
    data["compare_source"] = source or ("studio" if profile.get("adapter_dir") == adapter_dir else "manual")
    data["duration_seconds"] = round(time.time() - started, 3)
    data["stderr"] = result.stderr.strip()
    return data


def main() -> None:
    handler = partial(StudioHandler, directory=str(ROOT))
    with ReusableThreadingTCPServer(("127.0.0.1", PORT), handler) as server:
        print(f"Serving LLM Study Studio at http://127.0.0.1:{PORT}/visualizer/")
        print("Local API: /api/state, /api/run, /api/studio/run, /api/control, /api/chat")
        print("Press Ctrl-C to stop.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            with MANAGER.lock:
                MANAGER._terminate_locked()
            return


if __name__ == "__main__":
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    main()
