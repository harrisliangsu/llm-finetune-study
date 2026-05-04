#!/usr/bin/env python3
"""Standalone entrypoint for training-studio runs.

The studio owns method selection, data selection, and runtime artifacts. The
shared training engines live under training/ so the learning page and the
studio do not call into each other's folders.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from config import STUDIO_METHODS


ROOT = Path(__file__).resolve().parents[2]
EXTRA_ARG_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
RESERVED_EXTRA_ARGS = {
    "method",
    "model-name",
    "data",
    "trace",
    "trace-delay",
    "output-dir",
    "adapter-dir",
    "generation-dir",
    "report",
    "index",
    "metrics",
    "config",
    "eval-prompts",
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


def add_optional(command: list[str], value: object, flag: str) -> None:
    if value in (None, ""):
        return
    command.extend([flag, str(value)])


def add_flag(command: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        command.append(flag)


def extra_args_json_to_cli(raw: str) -> list[str]:
    if not raw.strip():
        return []
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--extra-args-json invalid JSON: {exc.msg}") from exc
    if not isinstance(config, dict) or isinstance(config, list):
        raise SystemExit("--extra-args-json must be a JSON object.")
    normalized: dict[str, object] = {}
    for key, value in config.items():
        flag_name = str(key).strip().replace("_", "-")
        if not EXTRA_ARG_NAME_RE.match(flag_name):
            raise SystemExit(f"Invalid extra arg name: {key}")
        if flag_name in RESERVED_EXTRA_ARGS:
            raise SystemExit(f"Extra arg is managed by Studio and cannot be overridden: {key}")
        normalized[flag_name] = value
    return ["--extra-args-json", json.dumps(normalized, ensure_ascii=False)]


def build_command(args: argparse.Namespace) -> list[str]:
    method = STUDIO_METHODS[args.method]
    engine_kind = str(method["engine_kind"])
    command = [
        sys.executable,
        str(method["engine_script"]),
        "--trace",
        args.trace,
        "--trace-delay",
        str(args.trace_delay),
        "--model-name",
        args.model_name,
    ]
    if method["schema"] != "none":
        if not args.data:
            raise SystemExit(f"--data is required for {args.method}")
        command.extend(["--data", args.data])
    if method.get("eval_prompts"):
        command.extend(["--eval-prompts", str(method["eval_prompts"])])
    if method.get("config"):
        command.extend(["--config", str(method["config"])])

    if engine_kind == "peft":
        command.extend(
            [
                "--output-dir",
                args.output_dir,
                "--adapter-dir",
                args.adapter_dir,
                "--report",
                args.report,
            ]
        )
    elif engine_kind in {"sft", "dpo"}:
        command.extend(
            [
                "--output-dir",
                args.output_dir,
                "--adapter-dir",
                args.adapter_dir,
                "--generation-dir",
                args.generation_dir,
                "--report",
                args.report,
                "--index",
                args.index,
            ]
        )
        if engine_kind == "sft":
            command.extend(["--metrics", args.metrics])
    elif engine_kind == "qlora":
        command.extend(
            [
                "--output-dir",
                args.output_dir,
                "--report",
                args.report,
                "--index",
                args.index,
            ]
        )

    if engine_kind != "qlora":
        add_optional(command, args.max_steps, "--max-steps")
        add_optional(command, args.max_length, "--max-length")
        add_optional(command, args.learning_rate, "--learning-rate")
        add_optional(command, args.alpha, "--alpha")
        add_optional(command, args.dropout, "--dropout")
    if engine_kind in {"sft", "dpo"}:
        add_optional(command, args.max_new_tokens, "--max-new-tokens")
    add_optional(command, args.rank, "--rank")
    if engine_kind in {"sft", "peft"}:
        add_optional(command, args.per_device_train_batch_size, "--per-device-train-batch-size")
        add_optional(command, args.per_device_eval_batch_size, "--per-device-eval-batch-size")
        add_optional(command, args.gradient_accumulation_steps, "--gradient-accumulation-steps")
        add_optional(command, args.warmup_steps, "--warmup-steps")
        add_optional(command, args.weight_decay, "--weight-decay")
        add_optional(command, args.lr_scheduler_type, "--lr-scheduler-type")
        add_optional(command, args.target_modules, "--target-modules")
    if engine_kind == "dpo":
        add_optional(command, args.beta, "--beta")
        add_optional(command, args.weight_decay, "--weight-decay")
        add_optional(command, args.device, "--device")
        add_optional(command, args.target_modules, "--target-modules")
        add_flag(command, args.quick, "--quick")
    if engine_kind == "qlora":
        add_optional(command, args.seq_length, "--seq-length")
        add_optional(command, args.per_device_batch_size, "--per-device-batch-size")
        add_optional(command, args.gradient_accumulation_steps, "--gradient-accumulation-steps")
        add_optional(command, args.target_module_count, "--target-module-count")
        add_flag(command, args.quick, "--quick")
        add_flag(command, args.skip_hf_metadata, "--skip-hf-metadata")
        add_flag(command, args.local_files_only, "--local-files-only")
        add_flag(command, args.no_gradient_checkpointing, "--no-gradient-checkpointing")
    command.extend(extra_args_json_to_cli(args.extra_args_json))
    return command


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(STUDIO_METHODS), required=True)
    parser.add_argument("--model-name", default="auto")
    parser.add_argument("--data", default="")
    parser.add_argument("--trace", default="visualizer/runtime/studio-manual-trace.json")
    parser.add_argument("--trace-delay", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--generation-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--alpha", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--per-device-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--lr-scheduler-type", default="")
    parser.add_argument("--target-modules", default="")
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--device", default="")
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--target-module-count", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-hf-metadata", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--extra-args-json", default="")
    args = parser.parse_args()

    command = build_command(args)
    print("$ " + shlex.join(command), flush=True)
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
