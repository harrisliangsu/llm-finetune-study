#!/usr/bin/env python3
"""Structured trace writer for the local lesson visualizer."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        try:
            return str(value.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "shape"):
        try:
            return {"shape": list(value.shape), "dtype": str(getattr(value, "dtype", "unknown"))}
        except TypeError:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


class VisualTrace:
    """Append-only lesson execution trace rendered by `visualizer/index.html`."""

    def __init__(
        self,
        lesson_id: str,
        title: str,
        path: str | Path = "visualizer/traces/live.json",
        delay: float = 0.0,
    ) -> None:
        self.lesson_id = lesson_id
        self.title = title
        self.path = self._resolve(path)
        self.delay = max(0.0, delay)
        self.events: list[dict[str, Any]] = []
        self.started_at = _now_iso()
        self.status = "running"
        self.current_event_id: str | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.archive_path = self._archive_path()
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        self._write()

    def _resolve(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return PROJECT_ROOT / candidate

    def _archive_path(self) -> Path:
        safe_lesson_id = "".join(
            character if character.isalnum() or character in {"-", "_"} else "-"
            for character in self.lesson_id
        ).strip("-")
        return PROJECT_ROOT / "visualizer" / "traces" / f"{safe_lesson_id}.json"

    def event(
        self,
        name: str,
        kind: str,
        summary: str,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        tensors: list[dict[str, Any]] | None = None,
        model: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        sample: dict[str, Any] | None = None,
    ) -> None:
        event_id = f"{len(self.events) + 1:02d}"
        payload = {
            "id": event_id,
            "name": name,
            "kind": kind,
            "summary": summary,
            "time": _now_iso(),
            "inputs": _jsonable(inputs or {}),
            "outputs": _jsonable(outputs or {}),
            "tensors": _jsonable(tensors or []),
            "model": _jsonable(model or {}),
            "metrics": _jsonable(metrics or {}),
            "sample": _jsonable(sample or {}),
        }
        self.current_event_id = event_id
        self.events.append(payload)
        self._write()
        if self.delay:
            time.sleep(self.delay)

    def finish(self, summary: str, metrics: dict[str, Any] | None = None) -> None:
        self.status = "done"
        self.event(
            "finish",
            "complete",
            summary,
            outputs={"trace": self.path},
            metrics=metrics,
        )

    def _write(self) -> None:
        payload = {
            "schema_version": 1,
            "lesson_id": self.lesson_id,
            "title": self.title,
            "status": self.status,
            "started_at": self.started_at,
            "updated_at": _now_iso(),
            "current_event_id": self.current_event_id,
            "events": self.events,
        }
        self._write_payload(self.path, payload)
        if self.archive_path != self.path:
            self._write_payload(self.archive_path, payload)

    def _write_payload(self, path: Path, payload: dict[str, Any]) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(path)


def make_trainer_trace_callback(trace: VisualTrace, label: str = "Trainer"):
    """Create a Transformers Trainer callback that streams log/eval events."""

    from transformers import TrainerCallback

    class TraceCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
            if not logs:
                return
            trace.event(
                f"{label} log step {state.global_step}",
                "train",
                "Trainer 写出训练日志，页面用它观察 loss、learning rate 和梯度变化。",
                inputs={"global_step": state.global_step, "epoch": state.epoch},
                outputs={"log_keys": sorted(logs.keys())},
                metrics=logs,
            )

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):  # noqa: ANN001
            if not metrics:
                return
            trace.event(
                f"{label} eval step {state.global_step}",
                "eval",
                "Trainer 在 validation split 上评估当前模型。",
                inputs={"global_step": state.global_step, "epoch": state.epoch},
                outputs={"metric_keys": sorted(metrics.keys())},
                metrics=metrics,
            )

    return TraceCallback()
