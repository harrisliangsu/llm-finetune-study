"""Helpers for Studio user-supplied extra training configuration."""

from __future__ import annotations

import inspect
import json
from typing import Any, Callable


def load_extra_args(raw: str) -> dict[str, Any]:
    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--extra-args-json invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict) or isinstance(payload, list):
        raise SystemExit("--extra-args-json must be a JSON object.")
    return {normalize_key(key): value for key, value in payload.items()}


def normalize_key(key: object) -> str:
    return str(key).strip().replace("-", "_")


def training_argument_overrides(TrainingArguments: type, raw: str) -> dict[str, Any]:
    overrides = load_extra_args(raw)
    if not overrides:
        return {}
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    unsupported = sorted(key for key in overrides if key not in parameters)
    if unsupported:
        names = ", ".join(f"--{key.replace('_', '-')}" for key in unsupported)
        raise SystemExit(f"Extra args are not supported by transformers.TrainingArguments: {names}")
    return overrides


def known_extra_args(
    raw: str,
    spec: dict[str, Callable[[Any], Any]],
    *,
    context: str,
) -> dict[str, Any]:
    overrides = load_extra_args(raw)
    if not overrides:
        return {}
    unsupported = sorted(key for key in overrides if key not in spec)
    if unsupported:
        names = ", ".join(f"--{key.replace('_', '-')}" for key in unsupported)
        raise SystemExit(f"Extra args are not supported by {context}: {names}")
    result: dict[str, Any] = {}
    for key, value in overrides.items():
        try:
            result[key] = spec[key](value)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"Extra arg --{key.replace('_', '-')} has invalid value: {value!r}") from exc
    return result
