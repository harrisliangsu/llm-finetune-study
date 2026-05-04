#!/usr/bin/env python3
"""Local hardware aware Hugging Face model defaults for the lessons."""

from __future__ import annotations

import os
import platform
from typing import Any


DEFAULT_TINY_CAUSAL_LM = "sshleifer/tiny-gpt2"
DEFAULT_LOCAL_INSTRUCT_LM = "Qwen/Qwen2.5-0.5B-Instruct"


def detect_local_config(torch: Any | None = None) -> dict[str, Any]:
    try:
        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError):
        mem_bytes = 0

    mps_available = False
    mps_built = False
    if torch is not None:
        mps_available = bool(torch.backends.mps.is_available())
        mps_built = bool(torch.backends.mps.is_built())

    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "memory_gb": round(mem_bytes / (1024**3)) if mem_bytes else 0,
        "mps_available": mps_available,
        "mps_built": mps_built,
    }


def choose_instruct_causal_lm(machine: dict[str, Any]) -> str:
    """Choose a real HF instruct model that is reasonable for this Mac.

    The user's current machine is arm64 with 32GB memory and MPS. A 0.5B Qwen
    instruct model is small enough for short LoRA/SFT lessons while still being
    a real Chinese-capable LLM. Smaller fallback keeps the course executable on
    weaker machines.
    """

    if machine.get("memory_gb", 0) >= 24:
        return DEFAULT_LOCAL_INSTRUCT_LM
    return DEFAULT_TINY_CAUSAL_LM


def resolve_model_name(requested: str, machine: dict[str, Any]) -> str:
    if requested == "auto":
        return choose_instruct_causal_lm(machine)
    return requested


def infer_lora_target_modules(model_name: str) -> list[str]:
    """Pick conservative LoRA targets for the small HF models used here."""

    normalized = model_name.lower()
    if "qwen" in normalized:
        return ["q_proj", "v_proj"]
    if "gpt2" in normalized:
        return ["c_attn"]
    return ["q_proj", "v_proj"]
