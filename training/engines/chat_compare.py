#!/usr/bin/env python3
"""Compare a base model and a studio adapter for one input."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ENGINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ENGINE_DIR.parents[1]
RUN_PATH = ENGINE_DIR / "sft_lora.py"
DEFAULT_INSTRUCTION = "把用户工单路由成严格 JSON，字段为 intent、priority、department、summary。不要输出 JSON 以外的文字。"


def load_engine_module():
    spec = importlib.util.spec_from_file_location("studio_sft_lora", RUN_PATH)
    if not spec or not spec.loader:
        raise RuntimeError(f"Unable to load {RUN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_result(target: str, generated: str, engine: Any, expected: dict[str, str]) -> dict[str, Any]:
    return {
        "target": target,
        "generated": generated,
        "answer": engine.strip_prompt(generated),
        "score": engine.score_generation(generated, expected),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--mode", choices=["base", "adapter", "both", "triple"], default="both")
    parser.add_argument("--model-name", default="auto")
    parser.add_argument("--adapter-dir", default="visualizer/runtime/no-adapter")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args()

    engine = load_engine_module()
    (
        torch,
        _LoraConfig,
        PeftModel,
        _TaskType,
        _get_peft_model,
        AutoModelForCausalLM,
        AutoTokenizer,
        _Trainer,
        _TrainingArguments,
        _default_data_collator,
        set_seed,
    ) = engine.require_training_stack()
    set_seed(42)

    machine = engine.detect_local_config(torch)
    model_name = engine.resolve_model_name(args.model_name, machine)
    tokenizer = engine.load_tokenizer(AutoTokenizer, model_name)
    prompt = engine.build_prompt(args.instruction, args.input)
    expected: dict[str, str] = {}
    results: list[dict[str, Any]] = []

    base_model = engine.load_base_model(torch, AutoModelForCausalLM, model_name)
    if args.mode in {"base", "both", "triple"}:
        generated = engine.greedy_generate(
            torch,
            base_model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
        )
        results.append(build_result("base", generated, engine, expected))

    adapter_path = engine.resolve_project_path(args.adapter_dir)
    adapter_available = adapter_path.exists() and (adapter_path / "adapter_config.json").exists()
    if args.mode in {"adapter", "both", "triple"}:
        if adapter_available:
            adapter_model = PeftModel.from_pretrained(base_model, str(adapter_path))
            generated = engine.greedy_generate(
                torch,
                adapter_model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
            )
            results.append(build_result("adapter", generated, engine, expected))
            if args.mode == "triple":
                del adapter_model
                del base_model
                gc.collect()
                if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                fresh_base = engine.load_base_model(torch, AutoModelForCausalLM, model_name)
                loaded_model = PeftModel.from_pretrained(fresh_base, str(adapter_path))
                generated = engine.greedy_generate(
                    torch,
                    loaded_model,
                    tokenizer,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                )
                results.append(build_result("reloaded_adapter", generated, engine, expected))
        else:
            results.append(
                {
                    "target": "adapter",
                    "error": f"Adapter is not available at {adapter_path.relative_to(PROJECT_ROOT)}.",
                    "answer": "",
                    "score": {},
                }
            )

    print(
        json.dumps(
            {
                "lesson_id": "studio-chat-compare",
                "model_name": model_name,
                "adapter_dir": str(adapter_path.relative_to(PROJECT_ROOT)),
                "adapter_available": adapter_available,
                "adapter_config_file": str((adapter_path / "adapter_config.json").relative_to(PROJECT_ROOT)) if adapter_available else None,
                "instruction": args.instruction,
                "input": args.input,
                "prompt": prompt,
                "mode": args.mode,
                "max_new_tokens": args.max_new_tokens,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    main()
