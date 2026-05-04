"""Shared configuration for the standalone training studio."""

from __future__ import annotations


DEFAULT_CHAT_INSTRUCTION = "把用户工单路由成严格 JSON，字段为 intent、priority、department、summary。不要输出 JSON 以外的文字。"

STUDIO_METHODS = {
    "sft-lora": {
        "label": "SFT + LoRA",
        "lesson_id": "studio-sft-lora",
        "engine_kind": "sft",
        "engine_script": "training/engines/sft_lora.py",
        "default_data": "visualizer/studio/data/sft-example.jsonl",
        "eval_prompts": "visualizer/studio/data/sft-eval-prompts.jsonl",
        "schema": "sft",
        "job_type": "studio-sft",
        "artifact_kind": "adapter",
    },
    "peft-lora": {
        "label": "PEFT LoRA",
        "lesson_id": "studio-peft-lora",
        "engine_kind": "peft",
        "engine_script": "training/engines/peft_lora.py",
        "default_data": "visualizer/studio/data/sft-example.jsonl",
        "eval_prompts": "",
        "schema": "sft",
        "job_type": "studio-peft-lora",
        "artifact_kind": "adapter",
    },
    "dpo": {
        "label": "DPO Preference",
        "lesson_id": "studio-dpo-preference",
        "engine_kind": "dpo",
        "engine_script": "training/engines/dpo_preference.py",
        "default_data": "visualizer/studio/data/dpo-example.jsonl",
        "eval_prompts": "visualizer/studio/data/dpo-eval-prompts.jsonl",
        "schema": "dpo",
        "job_type": "studio-dpo",
        "artifact_kind": "adapter",
    },
    "qlora-plan": {
        "label": "QLoRA Plan",
        "lesson_id": "studio-qlora-plan",
        "engine_kind": "qlora",
        "engine_script": "training/engines/qlora_plan.py",
        "default_data": "",
        "eval_prompts": "",
        "config": "training/engines/data/qlora-planning-config.json",
        "schema": "none",
        "job_type": "studio-qlora-plan",
        "artifact_kind": "plan",
    },
}

JSONL_SCHEMAS = {
    "sft": {
        "required": {"instruction", "input", "output"},
        "min_rows": 4,
        "summary_field": "output",
        "help": "SFT/PEFT JSONL: 每行一个 object，必须包含 instruction、input、output；input 可以为空字符串。",
    },
    "dpo": {
        "required": {"instruction", "input", "chosen", "rejected"},
        "min_rows": 2,
        "summary_field": "chosen",
        "help": "DPO JSONL: 每行一个 object，必须包含 instruction、input、chosen、rejected；chosen 是偏好答案，rejected 是反例。",
    },
}
