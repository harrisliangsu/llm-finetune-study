#!/usr/bin/env python3
"""Lesson 06: run PEFT LoRA on a real Hugging Face causal LM.

Model choice is based on the local machine used for this course: Apple M2 Max,
32GB memory, and PyTorch MPS available. `Qwen/Qwen2.5-0.5B-Instruct` is small
enough for a short local LoRA run while still behaving like a real Chinese LLM.
"""

from __future__ import annotations

import argparse
import inspect
import os
import platform
import sys
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from lessons.common.lesson_common import (
    decode_learned_labels,
    load_sft_splits,
    make_sft_tokenize_fn,
    resolve_project_path,
)


os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / ".cache" / "huggingface" / "datasets"))


def require_training_stack():
    try:
        import torch
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            default_data_collator,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Lesson 06 requires torch, transformers, accelerate, and peft. "
            "Install missing packages in .venv before running this lesson."
        ) from exc

    return (
        torch,
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
        set_seed,
    )


def make_training_arguments(TrainingArguments, **kwargs):
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    return TrainingArguments(**{key: value for key, value in kwargs.items() if key in parameters})


def detect_machine(torch) -> dict[str, str | bool | int]:
    try:
        mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError):
        mem_bytes = 0
    return {
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "system": platform.system(),
        "memory_gb": round(mem_bytes / (1024**3)),
        "mps_available": torch.backends.mps.is_available(),
        "mps_built": torch.backends.mps.is_built(),
    }


def load_tokenizer(AutoTokenizer, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(torch, AutoModelForCausalLM, model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    return model


def prepare_dataset(tokenizer, data_path: str, max_length: int):
    splits = load_sft_splits(data_path)
    tokenized = splits.map(
        make_sft_tokenize_fn(tokenizer, max_length),
        batched=True,
        remove_columns=splits["train"].column_names,
    )
    trainer_dataset = tokenized.remove_columns(["prompt_length", "answer_length"]).with_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )
    return tokenized, trainer_dataset


def trainable_parameter_counts(model) -> tuple[int, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def greedy_generate(torch, model, tokenizer, prompt: str, max_new_tokens: int = 80) -> str:
    model.eval()
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(generated[0].detach().cpu().tolist(), skip_special_tokens=False)


def write_report(
    report_path: Path,
    model_name: str,
    machine: dict,
    tokenizer,
    tokenized,
    target_modules: list[str],
    rank: int,
    alpha: int,
    trainable_params: int,
    total_params: int,
    eval_before: dict,
    train_metrics: dict,
    eval_after: dict,
    adapter_dir: Path,
    generated_before: str,
    generated_after: str,
    generated_loaded: str,
    max_steps: int,
    max_length: int,
) -> None:
    ratio = trainable_params / total_params
    sample_labels = tokenized["train"][0]["labels"]
    report = dedent(
        f"""
        # Lesson 06: PEFT + 真实 Hugging Face 小模型 LoRA

        ## 本机配置和模型选择

        - system: {machine["system"]}
        - machine: {machine["machine"]}
        - memory: {machine["memory_gb"]} GB
        - MPS available: {machine["mps_available"]}
        - selected model: `{model_name}`

        选择 `{model_name}` 的原因：它是 0.5B 级别的真实中文/多语 instruct causal LM，
        在 Apple M2 Max + 32GB 内存上适合做很短的 LoRA 学习实验，比 `tiny-gpt2`
        更接近真实中文 LLM 微调工程。

        ## 本课执行结果

        - train 样本数: {len(tokenized["train"])}
        - validation 样本数: {len(tokenized["validation"])}
        - max_length: {max_length}
        - max_steps: {max_steps}
        - tokenizer vocab size: {len(tokenizer)}
        - target modules: `{target_modules}`
        - LoRA rank `r`: {rank}
        - LoRA alpha: {alpha}
        - trainable params: {trainable_params}
        - total params: {total_params}
        - trainable ratio: {ratio:.4%}
        - eval loss before training: {eval_before.get("eval_loss")}
        - train loss: {train_metrics.get("train_loss")}
        - eval loss after training: {eval_after.get("eval_loss")}
        - adapter dir: `{adapter_dir}`

        ## 第 1 条训练样本的 label 检查

        ```text
        {decode_learned_labels(tokenizer, sample_labels)}
        ```

        ## 固定 prompt 生成对比

        Base model 输出：

        ```text
        {generated_before}
        ```

        LoRA 训练后输出：

        ```text
        {generated_after}
        ```

        重新加载 adapter 后输出：

        ```text
        {generated_loaded}
        ```

        ## 每一步的作用、输入、输出

        | 步骤 | 作用 | 输入 | 输出 |
        |---|---|---|---|
        | `AutoTokenizer.from_pretrained` | 下载/加载真实模型 tokenizer | `{model_name}` | Qwen tokenizer |
        | `AutoModelForCausalLM.from_pretrained` | 下载/加载真实 causal LM | `{model_name}` | base model |
        | `LoraConfig` | 描述 LoRA 插入位置和超参 | target modules, r, alpha | PEFT 配置 |
        | `get_peft_model` | 把 base model 包成 LoRA model | base model + config | trainable adapter |
        | `Trainer.train` | 只训练 LoRA adapter | tokenized dataset | train/eval loss |
        | `save_pretrained` | 保存 adapter-only checkpoint | PEFT model | adapter dir |
        | `PeftModel.from_pretrained` | 把 adapter 加载到 fresh base | base model + adapter dir | 可推理 LoRA model |

        ## 和 Lesson 05 的区别

        - Lesson 05 手写 LoRA，目标是看清 A/B 低秩矩阵。
        - Lesson 06 使用 PEFT，目标是学习真实工程接口。
        - Lesson 05 不下载模型；Lesson 06 下载真实 Hugging Face base model。
        - Lesson 05 把 LoRA 插在 `lm_head`；Lesson 06 插在 Qwen attention 的 `q_proj/v_proj`。

        ## 下一步

        下一课可以做真实 adapter 管理：比较多个 adapter、加载不同 checkpoint、固定 eval prompts 做训练前后对比。
        """
    ).strip()
    report = "\n".join(line[8:] if line.startswith("        ") else line for line in report.splitlines())
    report_path.write_text(report + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default="examples/sample_sft.jsonl")
    parser.add_argument("--report", default="lessons/06-peft-lora/report.md")
    parser.add_argument("--output-dir", default="outputs/lesson06-peft-lora/trainer")
    parser.add_argument("--adapter-dir", default="outputs/lesson06-peft-lora/adapter")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    args = parser.parse_args()

    (
        torch,
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
        set_seed,
    ) = require_training_stack()
    set_seed(42)

    report_path = resolve_project_path(args.report)
    output_dir = resolve_project_path(args.output_dir)
    adapter_dir = resolve_project_path(args.adapter_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    machine = detect_machine(torch)
    print("Step 0: selected model")
    print("machine:", machine)
    print("model:", args.model_name)

    print("\nStep 1: load tokenizer and base model")
    tokenizer = load_tokenizer(AutoTokenizer, args.model_name)
    base_model = load_base_model(torch, AutoModelForCausalLM, args.model_name)
    generated_before = greedy_generate(
        torch,
        base_model,
        tokenizer,
        "### Instruction:\n解释什么是梯度累积\n\n### Response:\n",
    )

    print("\nStep 2: build SFT dataset")
    tokenized, trainer_dataset = prepare_dataset(tokenizer, args.data, args.max_length)
    print(tokenized)

    target_modules = ["q_proj", "v_proj"]
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=target_modules,
        bias="none",
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.config.use_cache = False

    trainable_params, total_params = trainable_parameter_counts(peft_model)
    print("\nStep 3: attach PEFT LoRA")
    print("target modules:", target_modules)
    print("trainable params:", trainable_params)
    print("total params:", total_params)
    print("trainable ratio:", f"{trainable_params / total_params:.4%}")

    training_args = make_training_arguments(
        TrainingArguments,
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        eval_steps=3,
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=trainer_dataset["train"],
        eval_dataset=trainer_dataset["validation"],
        data_collator=default_data_collator,
    )

    print("\nStep 4: evaluate before PEFT LoRA training")
    eval_before = trainer.evaluate()
    print(eval_before)

    print("\nStep 5: train PEFT LoRA adapter")
    train_output = trainer.train()
    print(train_output.metrics)

    print("\nStep 6: evaluate after PEFT LoRA training")
    eval_after = trainer.evaluate()
    print(eval_after)

    print("\nStep 7: save adapter")
    peft_model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print("adapter dir:", adapter_dir)

    generated_after = greedy_generate(
        torch,
        peft_model,
        tokenizer,
        "### Instruction:\n解释什么是梯度累积\n\n### Response:\n",
    )

    print("\nStep 8: reload adapter into fresh base model")
    fresh_base = load_base_model(torch, AutoModelForCausalLM, args.model_name)
    loaded_model = PeftModel.from_pretrained(fresh_base, str(adapter_dir))
    generated_loaded = greedy_generate(
        torch,
        loaded_model,
        tokenizer,
        "### Instruction:\n解释什么是梯度累积\n\n### Response:\n",
    )

    print("\nStep 9: fixed prompt generation")
    print("base:", generated_before)
    print("trained LoRA:", generated_after)
    print("loaded adapter:", generated_loaded)

    write_report(
        report_path,
        args.model_name,
        machine,
        tokenizer,
        tokenized,
        target_modules,
        args.rank,
        args.alpha,
        trainable_params,
        total_params,
        eval_before,
        train_output.metrics,
        eval_after,
        adapter_dir.relative_to(resolve_project_path(".")),
        generated_before,
        generated_after,
        generated_loaded,
        args.max_steps,
        args.max_length,
    )
    print(f"\nReport written: {report_path.relative_to(resolve_project_path('.'))}")


if __name__ == "__main__":
    main()
