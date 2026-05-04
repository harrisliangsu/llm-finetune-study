#!/usr/bin/env python3
"""Lesson 04: run a minimal Trainer loop with a real Hugging Face causal LM."""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

LESSON_DIR = Path(__file__).resolve().parent
LESSON_OUTPUTS = LESSON_DIR / "outputs"
os.environ["HF_HOME"] = str(LESSON_OUTPUTS / "hf-cache")
os.environ["HF_DATASETS_CACHE"] = str(LESSON_OUTPUTS / "hf-cache" / "datasets")
os.environ["HF_XET_CACHE"] = str(LESSON_OUTPUTS / "hf-cache" / "xet")

from lessons.common.hf_model_policy import DEFAULT_TINY_CAUSAL_LM, detect_local_config
from lessons.common.lesson_common import (
    decode_learned_labels,
    load_sft_splits,
    make_sft_tokenize_fn,
    resolve_project_path,
)
from lessons.common.visual_trace import VisualTrace, make_trainer_trace_callback


def require_training_stack():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator, set_seed
    except ImportError as exc:
        raise SystemExit(
            "Lesson 04 requires torch, transformers, datasets, and accelerate. "
            "Install them in .venv before running this lesson."
        ) from exc

    return torch, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator, set_seed


def make_training_arguments(TrainingArguments, **kwargs):
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    return TrainingArguments(**{key: value for key, value in kwargs.items() if key in parameters})


def load_tokenizer(AutoTokenizer, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(AutoModelForCausalLM, model_name: str):
    model = AutoModelForCausalLM.from_pretrained(model_name)
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


def count_trainable_parameters(model) -> tuple[int, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def greedy_generate(torch, model, tokenizer, prompt: str, max_new_tokens: int = 48) -> str:
    model.eval()
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
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
    trainable_params: int,
    total_params: int,
    eval_before: dict,
    train_metrics: dict,
    eval_after: dict,
    generated_before: str,
    generated_after: str,
    max_steps: int,
    max_length: int,
) -> None:
    sample_labels = tokenized["train"][0]["labels"]
    report = dedent(
        f"""
        # Lesson 04: Trainer 最小训练闭环

        ## 本课执行结果

        - model: `{model_name}`
        - system: {machine["system"]}
        - machine: {machine["machine"]}
        - memory: {machine["memory_gb"]} GB
        - MPS available: {machine["mps_available"]}
        - train 样本数: {len(tokenized["train"])}
        - validation 样本数: {len(tokenized["validation"])}
        - max_length: {max_length}
        - max_steps: {max_steps}
        - tokenizer vocab size: {len(tokenizer)}
        - trainable params: {trainable_params}
        - total params: {total_params}
        - trainable ratio: {trainable_params / total_params:.4%}
        - eval loss before training: {eval_before.get("eval_loss")}
        - train loss: {train_metrics.get("train_loss")}
        - eval loss after training: {eval_after.get("eval_loss")}

        本课不再手写 tiny model，而是从 Hugging Face 下载 `{model_name}`。
        它很小，适合快速看懂 `Trainer` 怎么组织 model、dataset、collator、loss、eval 和 checkpoint。

        ## 第 1 条训练样本的 label 检查

        `decode(labels != -100)`:

        ```text
        {decode_learned_labels(tokenizer, sample_labels)}
        ```

        这一步确认只有 answer 部分参与 loss，prompt 区域是 `-100`。

        ## 固定 prompt 训练前后生成

        Base 输出：

        ```text
        {generated_before}
        ```

        训练后输出：

        ```text
        {generated_after}
        ```

        ## 每一步的作用、输入、输出

        | 步骤 | 作用 | 输入 | 输出 |
        |---|---|---|---|
        | `AutoTokenizer.from_pretrained` | 从 HF 加载 tokenizer | `{model_name}` | token/id 映射 |
        | `AutoModelForCausalLM.from_pretrained` | 从 HF 加载 causal LM | `{model_name}` | 可训练 base model |
        | 构造 tokenized dataset | 把 SFT JSONL 变成训练字段 | JSONL + tokenizer | `input_ids/attention_mask/labels` |
        | `with_format("torch")` | 让 Dataset 返回 tensor | tokenized DatasetDict | Trainer 可消费的 split |
        | `TrainingArguments` | 固定训练超参和输出目录 | batch、steps、lr、seed | 可复现实验设置 |
        | `Trainer.evaluate()` | 训练前后在 validation 上测 loss | eval split | eval loss |
        | `Trainer.train()` | 执行 forward、loss、backward、optimizer step | model + train split | train loss 和更新后的权重 |

        ## 你要理解的关键点

        1. `Trainer` 本质是训练循环封装，不替你决定数据是否正确。
        2. `labels` 仍然是核心，loss 只从非 `-100` 的位置来。
        3. 本课模型很小，目标是验证训练闭环，不追求中文回答质量。
        4. 后面换成 Qwen + LoRA 时，数据字段和 Trainer 闭环仍是同一套。

        ## 下一步

        Lesson 05 继续手写 LoRA 机制；Lesson 06/07 使用真实 Hugging Face Qwen 模型执行 PEFT/SFT。
        """
    ).strip()
    report = "\n".join(line[8:] if line.startswith("        ") else line for line in report.splitlines())
    report_path.write_text(report + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_TINY_CAUSAL_LM)
    parser.add_argument("--data", default="examples/sample_sft.jsonl")
    parser.add_argument("--report", default="lessons/04-trainer/report.md")
    parser.add_argument("--output-dir", default="lessons/04-trainer/outputs/trainer")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--trace", default="visualizer/traces/live.json")
    parser.add_argument("--trace-delay", type=float, default=0.0)
    args = parser.parse_args()

    torch, AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator, set_seed = require_training_stack()
    set_seed(42)
    machine = detect_local_config(torch)

    report_path = resolve_project_path(args.report)
    output_dir = resolve_project_path(args.output_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = VisualTrace("04-trainer", "Lesson 04 · Trainer Loop", args.trace, args.trace_delay)

    print("Step 0: load Hugging Face model")
    print("machine:", machine)
    print("model:", args.model_name)
    tokenizer = load_tokenizer(AutoTokenizer, args.model_name)
    model = load_model(AutoModelForCausalLM, args.model_name)
    trainable_params, total_params = count_trainable_parameters(model)
    trace.event(
        "load HF causal LM",
        "model",
        "从 Hugging Face 加载 tiny causal LM。本课学习 Trainer 闭环，不再手写本地模型。",
        inputs={"model_name": args.model_name, "machine": machine},
        outputs={"tokenizer_vocab_size": len(tokenizer), "trainable_params": trainable_params, "total_params": total_params},
        model={"base": args.model_name, "adapter": "none", "trainable_ratio": f"{trainable_params / total_params:.4%}"},
    )

    print("\nStep 1: build SFT tensors")
    tokenized, trainer_dataset = prepare_dataset(tokenizer, args.data, args.max_length)
    trace.event(
        "prepare tokenized dataset",
        "data",
        "Trainer 需要的 train/eval Dataset 已准备好，每条样本都有 input_ids、attention_mask、labels。",
        inputs={"data": args.data, "max_length": args.max_length},
        outputs={"train_rows": len(tokenized["train"]), "validation_rows": len(tokenized["validation"])},
        tensors=[
            {"name": "input_ids", "shape": [args.max_length]},
            {"name": "attention_mask", "shape": [args.max_length]},
            {"name": "labels", "shape": [args.max_length]},
        ],
        sample={key: tokenized["train"][0][key] for key in ["prompt_length", "answer_length"]},
    )

    training_args = make_training_arguments(
        TrainingArguments,
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=5,
        eval_steps=10,
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=trainer_dataset["train"],
        eval_dataset=trainer_dataset["validation"],
        data_collator=default_data_collator,
        callbacks=[make_trainer_trace_callback(trace, "HF Trainer")],
    )

    prompt = "### Instruction:\n解释什么是梯度累积\n\n### Response:\n"
    generated_before = greedy_generate(torch, model, tokenizer, prompt)
    trace.event(
        "generate before training",
        "generation",
        "训练前先固定 prompt 生成一次，作为行为基线。",
        inputs={"prompt": prompt},
        outputs={"generated_text": generated_before},
    )

    print("\nStep 2: evaluate before training")
    eval_before = trainer.evaluate()
    print(eval_before)
    trace.event(
        "evaluate before training",
        "eval",
        "训练前先在 validation 上测一次，作为参数更新前的基线。",
        inputs={"eval_rows": len(tokenized["validation"])},
        outputs={"eval_loss": eval_before.get("eval_loss")},
        metrics=eval_before,
    )

    print("\nStep 3: train Hugging Face causal LM with Trainer")
    train_output = trainer.train()
    print(train_output.metrics)
    trace.event(
        "train HF causal LM",
        "train",
        "Trainer 完成 forward、loss、backward 和 optimizer step，HF 模型权重已经更新。",
        inputs={"max_steps": args.max_steps, "learning_rate": args.learning_rate},
        outputs={"train_loss": train_output.metrics.get("train_loss")},
        metrics=train_output.metrics,
        model={"updated": "full model", "adapter": "none"},
    )

    print("\nStep 4: evaluate after training")
    eval_after = trainer.evaluate()
    print(eval_after)
    trace.event(
        "evaluate after training",
        "eval",
        "训练后再次评估，观察 validation loss 是否改善或过拟合。",
        inputs={"eval_rows": len(tokenized["validation"])},
        outputs={"eval_loss": eval_after.get("eval_loss")},
        metrics=eval_after,
    )

    generated_after = greedy_generate(torch, model, tokenizer, prompt)
    print("\nStep 5: fixed prompt generation")
    print(generated_after)
    trace.event(
        "generate after training",
        "generation",
        "用同一个固定 prompt 对比训练前后输出。",
        inputs={"prompt": prompt},
        outputs={"before": generated_before, "after": generated_after},
    )

    write_report(
        report_path,
        args.model_name,
        machine,
        tokenizer,
        tokenized,
        trainable_params,
        total_params,
        eval_before,
        train_output.metrics,
        eval_after,
        generated_before,
        generated_after,
        args.max_steps,
        args.max_length,
    )
    trace.finish("Lesson 04 完成：真实 Hugging Face tiny causal LM 的 Trainer 闭环已经可视化。", metrics={"eval_loss_after": eval_after.get("eval_loss")})
    print(f"\nReport written: {report_path.relative_to(resolve_project_path('.'))}")


if __name__ == "__main__":
    main()
