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

LESSON_DIR = Path(__file__).resolve().parent
LESSON_OUTPUTS = LESSON_DIR / "outputs"
os.environ["HF_HOME"] = str(LESSON_OUTPUTS / "hf-cache")
os.environ["HF_DATASETS_CACHE"] = str(LESSON_OUTPUTS / "hf-cache" / "datasets")
os.environ["HF_XET_CACHE"] = str(LESSON_OUTPUTS / "hf-cache" / "xet")

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

        这组对比能证明 adapter 路径生效，但不适合证明“训练效果明显”。原因是本课只有 5 条样本、6 个训练 step，而且 prompt 是模型本来就会回答的通用概念解释。更好的 SFT baseline 应使用结构化输出场景，例如客服工单路由到严格 JSON。

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

        ## PEFT 还有哪些具体方法

        PEFT 不是单一算法，而是一组参数高效微调方法。当前课程只执行 LoRA，因为它最常见，也最容易观察 adapter 的保存、加载和推理效果。

        | 方法 | 核心想法 | 学习优先级 |
        |---|---|---|
        | LoRA | 在目标线性层旁边加低秩 A/B 矩阵，只训练 adapter delta | 最高 |
        | AdaLoRA | 动态调整不同层的 rank，把参数预算给更重要的层 | 中 |
        | IA3 | 学习少量缩放向量，调节 attention/FFN 激活 | 中 |
        | Prompt Tuning | 冻结模型，只训练 soft prompt 向量 | 中 |
        | Prefix Tuning | 给每层 attention 加可训练 prefix key/value | 中 |
        | P-Tuning | 用可训练 prompt 表示或 prompt encoder 引导模型 | 中 |
        | LoHa / LoKr | LoRA 的 Hadamard/Kronecker 分解变体 | 低 |
        | OFT / BOFT | 用正交变换方式调整权重表示 | 低 |
        | X-LoRA | 用门控方式组合多个 LoRA adapter | 低 |
        | LayerNorm Tuning | 只训练 LayerNorm 等极少参数 | 低 |

        建议顺序：LoRA -> AdaLoRA / IA3 -> Prompt / Prefix / P-Tuning -> 多 adapter 组合。

        ## 和 Lesson 05 的区别

        - Lesson 05 手写 LoRA，目标是看清 A/B 低秩矩阵。
        - Lesson 06 使用 PEFT，目标是学习真实工程接口。
        - Lesson 05 不下载模型；Lesson 06 下载真实 Hugging Face base model。
        - Lesson 05 把 LoRA 插在 `lm_head`；Lesson 06 插在 Qwen attention 的 `q_proj/v_proj`。

        ## 下一步

        本课已经覆盖 adapter 保存、加载和重新加载后的输出对比，不需要再单独拆一个 Adapter Evaluation 必修课。下一课应补独立 SFT baseline：在没有 LoRA/PEFT 概念干扰的情况下，固定 eval prompts，比较 SFT 训练前后的回答变化。
        """
    ).strip()
    report = "\n".join(line[8:] if line.startswith("        ") else line for line in report.splitlines())
    report_path.write_text(report + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data", default="examples/sample_sft.jsonl")
    parser.add_argument("--report", default="lessons/06-peft-lora/report.md")
    parser.add_argument("--output-dir", default="lessons/06-peft-lora/outputs/trainer")
    parser.add_argument("--adapter-dir", default="lessons/06-peft-lora/outputs/adapter")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--trace", default="visualizer/traces/live.json")
    parser.add_argument("--trace-delay", type=float, default=0.0)
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
    trace = VisualTrace("06-peft-lora", "Lesson 06 · PEFT Qwen LoRA", args.trace, args.trace_delay)

    machine = detect_machine(torch)
    print("Step 0: selected model")
    print("machine:", machine)
    print("model:", args.model_name)
    trace.event(
        "select model",
        "setup",
        "根据本机配置选择真实 Hugging Face causal LM，并把缓存写入课程 outputs。",
        inputs={"machine": machine},
        outputs={"model_name": args.model_name, "hf_cache": os.environ["HF_HOME"]},
    )

    print("\nStep 1: load tokenizer and base model")
    tokenizer = load_tokenizer(AutoTokenizer, args.model_name)
    base_model = load_base_model(torch, AutoModelForCausalLM, args.model_name)
    trace.event(
        "load tokenizer and base model",
        "model",
        "加载真实 Qwen tokenizer 和 base model。此时模型仍是原始权重，尚未注入 LoRA。",
        inputs={"model_name": args.model_name},
        outputs={"tokenizer_vocab_size": len(tokenizer), "pad_token_id": tokenizer.pad_token_id},
        model={"base": "loaded", "adapter": "none", "cache": os.environ["HF_HOME"]},
    )
    generated_before = greedy_generate(
        torch,
        base_model,
        tokenizer,
        "### Instruction:\n解释什么是梯度累积\n\n### Response:\n",
    )

    print("\nStep 2: build SFT dataset")
    tokenized, trainer_dataset = prepare_dataset(tokenizer, args.data, args.max_length)
    print(tokenized)
    trace.event(
        "build SFT dataset",
        "data",
        "用真实 Qwen tokenizer 构造 SFT 训练张量，数据字段仍然是 input_ids、attention_mask、labels。",
        inputs={"data": args.data, "max_length": args.max_length},
        outputs={"train_rows": len(tokenized["train"]), "validation_rows": len(tokenized["validation"])},
        tensors=[
            {"name": "input_ids", "shape": [args.max_length]},
            {"name": "attention_mask", "shape": [args.max_length]},
            {"name": "labels", "shape": [args.max_length]},
        ],
    )

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
    trace.event(
        "attach PEFT LoRA",
        "model",
        "PEFT 在 q_proj/v_proj 上挂 LoRA adapter。base model 冻结，只训练少量 adapter 参数。",
        inputs={"target_modules": target_modules, "rank": args.rank, "alpha": args.alpha, "dropout": args.dropout},
        outputs={"trainable_params": trainable_params, "total_params": total_params},
        model={
            "base": "frozen",
            "adapter": "trainable",
            "target_modules": target_modules,
            "trainable_ratio": f"{trainable_params / total_params:.4%}",
        },
    )

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
        callbacks=[make_trainer_trace_callback(trace, "PEFT Trainer")],
    )

    print("\nStep 4: evaluate before PEFT LoRA training")
    eval_before = trainer.evaluate()
    print(eval_before)
    trace.event(
        "evaluate before PEFT LoRA training",
        "eval",
        "训练前测一次 validation loss，作为 Qwen + 未训练 adapter 的基线。",
        outputs={"eval_loss": eval_before.get("eval_loss")},
        metrics=eval_before,
    )

    print("\nStep 5: train PEFT LoRA adapter")
    train_output = trainer.train()
    print(train_output.metrics)
    trace.event(
        "train PEFT LoRA adapter",
        "train",
        "Trainer 只更新 q_proj/v_proj 上的 LoRA adapter，Qwen base 权重不更新。",
        inputs={"max_steps": args.max_steps, "learning_rate": args.learning_rate},
        outputs={"train_loss": train_output.metrics.get("train_loss")},
        metrics=train_output.metrics,
        model={"base": "frozen", "updated": ["q_proj LoRA", "v_proj LoRA"]},
    )

    print("\nStep 6: evaluate after PEFT LoRA training")
    eval_after = trainer.evaluate()
    print(eval_after)
    trace.event(
        "evaluate after PEFT LoRA training",
        "eval",
        "训练后再次评估，观察 adapter 对 validation loss 的影响。",
        outputs={"eval_loss": eval_after.get("eval_loss")},
        metrics=eval_after,
    )

    print("\nStep 7: save adapter")
    peft_model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print("adapter dir:", adapter_dir)
    trace.event(
        "save PEFT adapter",
        "checkpoint",
        "保存 adapter_config 和 adapter 权重，不保存完整 Qwen base model。",
        inputs={"peft_model": "trained"},
        outputs={"adapter_dir": adapter_dir},
        model={"saved": ["adapter_config.json", "adapter_model.safetensors"], "base_saved": False},
    )

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
    trace.event(
        "reload adapter into fresh Qwen",
        "checkpoint",
        "重新加载 fresh base，再用 PeftModel.from_pretrained 挂回 adapter，模拟真实推理部署。",
        inputs={"model_name": args.model_name, "adapter_dir": adapter_dir},
        outputs={"loaded_generation_matches_training_path": generated_loaded == generated_after},
        model={"base": "fresh Qwen", "adapter": "loaded"},
    )

    print("\nStep 9: fixed prompt generation")
    print("base:", generated_before)
    print("trained LoRA:", generated_after)
    print("loaded adapter:", generated_loaded)
    trace.event(
        "generation comparison",
        "generation",
        "对比 base、训练后 PEFT LoRA、重新加载 adapter 后的固定 prompt 输出。",
        outputs={
            "base": generated_before,
            "trained_lora": generated_after,
            "loaded_adapter": generated_loaded,
        },
    )

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
    trace.finish("Lesson 06 完成：真实 Qwen + PEFT LoRA 的训练、保存、加载流程已可视化。", metrics={"trainable_ratio": trainable_params / total_params})
    print(f"\nReport written: {report_path.relative_to(resolve_project_path('.'))}")


if __name__ == "__main__":
    main()
