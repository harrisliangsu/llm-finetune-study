#!/usr/bin/env python3
"""Lesson 05: train and reload a tiny LoRA adapter locally.

This lesson intentionally does not download a base model. It applies LoRA to
the tiny causal LM from Lesson 04 so the adapter mechanics are visible:
freeze base weights, train low-rank matrices, save adapter-only weights, and
load them back into a fresh base model.
"""

from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
LESSON_OUTPUTS = Path(__file__).resolve().parent / "outputs"
os.environ["HF_HOME"] = str(LESSON_OUTPUTS / "hf-cache")
os.environ["HF_DATASETS_CACHE"] = str(LESSON_OUTPUTS / "hf-cache" / "datasets")
os.environ["HF_XET_CACHE"] = str(LESSON_OUTPUTS / "hf-cache" / "xet")

from lessons.common.lesson_common import (
    count_trainable_parameters,
    decode_learned_labels,
    ensure_local_auto_tokenizer,
    load_sft_splits,
    make_sft_tokenize_fn,
    resolve_project_path,
)
from lessons.common.visual_trace import VisualTrace, make_trainer_trace_callback


def require_torch_and_trainer():
    try:
        import torch
        from torch import nn
        from transformers import Trainer, TrainingArguments, default_data_collator, set_seed
    except ImportError as exc:
        raise SystemExit(
            "Lesson 05 requires PyTorch and Trainer dependencies. "
            "Install them with: .venv/bin/python -m pip install torch accelerate"
        ) from exc

    return torch, nn, Trainer, TrainingArguments, default_data_collator, set_seed


def make_training_arguments(TrainingArguments, **kwargs):
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    return TrainingArguments(**{key: value for key, value in kwargs.items() if key in parameters})


def build_tiny_causal_lm(torch, nn, vocab_size: int, pad_token_id: int):
    class TinyCausalLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 64, padding_idx=pad_token_id)
            self.rnn = nn.GRU(64, 96, batch_first=True)
            self.lm_head = nn.Linear(96, vocab_size)

        def forward(self, input_ids, attention_mask=None, labels=None):
            hidden = self.embedding(input_ids)
            hidden, _ = self.rnn(hidden)
            logits = self.lm_head(hidden)
            loss = None

            if labels is not None:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))

            return {"loss": loss, "logits": logits}

    return TinyCausalLM()


def freeze_parameters(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def make_lora_linear(torch, nn):
    class LoRALinear(nn.Module):
        def __init__(self, base_linear, rank: int = 4, alpha: int = 8, dropout: float = 0.0) -> None:
            super().__init__()
            self.base = base_linear
            self.rank = rank
            self.alpha = alpha
            self.scaling = alpha / rank
            self.dropout = nn.Dropout(dropout)

            for parameter in self.base.parameters():
                parameter.requires_grad = False

            self.lora_A = nn.Parameter(torch.empty(rank, base_linear.in_features))
            self.lora_B = nn.Parameter(torch.zeros(base_linear.out_features, rank))
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

        def forward(self, values):
            base_output = self.base(values)
            lora_hidden = self.dropout(values).matmul(self.lora_A.transpose(0, 1))
            lora_output = lora_hidden.matmul(self.lora_B.transpose(0, 1))
            return base_output + lora_output * self.scaling

    return LoRALinear


def attach_lora_to_lm_head(torch, nn, model, rank: int, alpha: int, dropout: float):
    LoRALinear = make_lora_linear(torch, nn)
    freeze_parameters(model)
    model.lm_head = LoRALinear(model.lm_head, rank=rank, alpha=alpha, dropout=dropout)
    return model


def save_lora_adapter(torch, model, adapter_path: Path, target_module: str, rank: int, alpha: int) -> None:
    adapter_path.parent.mkdir(parents=True, exist_ok=True)
    module = getattr(model, target_module)
    torch.save(
        {
            "target_module": target_module,
            "rank": rank,
            "alpha": alpha,
            "lora_A": module.lora_A.detach().cpu(),
            "lora_B": module.lora_B.detach().cpu(),
        },
        adapter_path,
    )


def load_lora_adapter(torch, model, adapter_path: Path):
    payload = torch.load(adapter_path, map_location="cpu")
    module = getattr(model, payload["target_module"])
    module.lora_A.data.copy_(payload["lora_A"].to(module.lora_A.device))
    module.lora_B.data.copy_(payload["lora_B"].to(module.lora_B.device))
    return payload


def adapter_delta_norm(model) -> float:
    module = model.lm_head
    return float(module.lora_B.detach().norm().cpu().item())


def greedy_generate(torch, model, tokenizer, prompt: str, max_new_tokens: int = 48) -> str:
    model.eval()
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids=ids)["logits"]
            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
            if next_id == tokenizer.eos_token_id:
                break

    return tokenizer.decode(ids[0].detach().cpu().tolist(), skip_special_tokens=False)


def write_report(
    report_path: Path,
    tokenizer,
    tokenized,
    target_module: str,
    rank: int,
    alpha: int,
    trainable_params: int,
    total_params: int,
    trainable_ratio: float,
    eval_before: dict,
    train_metrics: dict,
    eval_after: dict,
    adapter_path: Path,
    delta_norm: float,
    generated_before: str,
    generated_after: str,
    generated_loaded: str,
    max_steps: int,
    max_length: int,
) -> None:
    sample_labels = tokenized["train"][0]["labels"]
    report = dedent(
        f"""
        # Lesson 05: LoRA Adapter 微调

        ## 是否下载模型

        本课没有下载模型。我们继续使用 Lesson 04 的 tiny causal LM 作为 base model，
        只在 `lm_head` 上插入 LoRA A/B 低秩矩阵。真实 LLM LoRA 后续需要下载 base model，
        但学习 LoRA 机制不需要一开始就下载大模型。

        ## 本课执行结果

        - train 样本数: {len(tokenized["train"])}
        - validation 样本数: {len(tokenized["validation"])}
        - max_length: {max_length}
        - max_steps: {max_steps}
        - tokenizer vocab size: {len(tokenizer)}
        - target module: `{target_module}`
        - LoRA rank `r`: {rank}
        - LoRA alpha: {alpha}
        - trainable params: {trainable_params}
        - total params: {total_params}
        - trainable ratio: {trainable_ratio:.4%}
        - eval loss before LoRA training: {eval_before.get("eval_loss")}
        - train loss: {train_metrics.get("train_loss")}
        - eval loss after LoRA training: {eval_after.get("eval_loss")}
        - LoRA B norm after training: {delta_norm}
        - adapter path: `{adapter_path}`

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
        | freeze base | 冻结原模型参数，避免 full fine-tune | tiny causal LM | base 参数 `requires_grad=False` |
        | attach LoRA | 在目标线性层旁边加 A/B 低秩矩阵 | `lm_head`, rank={rank}, alpha={alpha} | 只有 LoRA 参数可训练 |
        | count params | 验证参数高效微调 | model.parameters() | trainable ratio {trainable_ratio:.4%} |
        | Trainer.train() | 只更新 LoRA A/B | tokenized SFT dataset | train/eval loss |
        | save adapter | 只保存 LoRA 权重 | `lora_A`, `lora_B` | adapter-only checkpoint |
        | load adapter | 把 adapter 加载到新 base 上 | fresh base + adapter file | 可复用的 LoRA 模型 |

        ## 你要理解的关键点

        1. LoRA 不是复制一份模型重训，而是在目标线性层上学习一个低秩增量。
        2. base model 参数被冻结，训练成本来自少量 `lora_A/lora_B`。
        3. `r` 控制低秩瓶颈大小，`alpha/r` 控制 LoRA 增量缩放。
        4. adapter checkpoint 只保存 LoRA 参数，不保存完整 base model。
        5. 真实 LLM 里 target modules 通常是 `q_proj/v_proj/o_proj` 等注意力线性层；本课用 `lm_head` 是为了本地看清机制。

        ## 下一步

        下一课可以把自定义 LoRA 换成 `peft`，并用一个真正的小型 Hugging Face causal LM 做 adapter 训练。
        """
    ).strip()
    report = "\n".join(line[8:] if line.startswith("        ") else line for line in report.splitlines())
    report_path.write_text(report + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="examples/sample_sft.jsonl")
    parser.add_argument("--report", default="lessons/05-lora/report.md")
    parser.add_argument("--tokenizer-dir", default="lessons/02-tokenizer/outputs/local-sft-tokenizer")
    parser.add_argument("--output-dir", default="lessons/05-lora/outputs/lora")
    parser.add_argument("--adapter-path", default="lessons/05-lora/outputs/lora/lora_adapter.pt")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--trace", default="visualizer/traces/live.json")
    parser.add_argument("--trace-delay", type=float, default=0.0)
    args = parser.parse_args()

    torch, nn, Trainer, TrainingArguments, default_data_collator, set_seed = require_torch_and_trainer()
    set_seed(42)

    report_path = resolve_project_path(args.report)
    output_dir = resolve_project_path(args.output_dir)
    adapter_path = resolve_project_path(args.adapter_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace = VisualTrace("05-lora", "Lesson 05 · Handwritten LoRA", args.trace, args.trace_delay)

    tokenizer = ensure_local_auto_tokenizer(args.data, args.tokenizer_dir)
    splits = load_sft_splits(args.data)
    tokenized = splits.map(
        make_sft_tokenize_fn(tokenizer, args.max_length),
        batched=True,
        remove_columns=splits["train"].column_names,
    )
    trainer_dataset = tokenized.remove_columns(["prompt_length", "answer_length"]).with_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )
    trace.event(
        "prepare SFT tensors",
        "data",
        "复用前面课程的 SFT 数据字段，LoRA 只改变模型训练方式，不改变数据格式。",
        inputs={"data": args.data, "max_length": args.max_length},
        outputs={"train_rows": len(tokenized["train"]), "validation_rows": len(tokenized["validation"])},
        tensors=[
            {"name": "input_ids", "shape": [args.max_length]},
            {"name": "labels", "shape": [args.max_length]},
        ],
    )

    prompt = "### Instruction:\n解释什么是梯度累积\n\n### Response:\n"
    target_module = "lm_head"

    base_model = build_tiny_causal_lm(torch, nn, len(tokenizer), tokenizer.pad_token_id)
    generated_before = greedy_generate(torch, base_model, tokenizer, prompt)

    model = attach_lora_to_lm_head(
        torch,
        nn,
        base_model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )
    trainable_params, total_params = count_trainable_parameters(model)
    trainable_ratio = trainable_params / total_params
    trace.event(
        "freeze base and attach LoRA",
        "model",
        "冻结 tiny base，在 lm_head 旁边挂 LoRA A/B。后续 optimizer 只更新 adapter。",
        inputs={"target_module": target_module, "rank": args.rank, "alpha": args.alpha},
        outputs={"trainable_params": trainable_params, "total_params": total_params},
        model={
            "base": "frozen",
            "adapter": "trainable",
            "target_module": target_module,
            "trainable_ratio": f"{trainable_ratio:.4%}",
            "delta_formula": "base_logits + lora_B(lora_A(x)) * alpha/r",
        },
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
        callbacks=[make_trainer_trace_callback(trace, "LoRA Trainer")],
    )

    print("Step 1: freeze base and attach LoRA")
    print("target module:", target_module)
    print("trainable params:", trainable_params)
    print("total params:", total_params)
    print("trainable ratio:", f"{trainable_ratio:.4%}")

    print("\nStep 2: evaluate before LoRA training")
    eval_before = trainer.evaluate()
    print(eval_before)
    trace.event(
        "evaluate before LoRA training",
        "eval",
        "训练 adapter 前先测 validation loss，作为 LoRA delta 还没学习时的基线。",
        outputs={"eval_loss": eval_before.get("eval_loss")},
        metrics=eval_before,
    )

    print("\nStep 3: train LoRA adapter with Trainer")
    train_output = trainer.train()
    print(train_output.metrics)
    trace.event(
        "train LoRA adapter",
        "train",
        "Trainer 反向传播时梯度只写入 lora_A/lora_B，base 权重保持不变。",
        inputs={"max_steps": args.max_steps, "learning_rate": args.learning_rate},
        outputs={"train_loss": train_output.metrics.get("train_loss")},
        metrics=train_output.metrics,
        model={"base": "frozen", "updated": ["lora_A", "lora_B"]},
    )

    print("\nStep 4: evaluate after LoRA training")
    eval_after = trainer.evaluate()
    print(eval_after)
    trace.event(
        "evaluate after LoRA training",
        "eval",
        "训练后观察 adapter 对 validation loss 的影响。",
        outputs={"eval_loss": eval_after.get("eval_loss")},
        metrics=eval_after,
    )

    print("\nStep 5: save adapter-only checkpoint")
    save_lora_adapter(torch, model, adapter_path, target_module, args.rank, args.alpha)
    delta_norm = adapter_delta_norm(model)
    print("adapter path:", adapter_path)
    print("LoRA B norm:", delta_norm)
    trace.event(
        "save adapter-only checkpoint",
        "checkpoint",
        "只保存 LoRA adapter 参数，不保存完整 tiny base model。",
        inputs={"target_module": target_module},
        outputs={"adapter_path": adapter_path, "lora_B_norm": delta_norm},
        model={"saved": ["lora_A", "lora_B"], "base_saved": False},
    )

    generated_after = greedy_generate(torch, model, tokenizer, prompt)

    print("\nStep 6: load adapter into a fresh base model")
    set_seed(42)
    loaded_model = build_tiny_causal_lm(torch, nn, len(tokenizer), tokenizer.pad_token_id)
    loaded_model = attach_lora_to_lm_head(
        torch,
        nn,
        loaded_model,
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
    )
    load_lora_adapter(torch, loaded_model, adapter_path)
    generated_loaded = greedy_generate(torch, loaded_model, tokenizer, prompt)
    trace.event(
        "reload adapter into fresh base",
        "checkpoint",
        "重新创建 fresh base，再加载 adapter，验证 adapter 可以单独复用。",
        inputs={"adapter_path": adapter_path},
        outputs={"loaded_generation_matches_training_path": generated_loaded == generated_after},
        model={"base": "fresh", "adapter": "loaded"},
    )

    print("\nStep 7: fixed prompt generation")
    print("base:", generated_before)
    print("trained LoRA:", generated_after)
    print("loaded adapter:", generated_loaded)
    trace.event(
        "generation comparison",
        "generation",
        "对比 base、训练后 LoRA、重新加载 adapter 后的固定 prompt 输出。",
        inputs={"prompt": prompt},
        outputs={
            "base": generated_before,
            "trained_lora": generated_after,
            "loaded_adapter": generated_loaded,
        },
    )

    write_report(
        report_path,
        tokenizer,
        tokenized,
        target_module,
        args.rank,
        args.alpha,
        trainable_params,
        total_params,
        trainable_ratio,
        eval_before,
        train_output.metrics,
        eval_after,
        adapter_path.relative_to(resolve_project_path(".")),
        delta_norm,
        generated_before,
        generated_after,
        generated_loaded,
        args.max_steps,
        args.max_length,
    )
    trace.finish("Lesson 05 完成：手写 LoRA 的数据流、模型变化和 adapter 保存加载已可视化。", metrics={"trainable_ratio": trainable_ratio})
    print(f"\nReport written: {report_path.relative_to(resolve_project_path('.'))}")


if __name__ == "__main__":
    main()
