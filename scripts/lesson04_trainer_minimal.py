#!/usr/bin/env python3
"""Lesson 04: run a minimal local Trainer loop on SFT-shaped data."""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from textwrap import dedent

from lesson_common import (
    count_trainable_parameters,
    decode_learned_labels,
    ensure_local_auto_tokenizer,
    load_sft_splits,
    make_sft_tokenize_fn,
    resolve_project_path,
)


def require_torch_and_trainer():
    try:
        import torch
        from torch import nn
        from transformers import Trainer, TrainingArguments, default_data_collator, set_seed
    except ImportError as exc:
        raise SystemExit(
            "Lesson 04 requires PyTorch and Trainer dependencies. "
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


def greedy_generate(torch, model, tokenizer, prompt: str, max_new_tokens: int = 48) -> str:
    model.eval()
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ids = torch.tensor([input_ids], dtype=torch.long)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids=ids)["logits"]
            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            ids = torch.cat([ids, torch.tensor([[next_id]], dtype=torch.long)], dim=1)
            if next_id == tokenizer.eos_token_id:
                break

    return tokenizer.decode(ids[0].tolist(), skip_special_tokens=False)


def write_report(
    report_path: Path,
    tokenizer,
    tokenized,
    trainable_params: int,
    total_params: int,
    eval_before: dict,
    train_metrics: dict,
    eval_after: dict,
    generated_text: str,
    max_steps: int,
    max_length: int,
) -> None:
    sample_labels = tokenized["train"][0]["labels"]

    report = dedent(
        f"""
        # Lesson 04: Trainer 最小训练闭环

        ## 本课执行结果

        - train 样本数: {len(tokenized["train"])}
        - validation 样本数: {len(tokenized["validation"])}
        - max_length: {max_length}
        - max_steps: {max_steps}
        - tokenizer vocab size: {len(tokenizer)}
        - tiny model trainable params: {trainable_params}
        - tiny model total params: {total_params}
        - eval loss before training: {eval_before.get("eval_loss")}
        - train loss: {train_metrics.get("train_loss")}
        - eval loss after training: {eval_after.get("eval_loss")}
        - 观察: train loss 下降但 eval loss 上升，这是 4 条训练样本上开始过拟合的信号。

        ## 第 1 条训练样本的 label 检查

        `decode(labels != -100)`:

        ```text
        {decode_learned_labels(tokenizer, sample_labels)}
        ```

        ## 训练后固定 prompt 生成

        ```text
        {generated_text}
        ```

        ## 每一步的作用、输入、输出

        | 步骤 | 作用 | 输入 | 输出 |
        |---|---|---|---|
        | 构造 tokenizer | 本地加载真实 `AutoTokenizer` | `.cache/local-sft-tokenizer` | token/id 映射 |
        | 构造 tokenized dataset | 复用 Lesson 02 的 SFT labels | JSONL + tokenizer | `input_ids/attention_mask/labels` |
        | `with_format("torch")` | 让 Dataset 取样时返回 torch tensor | tokenized DatasetDict | Trainer 可消费的 split |
        | tiny causal LM | 提供最小可训练模型 | vocab size | logits/loss |
        | `TrainingArguments` | 固定训练超参和输出目录 | batch、steps、lr、seed | 可复现实验设置 |
        | `Trainer.train()` | 执行反向传播和参数更新 | model + train_dataset | train loss 和训练状态 |
        | `Trainer.evaluate()` | 用 validation 观察训练效果 | eval_dataset | eval loss |

        ## 你要理解的关键点

        1. `Trainer` 不神秘，本质是把 model、dataset、collator、optimizer、评估循环组织起来。
        2. `labels` 仍然是核心，loss 只从非 `-100` 的位置来。
        3. 本课的 tiny model 不是可用 LLM，只是为了把训练闭环在本地跑通。
        4. `eval_loss` 是验证集上的下一个 token 预测损失，不等于回答质量。
        5. 后面换成真实 Transformer/LoRA 时，数据字段和 Trainer 闭环仍然是同一套。
        6. 训练后生成的英文混杂输出不是失败，而是在提醒你：tiny model + 5 条样本只能验证流程，不能期待语言能力。

        ## 下一步

        下一课可以进入 LoRA：冻结 base model，只训练少量 adapter 参数，并学习 adapter 保存、加载和合并。
        """
    ).strip()
    report = "\n".join(line[8:] if line.startswith("        ") else line for line in report.splitlines())
    report_path.write_text(report + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="examples/sample_sft.jsonl")
    parser.add_argument("--report", default="reports/lesson04-trainer.md")
    parser.add_argument("--tokenizer-dir", default=".cache/local-sft-tokenizer")
    parser.add_argument("--output-dir", default="outputs/lesson04-trainer")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    args = parser.parse_args()

    torch, nn, Trainer, TrainingArguments, default_data_collator, set_seed = require_torch_and_trainer()
    set_seed(42)

    report_path = resolve_project_path(args.report)
    output_dir = resolve_project_path(args.output_dir)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    model = build_tiny_causal_lm(torch, nn, len(tokenizer), tokenizer.pad_token_id)
    trainable_params, total_params = count_trainable_parameters(model)

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
    )

    print("Step 1: evaluate before training")
    eval_before = trainer.evaluate()
    print(eval_before)

    print("\nStep 2: train tiny causal LM with Trainer")
    train_output = trainer.train()
    print(train_output.metrics)

    print("\nStep 3: evaluate after training")
    eval_after = trainer.evaluate()
    print(eval_after)

    prompt = "### Instruction:\n解释什么是梯度累积\n\n### Response:\n"
    generated_text = greedy_generate(torch, model, tokenizer, prompt)
    print("\nStep 4: fixed prompt generation")
    print(generated_text)

    write_report(
        report_path,
        tokenizer,
        tokenized,
        trainable_params,
        total_params,
        eval_before,
        train_output.metrics,
        eval_after,
        generated_text,
        args.max_steps,
        args.max_length,
    )
    print(f"\nReport written: {report_path.relative_to(resolve_project_path('.'))}")


if __name__ == "__main__":
    main()
