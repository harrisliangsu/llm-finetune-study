#!/usr/bin/env python3
"""Lesson 03: turn tokenized SFT rows into a training batch."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from textwrap import dedent

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
LESSON_OUTPUTS = Path(__file__).resolve().parent / "outputs"
os.environ["HF_HOME"] = str(LESSON_OUTPUTS / "hf-cache")
os.environ["HF_DATASETS_CACHE"] = str(LESSON_OUTPUTS / "hf-cache" / "datasets")
os.environ["HF_XET_CACHE"] = str(LESSON_OUTPUTS / "hf-cache" / "xet")

from lessons.common.lesson_common import (
    IGNORE_INDEX,
    decode_learned_labels,
    ensure_local_auto_tokenizer,
    load_sft_splits,
    make_sft_tokenize_fn,
    resolve_project_path,
)


def sft_numpy_collator(features: list[dict]) -> dict[str, np.ndarray]:
    keys = ["input_ids", "attention_mask", "labels"]
    return {key: np.asarray([feature[key] for feature in features], dtype=np.int64) for key in keys}


def write_report(
    report_path: Path,
    tokenizer,
    tokenized,
    batch: dict[str, np.ndarray],
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    max_length: int,
) -> None:
    labels = batch["labels"][0].tolist()
    learned_label_ids = [token_id for token_id in labels if token_id != IGNORE_INDEX]
    effective_batch_size = micro_batch_size * gradient_accumulation_steps

    report = dedent(
        f"""
        # Lesson 03: Batch、Collator 和 Padding

        ## 本课执行结果

        - train 样本数: {len(tokenized["train"])}
        - validation 样本数: {len(tokenized["validation"])}
        - max_length: {max_length}
        - micro batch size: {micro_batch_size}
        - gradient_accumulation_steps: {gradient_accumulation_steps}
        - effective batch size: {effective_batch_size}
        - batch `input_ids.shape`: {tuple(batch["input_ids"].shape)}
        - batch `attention_mask.shape`: {tuple(batch["attention_mask"].shape)}
        - batch `labels.shape`: {tuple(batch["labels"].shape)}
        - 第 1 条样本参与 loss 的 token 数: {len(learned_label_ids)}

        ## batch 中的三个核心张量

        ```text
        input_ids      -> {tuple(batch["input_ids"].shape)}
        attention_mask -> {tuple(batch["attention_mask"].shape)}
        labels         -> {tuple(batch["labels"].shape)}
        ```

        第一维是 batch 维度，第二维是 sequence length。本课是 `{micro_batch_size} x {max_length}`。

        ## 第一条样本的局部预览

        `input_ids[0][:32]`:

        ```text
        {batch["input_ids"][0][:32].tolist()}
        ```

        `attention_mask[0][:32]`:

        ```text
        {batch["attention_mask"][0][:32].tolist()}
        ```

        `labels[0][:32]`:

        ```text
        {batch["labels"][0][:32].tolist()}
        ```

        `decode(labels[0] != -100)`:

        ```text
        {decode_learned_labels(tokenizer, labels)}
        ```

        ## 每一步的作用、输入、输出

        | 步骤 | 作用 | 输入 | 输出 |
        |---|---|---|---|
        | `map(tokenize_fn)` | 把原始文本行变成定长训练字段 | `instruction/input/output` | 每条样本都有 `input_ids/attention_mask/labels` |
        | `select(range(batch_size))` | 取一个 micro batch 做演示 | tokenized train split | 2 条样本列表 |
        | `collator(features)` | 把样本列表堆叠成数组 | list of dict | dict of numpy arrays |
        | `gradient_accumulation_steps` | 多个 micro batch 后再更新参数 | micro batch loss | 更大的 effective batch |

        ## 你要理解的关键点

        1. dataset 里是一条条样本，训练时必须 collate 成 batch。
        2. batch 的形状通常是 `[batch_size, sequence_length]`。
        3. padding 让同一个 batch 中的样本长度一致。
        4. `attention_mask` 让模型知道哪些位置是 padding。
        5. `labels=-100` 和 padding 是两件事：前者控制 loss，后者控制有效 token。

        ## 下一步

        Lesson 04 会把这个 batch 喂给 `Trainer`，用一个本地 tiny causal LM 跑最小训练闭环。
        """
    ).strip()
    report_path.write_text(report + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="examples/sample_sft.jsonl")
    parser.add_argument("--report", default="lessons/03-batching/report.md")
    parser.add_argument("--tokenizer-dir", default="lessons/02-tokenizer/outputs/local-sft-tokenizer")
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    args = parser.parse_args()

    report_path = resolve_project_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = ensure_local_auto_tokenizer(args.data, args.tokenizer_dir)
    splits = load_sft_splits(args.data)
    tokenized = splits.map(
        make_sft_tokenize_fn(tokenizer, args.max_length),
        batched=True,
        remove_columns=splits["train"].column_names,
    )

    features = [tokenized["train"][i] for i in range(args.micro_batch_size)]
    batch = sft_numpy_collator(features)

    print("Step 1: tokenize train/validation splits")
    print(tokenized)

    print("\nStep 2: collate list[dict] into dict[array]")
    for key, value in batch.items():
        print(f"{key}.shape:", value.shape)

    print("\nStep 3: effective batch size")
    print("effective batch size:", args.micro_batch_size * args.gradient_accumulation_steps)

    write_report(
        report_path,
        tokenizer,
        tokenized,
        batch,
        args.micro_batch_size,
        args.gradient_accumulation_steps,
        args.max_length,
    )
    print(f"\nReport written: {report_path.relative_to(resolve_project_path('.'))}")


if __name__ == "__main__":
    main()
