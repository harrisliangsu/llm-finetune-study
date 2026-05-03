#!/usr/bin/env python3
"""Lesson 02: replace the toy tokenizer with a real AutoTokenizer path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import dedent

from lesson_common import (
    IGNORE_INDEX,
    build_prompt,
    decode_learned_labels,
    ensure_local_auto_tokenizer,
    load_sft_splits,
    resolve_project_path,
    tokenize_sft_row,
)


def preview_ids(values: list[int], limit: int = 40) -> str:
    shown = values[:limit]
    suffix = " ..." if len(values) > limit else ""
    return f"{shown}{suffix}"


def write_report(report_path: Path, tokenizer, sample: dict, tokenized: dict, max_length: int) -> None:
    learned_label_ids = [token_id for token_id in tokenized["labels"] if token_id != IGNORE_INDEX]
    ignored_count = sum(1 for token_id in tokenized["labels"] if token_id == IGNORE_INDEX)
    prompt_ignored = len(tokenized["prompt_ids"])
    padding_ignored = ignored_count - prompt_ignored
    sample_json = json.dumps(dict(sample), ensure_ascii=False, indent=2)
    prompt = build_prompt(sample["instruction"], sample["input"]).rstrip()

    report = dedent(
        f"""
        # Lesson 02: AutoTokenizer 和 SFT labels

        ## 本课执行结果

        - tokenizer class: `{tokenizer.__class__.__name__}`
        - tokenizer vocab size: {len(tokenizer)}
        - pad token/id: `{tokenizer.pad_token}` / {tokenizer.pad_token_id}
        - eos token/id: `{tokenizer.eos_token}` / {tokenizer.eos_token_id}
        - unk token/id: `{tokenizer.unk_token}` / {tokenizer.unk_token_id}
        - max_length: {max_length}
        - prompt token 数: {len(tokenized["prompt_ids"])}
        - answer token 数: {len(tokenized["answer_ids"])}
        - labels 中 prompt mask 的 `-100` 数量: {prompt_ignored}
        - labels 中 padding ignore 的 `-100` 数量: {padding_ignored}
        - labels 中 `-100` 总数量: {ignored_count}
        - labels 中参与 loss 的 token 数: {len(learned_label_ids)}

        ## 实际输入样本

        ```json
        {sample_json}
        ```

        ## prompt 拼接结果

        ```text
        {prompt}
        ```

        ## tokenizer 输出

        `input_ids` 前 40 个:

        ```text
        {preview_ids(tokenized["input_ids"])}
        ```

        `attention_mask` 前 40 个:

        ```text
        {preview_ids(tokenized["attention_mask"])}
        ```

        `labels` 前 40 个:

        ```text
        {preview_ids(tokenized["labels"])}
        ```

        ## decode 检查

        `decode(input_ids)` 会看到 prompt + answer + padding 相关 token：

        ```text
        {tokenizer.decode(tokenized["input_ids"], skip_special_tokens=False)}
        ```

        `decode(labels != -100)` 只能看到回答：

        ```text
        {decode_learned_labels(tokenizer, tokenized["labels"])}
        ```

        ## 你要理解的关键点

        1. `AutoTokenizer.from_pretrained(local_dir)` 不一定要联网，目录里有 tokenizer 文件就能加载。
        2. `input_ids` 是模型真正接收的整数序列，文本只是人类可读的中间形态。
        3. `attention_mask=1` 表示真实 token，`attention_mask=0` 表示 padding。
        4. `labels=-100` 的位置会被 loss 函数忽略，所以 prompt 不参与 SFT loss。
        5. 每次进入训练前都要 decode 两次：`input_ids` 和 `labels != -100`。

        ## 下一步

        Lesson 03 会把多条 tokenized 样本合成 batch，解释 batch 维度、padding、collator 和 effective batch size。
        """
    ).strip()
    report = "\n".join(line[8:] if line.startswith("        ") else line for line in report.splitlines())
    report_path.write_text(report + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="examples/sample_sft.jsonl")
    parser.add_argument("--report", default="reports/lesson02-tokenizer.md")
    parser.add_argument("--tokenizer-dir", default=".cache/local-sft-tokenizer")
    parser.add_argument("--max-length", type=int, default=96)
    args = parser.parse_args()

    report_path = resolve_project_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = ensure_local_auto_tokenizer(args.data, args.tokenizer_dir)
    splits = load_sft_splits(args.data)
    sample = splits["train"][0]
    tokenized = tokenize_sft_row(
        tokenizer,
        sample["instruction"],
        sample["input"],
        sample["output"],
        args.max_length,
    )

    print("Step 1: load local AutoTokenizer")
    print("tokenizer class:", tokenizer.__class__.__name__)
    print("vocab size:", len(tokenizer))
    print("special token ids:", tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.unk_token_id)

    print("\nStep 2: build prompt and tokenize one SFT sample")
    print("prompt token count:", len(tokenized["prompt_ids"]))
    print("answer token count:", len(tokenized["answer_ids"]))
    print("input_ids length:", len(tokenized["input_ids"]))

    print("\nStep 3: inspect labels != -100")
    print("decoded learned labels:", decode_learned_labels(tokenizer, tokenized["labels"]))

    write_report(report_path, tokenizer, sample, tokenized, args.max_length)
    print(f"\nReport written: {report_path.relative_to(resolve_project_path('.'))}")


if __name__ == "__main__":
    main()
