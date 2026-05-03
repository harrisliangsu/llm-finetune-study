#!/usr/bin/env python3
"""Lesson 01: execute a local Hugging Face datasets fine-tuning pipeline.

This script intentionally avoids downloading a remote tokenizer. It uses a tiny
character-level tokenizer so the lesson stays fully local while exercising the
real `datasets` API: load_dataset, train_test_split, filter, map, remove_columns,
and with_format.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from textwrap import dedent

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LESSON_OUTPUTS = Path(__file__).resolve().parent / "outputs"
os.environ["HF_HOME"] = str(LESSON_OUTPUTS / "hf-cache")
os.environ["HF_DATASETS_CACHE"] = str(LESSON_OUTPUTS / "hf-cache" / "datasets")
os.environ["HF_XET_CACHE"] = str(LESSON_OUTPUTS / "hf-cache" / "xet")

from datasets import DatasetDict, load_dataset


PAD_ID = 0
EOS_ID = 1
UNK_ID = 2
IGNORE_INDEX = -100


class ToyCharTokenizer:
    """A tiny tokenizer for learning data flow, not for real model training."""

    def __init__(self) -> None:
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}

    def encode(self, text: str) -> list[int]:
        ids = []
        for char in text:
            if char not in self.char_to_id:
                token_id = len(self.char_to_id) + 3
                self.char_to_id[char] = token_id
                self.id_to_char[token_id] = char
            ids.append(self.char_to_id.get(char, UNK_ID))
        return ids

    def decode(self, ids: list[int]) -> str:
        chars = []
        for token_id in ids:
            if token_id == EOS_ID:
                chars.append("<eos>")
            elif token_id > 2:
                chars.append(self.id_to_char.get(token_id, "<unk>"))
        return "".join(chars)


def build_prompt(instruction: str, user_input: str) -> str:
    if user_input.strip():
        return (
            "### Instruction:\n"
            f"{instruction.strip()}\n\n"
            "### Input:\n"
            f"{user_input.strip()}\n\n"
            "### Response:\n"
        )
    return f"### Instruction:\n{instruction.strip()}\n\n### Response:\n"


def pad(values: list[int], max_length: int, pad_value: int) -> list[int]:
    return values[:max_length] + [pad_value] * max(0, max_length - len(values))


def make_tokenize_fn(tokenizer: ToyCharTokenizer, max_length: int):
    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        input_ids_batch = []
        attention_mask_batch = []
        labels_batch = []
        prompt_lengths = []
        answer_lengths = []

        for instruction, user_input, output in zip(
            batch["instruction"], batch["input"], batch["output"]
        ):
            prompt = build_prompt(instruction, user_input)
            answer = output.strip()

            prompt_ids = tokenizer.encode(prompt)
            answer_ids = tokenizer.encode(answer) + [EOS_ID]
            input_ids = prompt_ids + answer_ids
            labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
            attention_mask = [1] * min(len(input_ids), max_length)

            input_ids_batch.append(pad(input_ids, max_length, PAD_ID))
            labels_batch.append(pad(labels, max_length, IGNORE_INDEX))
            attention_mask_batch.append(pad(attention_mask, max_length, 0))
            prompt_lengths.append(min(len(prompt_ids), max_length))
            answer_lengths.append(max(0, min(len(answer_ids), max_length - len(prompt_ids))))

        return {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels_batch,
            "prompt_length": prompt_lengths,
            "answer_length": answer_lengths,
        }

    return tokenize_batch


def write_report(
    report_path: Path,
    raw,
    splits: DatasetDict,
    tokenized: DatasetDict,
    tokenizer: ToyCharTokenizer,
    max_length: int,
) -> None:
    sample = tokenized["train"][0]
    learned_label_ids = [token_id for token_id in sample["labels"] if token_id != IGNORE_INDEX]
    prompt_ignored = sample["prompt_length"]

    report = dedent(
        f"""
        # Lesson 01: Hugging Face Datasets 微调数据管线

        ## 本课执行结果

        - 原始样本数: {len(raw)}
        - 字段: {raw.column_names}
        - train 样本数: {len(splits["train"])}
        - validation 样本数: {len(splits["validation"])}
        - tokenized 字段: {tokenized["train"].column_names}
        - max_length: {max_length}
        - toy tokenizer vocab size: {len(tokenizer.char_to_id) + 3}

        ## 你要理解的关键点

        1. `load_dataset("json", data_files=..., split="train")` 把 JSONL 读成一个 `Dataset`。
        2. `train_test_split(test_size=..., seed=42)` 生成可复现的 train/validation。
        3. `filter()` 用来删除空回答、异常样本、过长样本等。
        4. `map(..., batched=True)` 是微调数据预处理主干，适合批量 tokenization。
        5. SFT 的 `labels` 不等于无脑复制 `input_ids`，prompt 区域应该被 `-100` mask。
        6. decode 非 `-100` labels 时，应该只看到 answer 区域。

        ## SFT labels 检查

        - prompt 被 mask 的 token 数: {prompt_ignored}
        - answer 参与 loss 的 token 数: {len(learned_label_ids)}
        - labels 非 `-100` 解码结果:

        ```text
        {tokenizer.decode(learned_label_ids)}
        ```

        ## 下一步

        进入 Lesson 02 时，把 toy tokenizer 换成真实 `AutoTokenizer`，
        然后继续检查 `decode(input_ids)` 和 `decode(labels != -100)` 是否符合预期。
        """
    ).strip()
    report_path.write_text(report + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="examples/sample_sft.jsonl")
    parser.add_argument("--report", default="lessons/01-datasets/report.md")
    parser.add_argument("--max-length", type=int, default=192)
    args = parser.parse_args()

    data_path = Path(args.data)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    raw = load_dataset("json", data_files=str(data_path), split="train")
    print("Step 1: load_dataset")
    print(raw)
    print("first sample:", raw[0])

    print("\nStep 2: train_test_split")
    splits = raw.train_test_split(test_size=0.2, seed=42)
    splits = DatasetDict({"train": splits["train"], "validation": splits["test"]})
    print(splits)

    print("\nStep 3: filter empty outputs")
    splits = splits.filter(lambda item: bool(str(item["output"]).strip()))
    print(splits)

    print("\nStep 4: batched map to input_ids/attention_mask/labels")
    tokenizer = ToyCharTokenizer()
    tokenized = splits.map(
        make_tokenize_fn(tokenizer, args.max_length),
        batched=True,
        remove_columns=splits["train"].column_names,
    )
    print(tokenized)
    print("tokenized train columns:", tokenized["train"].column_names)

    print("\nStep 5: inspect labels != -100")
    sample = tokenized["train"][0]
    learned_label_ids = [token_id for token_id in sample["labels"] if token_id != IGNORE_INDEX]
    print("prompt_length:", sample["prompt_length"])
    print("answer_length:", sample["answer_length"])
    print("decoded learned labels:", tokenizer.decode(learned_label_ids))

    print("\nStep 6: with_format('numpy')")
    formatted = tokenized.with_format("numpy", columns=["input_ids", "attention_mask", "labels"])
    formatted_sample = formatted["train"][0]
    print("input_ids shape:", formatted_sample["input_ids"].shape)
    print("labels shape:", formatted_sample["labels"].shape)

    write_report(report_path, raw, splits, tokenized, tokenizer, args.max_length)
    print(f"\nReport written: {report_path}")


if __name__ == "__main__":
    main()
