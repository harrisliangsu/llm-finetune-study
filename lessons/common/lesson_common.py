#!/usr/bin/env python3
"""Shared helpers for the local fine-tuning lessons."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
os.environ.setdefault(
    "HF_DATASETS_CACHE",
    str(PROJECT_ROOT / ".cache" / "huggingface" / "datasets"),
)

from datasets import DatasetDict, load_dataset


IGNORE_INDEX = -100
PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, EOS_TOKEN, UNK_TOKEN]


def resolve_project_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


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


def load_sft_splits(data_path: str | Path, test_size: float = 0.2, seed: int = 42) -> DatasetDict:
    path = resolve_project_path(data_path)
    raw = load_dataset("json", data_files=str(path), split="train")
    splits = raw.train_test_split(test_size=test_size, seed=seed)
    splits = DatasetDict({"train": splits["train"], "validation": splits["test"]})
    return splits.filter(lambda item: bool(str(item["output"]).strip()))


def read_sft_records(data_path: str | Path) -> list[dict[str, str]]:
    path = resolve_project_path(data_path)
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def iter_training_texts(data_path: str | Path) -> list[str]:
    texts = []
    for row in read_sft_records(data_path):
        prompt = build_prompt(row["instruction"], row.get("input", ""))
        texts.append(prompt + row["output"].strip() + EOS_TOKEN)
    return texts


def ensure_local_auto_tokenizer(
    data_path: str | Path,
    tokenizer_dir: str | Path = ".cache/local-sft-tokenizer",
    vocab_size: int = 256,
):
    """Train a tiny local tokenizer and load it through AutoTokenizer.

    This keeps the lesson offline while still using the real Hugging Face fast
    tokenizer loading path.
    """

    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import AutoTokenizer

    target_dir = resolve_project_path(tokenizer_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_json = target_dir / "tokenizer.json"

    if not tokenizer_json.exists():
        tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS)
        tokenizer.train_from_iterator(iter_training_texts(data_path), trainer=trainer)
        tokenizer.save(str(tokenizer_json))

        (target_dir / "tokenizer_config.json").write_text(
            json.dumps(
                {
                    "tokenizer_class": "PreTrainedTokenizerFast",
                    "model_max_length": 512,
                    "pad_token": PAD_TOKEN,
                    "eos_token": EOS_TOKEN,
                    "unk_token": UNK_TOKEN,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (target_dir / "special_tokens_map.json").write_text(
            json.dumps(
                {
                    "pad_token": PAD_TOKEN,
                    "eos_token": EOS_TOKEN,
                    "unk_token": UNK_TOKEN,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    tokenizer = AutoTokenizer.from_pretrained(str(target_dir), local_files_only=True, use_fast=True)
    tokenizer.pad_token = PAD_TOKEN
    tokenizer.eos_token = EOS_TOKEN
    tokenizer.unk_token = UNK_TOKEN
    return tokenizer


def pad(values: list[int], max_length: int, pad_value: int) -> list[int]:
    return values[:max_length] + [pad_value] * max(0, max_length - len(values))


def tokenize_sft_row(
    tokenizer: Any,
    instruction: str,
    user_input: str,
    output: str,
    max_length: int,
) -> dict[str, Any]:
    prompt = build_prompt(instruction, user_input)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(output.strip(), add_special_tokens=False) + [tokenizer.eos_token_id]

    input_ids = prompt_ids + answer_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids
    truncated_input_ids = input_ids[:max_length]
    truncated_labels = labels[:max_length]
    attention_mask = [1] * len(truncated_input_ids)

    prompt_length = min(len(prompt_ids), max_length)
    answer_length = sum(1 for token_id in truncated_labels if token_id != IGNORE_INDEX)

    return {
        "prompt": prompt,
        "answer": output.strip() + tokenizer.eos_token,
        "prompt_ids": prompt_ids,
        "answer_ids": answer_ids,
        "input_ids": pad(truncated_input_ids, max_length, tokenizer.pad_token_id),
        "attention_mask": pad(attention_mask, max_length, 0),
        "labels": pad(truncated_labels, max_length, IGNORE_INDEX),
        "prompt_length": prompt_length,
        "answer_length": answer_length,
    }


def make_sft_tokenize_fn(tokenizer: Any, max_length: int):
    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, list[Any]]:
        output: dict[str, list[Any]] = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "prompt_length": [],
            "answer_length": [],
        }

        for instruction, user_input, answer in zip(
            batch["instruction"], batch["input"], batch["output"]
        ):
            item = tokenize_sft_row(tokenizer, instruction, user_input, answer, max_length)
            for key in output:
                output[key].append(item[key])

        return output

    return tokenize_batch


def decode_learned_labels(tokenizer: Any, labels: list[int]) -> str:
    learned_ids = [token_id for token_id in labels if token_id != IGNORE_INDEX]
    return tokenizer.decode(learned_ids, skip_special_tokens=False)


def count_trainable_parameters(model: Any) -> tuple[int, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total
