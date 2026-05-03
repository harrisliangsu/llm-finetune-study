#!/usr/bin/env python3
"""Inspect a local JSONL SFT dataset before fine-tuning."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LESSON_OUTPUTS = Path(__file__).resolve().parent / "outputs"
os.environ["HF_HOME"] = str(LESSON_OUTPUTS / "hf-cache")
os.environ["HF_DATASETS_CACHE"] = str(LESSON_OUTPUTS / "hf-cache" / "datasets")
os.environ["HF_XET_CACHE"] = str(LESSON_OUTPUTS / "hf-cache" / "xet")

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def read_jsonl_fallback(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to a JSONL file with instruction/input/output fields")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    data_path = Path(args.path)
    if not data_path.exists():
        raise SystemExit(f"Dataset file not found: {data_path}")

    if load_dataset is None:
        rows = read_jsonl_fallback(data_path)
        validation_size = max(1, int(len(rows) * args.test_size)) if rows else 0
        train_size = max(0, len(rows) - validation_size)

        print("Dataset")
        print(f"  path: {data_path}")
        print(f"  rows: {len(rows)}")
        print(f"  columns: {list(rows[0].keys()) if rows else []}")
        print(f"  train: {train_size}")
        print(f"  validation: {validation_size}")
        print("  loader: standard-library fallback; install `datasets` for real Dataset checks")

        first = rows[0] if rows else {}
        columns = list(first.keys())
        empty_outputs = [item for item in rows if not str(item.get("output", "")).strip()]
    else:
        raw = load_dataset("json", data_files=str(data_path), split="train")
        split = raw.train_test_split(test_size=args.test_size, seed=42)

        print("Dataset")
        print(f"  path: {data_path}")
        print(f"  rows: {len(raw)}")
        print(f"  columns: {raw.column_names}")
        print(f"  train: {len(split['train'])}")
        print(f"  validation: {len(split['test'])}")
        print("  loader: huggingface/datasets")

        first = raw[0]
        columns = raw.column_names
        empty_outputs = raw.filter(lambda item: not str(item.get("output", "")).strip())

    print("\nFirst sample")
    for key in columns:
        value = str(first[key]).replace("\n", "\\n")
        print(f"  {key}: {value[:160]}")

    print("\nSanity checks")
    print(f"  empty outputs: {len(empty_outputs)}")
    print("  next: tokenize and inspect labels before training")


if __name__ == "__main__":
    main()
