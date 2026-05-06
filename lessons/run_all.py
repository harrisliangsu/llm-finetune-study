#!/usr/bin/env python3
"""Run the lesson scripts in order."""
# https://github.com/harrisliangsu/llm-finetune-study
# version: 0.1.0

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


LESSONS = [
    ("01", "lessons/01-datasets/run.py", []),
    ("02", "lessons/02-tokenizer/run.py", []),
    ("03", "lessons/03-batching/run.py", []),
    ("04", "lessons/04-trainer/run.py", []),
    ("05", "lessons/05-lora/run.py", []),
    ("06", "lessons/06-peft-lora/run.py", []),
    ("07", "lessons/07-sft-baseline/run.py", []),
    ("08", "lessons/08-dpo-preference/run.py", []),
    ("09", "lessons/09-rlhf-reward/run.py", []),
    ("10", "lessons/10-qlora-engineering/run.py", []),
]


QUICK_ARGS = {
    "04": ["--max-steps", "1"],
    "05": ["--max-steps", "1"],
    "06": ["--max-steps", "1"],
    "07": ["--max-steps", "1", "--max-new-tokens", "16"],
    "08": ["--quick", "--max-steps", "1", "--max-new-tokens", "16"],
    "09": ["--quick"],
    "10": ["--quick"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-lesson", default="01")
    parser.add_argument("--to-lesson", default="10")
    parser.add_argument("--trace-delay", type=float, default=0.0)
    parser.add_argument("--quick", action="store_true", help="Use one-step training for smoke tests.")
    args = parser.parse_args()

    selected = [
        item for item in LESSONS
        if args.from_lesson <= item[0] <= args.to_lesson
    ]
    if not selected:
        raise SystemExit("No lessons selected.")

    for lesson_id, script, extra in selected:
        command = [
            sys.executable,
            script,
            "--trace",
            "visualizer/traces/live.json",
            "--trace-delay",
            str(args.trace_delay),
            *extra,
        ]
        if args.quick:
            command.extend(QUICK_ARGS.get(lesson_id, []))
        print(f"\n== Lesson {lesson_id}: {' '.join(command)} ==")
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
