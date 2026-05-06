# Makefile — 一行命令跑完日常操作。
# 默认假设已经跑过 ./bootstrap.sh install，并使用 .venv/bin/python。

PY ?= .venv/bin/python

.PHONY: help install install-core studio test compileall \
        lesson01 lesson02 lesson03 lesson04 lesson05 \
        lesson06 lesson07 lesson08 lesson09 lesson10 \
        clean

help:
	@echo "目标："
	@echo "  install        建 venv + 装全部依赖（含 torch）"
	@echo "  install-core   只装核心依赖（datasets/tokenizers，先看 dataset 不跑训练）"
	@echo "  studio         启动训练 studio"
	@echo "  test           跑最小冒烟（compileall + run_all --help）"
	@echo "  lesson01..10   跑某一节课的 run.py"
	@echo "  clean          清理 __pycache__ / outputs"

install:
	./bootstrap.sh install

install-core:
	./bootstrap.sh core

studio:
	./bootstrap.sh studio

test: compileall
	$(PY) lessons/run_all.py --help >/dev/null
	@echo "✓ smoke 通过"

compileall:
	$(PY) -m compileall lessons training visualizer

lesson01: ; $(PY) lessons/01-datasets/run.py
lesson02: ; $(PY) lessons/02-tokenizer/run.py
lesson03: ; $(PY) lessons/03-batching/run.py
lesson04: ; $(PY) lessons/04-trainer/run.py
lesson05: ; $(PY) lessons/05-lora/run.py
lesson06: ; $(PY) lessons/06-peft-lora/run.py
lesson07: ; $(PY) lessons/07-sft-baseline/run.py
lesson08: ; $(PY) lessons/08-dpo-preference/run.py
lesson09: ; $(PY) lessons/09-rlhf-reward/run.py
lesson10: ; $(PY) lessons/10-qlora-engineering/run.py

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find lessons -type d -name outputs -prune -exec rm -rf {} +
