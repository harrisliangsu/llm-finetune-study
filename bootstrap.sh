#!/usr/bin/env bash
# bootstrap.sh — 一行命令把仓库准备好。
#
# 用法：
#   ./bootstrap.sh           # 等价于 install
#   ./bootstrap.sh install   # 建 venv + 装全部依赖（含 torch/transformers/peft，约 ~3GB）
#   ./bootstrap.sh core      # 建 venv + 只装核心依赖（datasets/tokenizers/web，~100MB；先看 dataset 不跑训练）
#   ./bootstrap.sh studio    # 启动训练 studio（http://127.0.0.1:8765/visualizer/）
#   ./bootstrap.sh test      # 跑最小冒烟（compileall + run_all --help）

set -euo pipefail

PYTHON="${PYTHON:-python3}"
VENV=".venv"

ensure_venv() {
  if [[ ! -d "$VENV" ]]; then
    echo "▶ 建 venv：$VENV"
    "$PYTHON" -m venv "$VENV"
  fi
  # shellcheck source=/dev/null
  source "$VENV/bin/activate"
  python -m pip install --upgrade pip >/dev/null
}

cmd="${1:-install}"

case "$cmd" in
  install)
    ensure_venv
    pip install -r requirements.txt
    echo "✓ 依赖装完。下一步：./bootstrap.sh studio"
    ;;
  core)
    ensure_venv
    pip install -r requirements-core.txt
    echo "✓ 核心依赖装完（不含 torch/transformers）。"
    echo "  适合先看 lessons/01-datasets / 02-tokenizer 的 dataset 处理。"
    echo "  真正训练前请跑：./bootstrap.sh install"
    ;;
  studio)
    if [[ ! -d "$VENV" ]]; then
      echo "✗ 还没建 venv。先跑：./bootstrap.sh install" >&2
      exit 1
    fi
    # shellcheck source=/dev/null
    source "$VENV/bin/activate"
    echo "▶ 启动 studio：http://127.0.0.1:8765/visualizer/"
    exec python visualizer/serve.py
    ;;
  test)
    ensure_venv
    python -m compileall lessons training visualizer
    python lessons/run_all.py --help >/dev/null
    echo "✓ smoke 通过"
    ;;
  -h|--help|help)
    sed -n '2,11p' "$0"
    ;;
  *)
    echo "未知命令：$cmd" >&2
    sed -n '2,11p' "$0"
    exit 1
    ;;
esac
