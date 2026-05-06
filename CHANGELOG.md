# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `.github/` 模板：bug / feature / question issue templates、PR template
- `.github/workflows/ci.yml`：最小冒烟 CI（compileall + bash -n），跑在 ubuntu + macos
- `bootstrap.sh`：一行命令建 venv、装依赖、启动 studio
- `Makefile`：`make install / studio / test / lesson01..10`
- `requirements-core.txt` / `requirements-train.txt`：依赖分层（轻量学习 vs 真实训练）
- `CODE_OF_CONDUCT.md`：Contributor Covenant 行为准则
- `README.en.md`：英文版 README
- `VERSION` 文件：项目语义版本号

## [0.1.0] - 2026-05-06

### Added

- 10 课课程材料（01-datasets → 10-qlora-engineering），含 `run.py` / `index.html` / `report.md`
- 训练 studio（`visualizer/`）和后端 API（`visualizer/serve.py`）
- 训练脚本 `training/` 模块
- 文档 `docs/00-08`：本地优先原则、微调地图、数据/tokenizer、Trainer、SFT/LoRA、参考仓库、评估排错、训练方法选型、Studio 说明
- 学习计划 `ROADMAP.md` 和产品定义 `PRODUCT.md`
- README badges（License / Python / Platform / Lessons）、CONTRIBUTING.md、GitHub Discussions、16 个 GitHub topics

[Unreleased]: https://github.com/harrisliangsu/llm-finetune-study/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/harrisliangsu/llm-finetune-study/releases/tag/v0.1.0
