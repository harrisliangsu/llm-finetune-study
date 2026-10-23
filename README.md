# LLM Finetune Study

[![Stars](https://img.shields.io/github/stars/harrisliangsu/llm-finetune-study?style=flat-square)](https://github.com/harrisliangsu/llm-finetune-study/stargazers)
[![Forks](https://img.shields.io/github/forks/harrisliangsu/llm-finetune-study?style=flat-square)](https://github.com/harrisliangsu/llm-finetune-study/forks)
[![License: MIT](https://img.shields.io/github/license/harrisliangsu/llm-finetune-study?style=flat-square)](LICENSE)
[![Last commit](https://img.shields.io/github/last-commit/harrisliangsu/llm-finetune-study?style=flat-square)](https://github.com/harrisliangsu/llm-finetune-study/commits/main)
[![Read online](https://img.shields.io/badge/read-GitHub%20Pages-c2410c?style=flat-square)](https://harrisliangsu.github.io/llm-finetune-study/)

> 在本地电脑上把 LLM 微调走通：10 节可执行课程，从数据、tokenizer、Trainer 到 SFT、LoRA / PEFT、DPO 和 QLoRA 规划。
> [English](README.en.md) · [在线阅读](https://harrisliangsu.github.io/llm-finetune-study/) · [Discussions](https://github.com/harrisliangsu/llm-finetune-study/discussions) · [Issues](https://github.com/harrisliangsu/llm-finetune-study/issues)

这个仓库不是追求一上来训练 7B/13B。训练工作室用来看 loss、张量和 adapter，必须在本机启动，GitHub Pages 上不能训练。

## 目录

- [在线阅读](#在线阅读)
- [适合谁](#适合谁)
- [30 秒上手](#30-秒上手)
- [学习路线](#学习路线总览)
- [建议顺序](#建议顺序)
- [训练工作室](#llm-study-studio)
- [十节课](#十节课)
- [模型选择](#模型选择原则)
- [参考仓库](#参考仓库)
- [社区](#社区)

## 在线阅读

不用 clone，用浏览器读课程 HTML：

**https://harrisliangsu.github.io/llm-finetune-study/**

门户是仓库根目录的 `index.html`。站点从 `main` 的 `/ (root)` 发布。`.nojekyll` 让这些手写 HTML 原样提供，不经 Jekyll。

Pages 只包含课程页面。Markdown 文档仍在 GitHub 上阅读。`visualizer/index.html` 依赖本机 API，**不要把它当成在线工作室**。

| 课 | 主题 | 在线 HTML | 仓库 |
|---|---|---|---|
| 01 | Datasets 管线 | [阅读](https://harrisliangsu.github.io/llm-finetune-study/lessons/01-datasets/index.html) | [01-datasets](lessons/01-datasets) |
| 02 | AutoTokenizer | [阅读](https://harrisliangsu.github.io/llm-finetune-study/lessons/02-tokenizer/index.html) | [02-tokenizer](lessons/02-tokenizer) |
| 03 | Batch / Collator | [阅读](https://harrisliangsu.github.io/llm-finetune-study/lessons/03-batching/index.html) | [03-batching](lessons/03-batching) |
| 04 | Trainer 闭环 | [阅读](https://harrisliangsu.github.io/llm-finetune-study/lessons/04-trainer/index.html) | [04-trainer](lessons/04-trainer) |
| 05 | 手写 LoRA | [阅读](https://harrisliangsu.github.io/llm-finetune-study/lessons/05-lora/index.html) | [05-lora](lessons/05-lora) |
| 06 | PEFT LoRA | [阅读](https://harrisliangsu.github.io/llm-finetune-study/lessons/06-peft-lora/index.html) | [06-peft-lora](lessons/06-peft-lora) |
| 07 | SFT baseline | [阅读](https://harrisliangsu.github.io/llm-finetune-study/lessons/07-sft-baseline/index.html) | [07-sft-baseline](lessons/07-sft-baseline) |
| 08 | DPO | [阅读](https://harrisliangsu.github.io/llm-finetune-study/lessons/08-dpo-preference/index.html) | [08-dpo-preference](lessons/08-dpo-preference) |
| 09 | Reward / RLHF 概念 | [阅读](https://harrisliangsu.github.io/llm-finetune-study/lessons/09-rlhf-reward/index.html) | [09-rlhf-reward](lessons/09-rlhf-reward) |
| 10 | QLoRA 工程边界 | [阅读](https://harrisliangsu.github.io/llm-finetune-study/lessons/10-qlora-engineering/index.html) | [10-qlora-engineering](lessons/10-qlora-engineering) |

先读的文档（GitHub Markdown，不在 Pages 上排版）：

- [ROADMAP.md](ROADMAP.md)
- [docs/00-local-first-principles.md](docs/00-local-first-principles.md)
- [docs/01-finetuning-map.md](docs/01-finetuning-map.md)
- [docs/07-training-methods-guide.md](docs/07-training-methods-guide.md)
- [docs/08-training-studio.md](docs/08-training-studio.md)
- [checklists/local-finetuning-checklist.md](checklists/local-finetuning-checklist.md)

然后照着 [examples/README.md](examples/README.md) 里的练习顺序做小实验。

## 适合谁

- 想系统学习 LLM 微调，但只有本地电脑或普通开发机
- 想从小模型、小数据集开始，先把训练流程跑明白
- 想理解中文大模型微调仓库的训练脚本、数据格式和工程结构
- 想为后续学习 Agent、RAG、模型应用打好微调基础

不适合作为大规模预训练指南，也不建议直接在本地 full fine-tune 7B/13B。刷榜和堆参数不是这里的第一目标。

## 30 秒上手

```bash
./bootstrap.sh install      # 建 venv + 装全部依赖（含 torch，约 3GB）
./bootstrap.sh studio       # 启动 studio，打开 http://127.0.0.1:8765/visualizer/
```

只想先看 dataset / tokenizer，不跑训练：

```bash
./bootstrap.sh core         # 只装 datasets + tokenizers（约 100MB）
```

`Makefile` 快捷命令：`make studio` / `make test` / `make lesson01` … `make lesson10`。

本地预览课程 HTML（仍然不是工作室）：

```bash
python3 -m http.server 8000
# http://127.0.0.1:8000
```

## 学习路线总览

| 阶段 | 目标 | 推荐材料 |
|---|---|---|
| 0. 本地原则 | 明确本地机器能跑什么、不能跑什么 | [docs/00-local-first-principles.md](docs/00-local-first-principles.md) |
| 1. 微调地图 | 搞清 full fine-tune、SFT、LoRA、QLoRA、DPO 的区别 | [docs/01-finetuning-map.md](docs/01-finetuning-map.md) |
| 2. 数据和 tokenizer | 掌握数据格式、prompt 拼接、labels 构造 | [docs/02-data-tokenization.md](docs/02-data-tokenization.md) |
| 3. Trainer 闭环 | 跑通分类/生成任务微调 | [docs/03-transformers-trainer.md](docs/03-transformers-trainer.md) |
| 4. SFT + LoRA | 进入指令微调和 PEFT | [docs/04-sft-lora.md](docs/04-sft-lora.md) |
| 5. 中文仓库阅读 | 对照开源中文 LLM 微调项目 | [docs/05-reference-repos.md](docs/05-reference-repos.md) |
| 6. 评估和排错 | 看 loss、样例、指标、过拟合和数据问题 | [docs/06-evaluation-debugging.md](docs/06-evaluation-debugging.md) |
| 7. 训练方法指南 | 从开发者视角选择 SFT、PEFT/LoRA、DPO、RLHF、蒸馏等训练路线 | [docs/07-training-methods-guide.md](docs/07-training-methods-guide.md) |
| 工具. 训练工作室 | 在网页里配置独立训练 run、校验 JSONL、查看 adapter/report/trace | [docs/08-training-studio.md](docs/08-training-studio.md) |

## 建议顺序

1. 先用传统 ML/小型 NLP 任务理解训练闭环。
2. 再用 `datasets + transformers` 跑一个文本分类微调。
3. 然后跑一个 causal LM 生成式微调。
4. 接着学习 instruction/input/output 数据格式，做一个小 SFT。
5. 再用同一批固定 prompts 做 SFT 前后效果评估。
6. 最后看 LoRA/QLoRA、DPO、中文指令数据、adapter 保存和合并。

注意：当前 Lesson 02-06 已经使用 SFT 形态的数据和 labels，但课程标题分别聚焦 tokenizer、batch、Trainer、LoRA 和 PEFT。Lesson 07 已经补成独立 SFT baseline：使用“客服工单 -> 严格 JSON 路由”这个经典结构化输出任务，固定 eval prompts，对比训练前后 JSON 合法率、字段完整性和 intent/department 是否正确。PEFT adapter 的保存、加载、重新加载对比已经在 Lesson 06/07 覆盖，后续再进入 DPO。

## LLM Study Studio

本仓库带一个本地网页工作室，用来把“配置训练”和“阅读课程 trace”放在同一个界面里。它不是 GitHub Pages 的一部分。

```bash
./bootstrap.sh studio
# 或：.venv/bin/python visualizer/serve.py
# 打开 http://127.0.0.1:8765/visualizer/
```

| 页面 | 用途 | 主要产物 |
|---|---|---|
| 模型训练 | 默认首页。选择训练方法、模型、JSONL 数据、基础参数/高级参数，启动一次独立 Studio run。 | `visualizer/runtime/studio-runs/<run-id>/` |
| 课程学习 | 观察 01-10 课程 trace，运行课程脚本，查看数据流、张量、loss、adapter、checkpoint 和 Chat Lab。 | `visualizer/traces/live.json` 和 `visualizer/traces/<lesson-id>.json` |

训练工作室和课程学习页是刻意隔离的：

- Studio run 入口是 `visualizer/studio/run.py`，训练引擎在 `training/engines/`，公共 helper 在 `training/common/`。
- Studio 不调用 `lessons/` 课程脚本，也不会覆盖课程报告或课程归档 trace。
- 每次 Studio run 都是自包含目录，包含数据副本、trace、report、metrics、generation、adapter 或 QLoRA 规划结果。
- 训练页内的 Chat 对比会固定展示 base、adapter、reloaded adapter，用来验证 adapter 保存路径是否可以重新加载。

当前支持的 Studio 方法：

| 方法 | 数据格式 | 说明 |
|---|---|---|
| SFT + LoRA | `instruction/input/output` JSONL | 用严格 JSON 工单路由任务跑 SFT，并保存 LoRA adapter。 |
| PEFT LoRA | `instruction/input/output` JSONL | 直接学习真实 Hugging Face causal LM + PEFT LoRA 训练路径。 |
| DPO Preference | `instruction/input/chosen/rejected` JSONL | 不依赖 TRL，在本地学习 preference objective、policy/reference 和 adapter 保存。 |
| QLoRA Plan | 不需要训练 JSONL | 生成本地/CUDA/显存/量化工程规划；Mac/MPS 上不伪装成 bitsandbytes 4-bit 训练。 |

更完整的使用说明、数据 schema、目录结构和排错见 [docs/08-training-studio.md](docs/08-training-studio.md)；页面/API 细节见 [visualizer/README.md](visualizer/README.md)。

如果仍想手动运行课程脚本，页面会自动读取 `visualizer/traces/live.json`：

```bash
.venv/bin/python lessons/06-peft-lora/run.py --trace-delay 0.5
.venv/bin/python lessons/07-sft-baseline/run.py --trace-delay 0.5
.venv/bin/python lessons/08-dpo-preference/run.py --trace-delay 0.5
.venv/bin/python lessons/09-rlhf-reward/run.py --trace-delay 0.5
.venv/bin/python lessons/10-qlora-engineering/run.py --trace-delay 0.5
```

`--trace-delay` 只用于演示时放慢步骤，方便观察页面变化；正常跑课可以不加。

课程页右侧 `Chat Lab` 默认连接 Lesson 07 SFT baseline；训练页内的 `Chat 对比` 可以改 model、adapter dir 和 instruction，也可以一键使用最新 Studio 训练产物。这样对比时能直接看到 base model 和 adapter 分别来自哪里。

## 十节课

| 课 | 主题 | 执行记录 |
|---|---|---|
| 01 | Datasets 管线 | [report](lessons/01-datasets/report.md) |
| 02 | AutoTokenizer 与 SFT labels | [report](lessons/02-tokenizer/report.md) |
| 03 | batch、collator、padding | [report](lessons/03-batching/report.md) |
| 04 | `transformers.Trainer` 最小闭环 | [report](lessons/04-trainer/report.md) |
| 05 | 冻结 base、训练并重新加载手写 LoRA | [report](lessons/05-lora/report.md) |
| 06 | `Qwen/Qwen2.5-0.5B-Instruct` 上的真实 PEFT LoRA | [report](lessons/06-peft-lora/report.md) |
| 07 | 客服工单严格 JSON SFT baseline | [report](lessons/07-sft-baseline/report.md) |
| 08 | `prompt/chosen/rejected` 的 DPO | [report](lessons/08-dpo-preference/report.md) |
| 09 | reward model、reference、KL 和 PPO/RLHF 信号 | [report](lessons/09-rlhf-reward/report.md) |
| 10 | QLoRA、量化、显存和 Mac/CUDA 边界 | [report](lessons/10-qlora-engineering/report.md) |

顺序和验收点见 [lessons/README.md](lessons/README.md)。

一次执行 01-10：

```bash
.venv/bin/python lessons/run_all.py
```

快速冒烟测试：

```bash
.venv/bin/python lessons/run_all.py --quick
```

## 模型选择原则

课程里需要模型时，优先从 Hugging Face 下载真实模型，而不是本地随机生成模型：

- Lesson 04: `sshleifer/tiny-gpt2`，用于快速学习 Trainer 训练闭环。
- Lesson 06: `--model-name auto`，当前 32GB + MPS 本机会选择 `Qwen/Qwen2.5-0.5B-Instruct`，用于真实 PEFT LoRA。
- Lesson 07: `--model-name auto`，当前 32GB + MPS 本机会选择 `Qwen/Qwen2.5-0.5B-Instruct`，用于中文严格 JSON SFT baseline。
- Lesson 08: `--model-name auto`，继续使用真实 HF causal LM 学习 DPO 偏好优化。
- Lesson 09: 本地轻量模拟 reward/KL/PPO 信号，避免把完整 RLHF 强行塞进 Mac 本地训练。
- Lesson 10: 以硬件和训练工程规划为主，明确 QLoRA 在 CUDA/bitsandbytes 与 Mac/MPS 上的边界。

Lesson 05 保留手写 LoRA，是为了教学低秩矩阵 `A/B` 的作用，不属于真实模型训练课。

## 参考仓库

这些仓库来自个人 GitHub stars 中适合学习模型训练/微调的项目：

- [huggingface/datasets](https://github.com/huggingface/datasets)
- [huggingface/transformers](https://github.com/huggingface/transformers)
- [Lordog/dive-into-llms](https://github.com/Lordog/dive-into-llms)
- [ymcui/Chinese-LLaMA-Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca)
- [LianjiaTech/BELLE](https://github.com/LianjiaTech/BELLE)
- [LlamaChinese/Llama-Chinese](https://github.com/LlamaChinese/Llama-Chinese)
- [LAION-AI/Open-Assistant](https://github.com/LAION-AI/Open-Assistant)
- [deepspeedai/DeepSpeed](https://github.com/deepspeedai/DeepSpeed)

完整阅读建议见 [docs/05-reference-repos.md](docs/05-reference-repos.md)。

## 本地学习原则

本地电脑学习微调时，目标不是“大”，而是完整：

- 能解释每个字段为什么进入模型
- 能解释哪些 token 参与 loss
- 能看懂训练曲线和评估结果
- 能保存、加载、继续训练 checkpoint
- 能复现一个最小 SFT + LoRA 流程

把小模型训练明白，比盲目启动一个跑不动的大模型更有价值。

## 社区

怎么参与，见 [CONTRIBUTING.md](CONTRIBUTING.md)。行为准则见 [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)。

| 你想做的事 | 去哪里 |
|---|---|
| 脚本跑失败，能附上命令和 traceback | [Issue · 环境复现](https://github.com/harrisliangsu/llm-finetune-study/issues/new/choose) |
| 文档、课程说明或链接写错了 | Issue · 文档纠错 |
| 提议新的一课或 Studio 能力 | Issue · 课程或想法，或 Discussions 的 **Ideas** |
| 怎么学、卡在概念上 | Discussions 的 **Q&A** |
| 自我介绍、学习进度、闲聊 | Discussions 的 **General** |
| 展示自己的小实验 | Discussions 的 **Show and tell** |

欢迎帖：[欢迎：自我介绍 / 学习进度 / 纠错建议](https://github.com/harrisliangsu/llm-finetune-study/discussions/1)。需要改文件的纠错请开 Issue 或 PR；开放式讨论留在 Discussions。

仓库 Topics（已设置，此处只作索引）：`apple-silicon` · `chinese` · `chinese-llm` · `dpo` · `fine-tuning` · `finetuning` · `huggingface` · `llm` · `lora` · `machine-learning` · `peft` · `qlora` · `sft` · `study-notes` · `transformers` · `tutorial`

## License

[MIT](LICENSE)。
