# LLM Finetune Study

面向本地电脑的 LLM 微调学习路线。

这个仓库不是追求一上来训练 7B/13B，而是把微调拆成可以在本地逐步掌握的能力：

- 数据集构造、清洗、切分和 tokenization
- Hugging Face `datasets` / `transformers` 训练闭环
- 分类微调、生成式微调、指令微调 SFT
- LoRA / QLoRA / PEFT 的核心思想和实践路径
- 中文指令数据、评估、checkpoint 和 adapter 管理
- 从单机小实验过渡到 DeepSpeed、RLHF、多模态等进阶主题

## 适合谁

- 想系统学习 LLM 微调，但只有本地电脑或普通开发机
- 想从小模型、小数据集开始，先把训练流程跑明白
- 想理解中文大模型微调仓库的训练脚本、数据格式和工程结构
- 想为后续学习 Agent、RAG、模型应用打好微调基础

## 不适合什么

- 不建议把本仓库作为大规模预训练指南
- 不建议直接本地 full fine-tune 7B/13B
- 不把刷榜、炼丹参数堆叠作为第一目标

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

## 建议顺序

1. 先用传统 ML/小型 NLP 任务理解训练闭环。
2. 再用 `datasets + transformers` 跑一个文本分类微调。
3. 然后跑一个 causal LM 生成式微调。
4. 接着学习 instruction/input/output 数据格式，做一个小 SFT。
5. 再用同一批固定 prompts 做 SFT 前后效果评估。
6. 最后看 LoRA/QLoRA、DPO、中文指令数据、adapter 保存和合并。

注意：当前 Lesson 02-06 已经使用 SFT 形态的数据和 labels，但课程标题分别聚焦 tokenizer、batch、Trainer、LoRA 和 PEFT。Lesson 07 已经补成独立 SFT baseline：使用“客服工单 -> 严格 JSON 路由”这个经典结构化输出任务，固定 eval prompts，对比训练前后 JSON 合法率、字段完整性和 intent/department 是否正确。PEFT adapter 的保存、加载、重新加载对比已经在 Lesson 06/07 覆盖，后续再进入 DPO。

## 你可以从这里开始

安装依赖：

```bash
.venv/bin/python -m pip install -r requirements.txt
```

先读：

- [ROADMAP.md](ROADMAP.md)
- [lessons/README.md](lessons/README.md)
- [lessons/01-datasets/index.html](lessons/01-datasets/index.html)
- [lessons/02-tokenizer/index.html](lessons/02-tokenizer/index.html)
- [lessons/03-batching/index.html](lessons/03-batching/index.html)
- [lessons/04-trainer/index.html](lessons/04-trainer/index.html)
- [lessons/05-lora/index.html](lessons/05-lora/index.html)
- [lessons/06-peft-lora/index.html](lessons/06-peft-lora/index.html)
- [lessons/07-sft-baseline/index.html](lessons/07-sft-baseline/index.html)
- [docs/00-local-first-principles.md](docs/00-local-first-principles.md)
- [docs/01-finetuning-map.md](docs/01-finetuning-map.md)
- [docs/07-training-methods-guide.md](docs/07-training-methods-guide.md)
- [checklists/local-finetuning-checklist.md](checklists/local-finetuning-checklist.md)

然后照着 [examples/README.md](examples/README.md) 里的练习顺序做小实验。

## 实时训练可视化

启动本地可视化页面：

```bash
.venv/bin/python visualizer/serve.py
```

打开：

```text
http://127.0.0.1:8765/visualizer/
```

另开一个终端运行课程脚本，页面会自动读取 `visualizer/traces/live.json` 并展示数据流、张量形状、模型状态、训练指标和 checkpoint 变化：

```bash
.venv/bin/python lessons/06-peft-lora/run.py --trace-delay 0.5
.venv/bin/python lessons/07-sft-baseline/run.py --trace-delay 0.5
```

`--trace-delay` 只用于演示时放慢步骤，方便观察页面变化；正常跑课可以不加。

## 已执行课程

- [lessons/01-datasets/report.md](lessons/01-datasets/report.md): Hugging Face Datasets 微调数据管线执行结果
- [lessons/02-tokenizer/report.md](lessons/02-tokenizer/report.md): 本地 `AutoTokenizer`、`input_ids`、`attention_mask`、SFT `labels` 检查
- [lessons/03-batching/report.md](lessons/03-batching/report.md): batch、collator、padding 和 effective batch size
- [lessons/04-trainer/report.md](lessons/04-trainer/report.md): `transformers.Trainer` 最小训练闭环和过拟合观察
- [lessons/05-lora/report.md](lessons/05-lora/report.md): 冻结 base、训练 LoRA adapter、保存并重新加载 adapter
- [lessons/06-peft-lora/report.md](lessons/06-peft-lora/report.md): 使用 `Qwen/Qwen2.5-0.5B-Instruct` 跑真实 PEFT LoRA
- [lessons/07-sft-baseline/report.md](lessons/07-sft-baseline/report.md): 使用真实 Hugging Face Qwen + LoRA 做客服工单严格 JSON SFT baseline

一次执行 01-07：

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
