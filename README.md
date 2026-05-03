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

## 建议顺序

1. 先用传统 ML/小型 NLP 任务理解训练闭环。
2. 再用 `datasets + transformers` 跑一个文本分类微调。
3. 然后跑一个 causal LM 生成式微调。
4. 接着学习 instruction/input/output 数据格式，做一个小 SFT。
5. 最后看 LoRA/QLoRA、中文指令数据、adapter 保存和合并。

## 你可以从这里开始

先读：

- [ROADMAP.md](ROADMAP.md)
- [docs/00-local-first-principles.md](docs/00-local-first-principles.md)
- [docs/01-finetuning-map.md](docs/01-finetuning-map.md)
- [checklists/local-finetuning-checklist.md](checklists/local-finetuning-checklist.md)

然后照着 [examples/README.md](examples/README.md) 里的练习顺序做小实验。

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
