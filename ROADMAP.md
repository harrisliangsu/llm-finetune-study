# 6 周 LLM 微调学习计划

这个计划默认你使用本地电脑学习，目标是理解和复现微调流程，而不是训练大模型。

## Week 1：训练闭环和基础概念

目标：

- 理解数据、模型、loss、优化器、评估之间的关系
- 跑一个最小机器学习或文本分类训练

任务：

- 阅读 [docs/00-local-first-principles.md](docs/00-local-first-principles.md)
- 阅读 [docs/01-finetuning-map.md](docs/01-finetuning-map.md)
- 阅读 [docs/07-training-methods-guide.md](docs/07-training-methods-guide.md) 的第 0-1 节，先建立方法选型框架
- 跑一个小型分类任务
- 记录第一次实验

验收：

- 能解释 train loss 和 eval loss
- 能解释 batch size、learning rate、epoch
- 能复现一次训练并保存结果

## Week 2：数据集和 Tokenization

目标：

- 掌握数据如何进入模型
- 掌握 labels 和 loss 的关系

任务：

- 阅读 [docs/02-data-tokenization.md](docs/02-data-tokenization.md)
- 用 `datasets` 加载一个小数据集
- 手写一个 tokenize function
- 打印 `input_ids`、`attention_mask`、`labels`

验收：

- 能 decode tokenized 样本
- 能解释 padding 和 truncation
- 能解释为什么某些 label 是 `-100`

## Week 3：Transformers Trainer

目标：

- 跑通标准 Hugging Face 微调流程

任务：

- 阅读 [docs/03-transformers-trainer.md](docs/03-transformers-trainer.md)
- 跑一个文本分类微调
- 跑一个 causal LM 小语料微调
- 保存并重新加载 checkpoint

验收：

- 能解释 `TrainingArguments`
- 能解释 effective batch size
- 能用固定 eval prompts 比较训练前后

## Week 4：SFT

目标：

- 理解指令微调数据格式
- 自己构造小型 SFT 数据

任务：

- 构造 50 条 `instruction/input/output` 数据
- 拼接训练 prompt
- mask prompt loss
- 在小模型上跑 100 到 500 step
- 对照执行 [lessons/07-sft-baseline](lessons/07-sft-baseline)，先用 40 条客服工单路由数据跑通严格 JSON SFT baseline

验收：

- 能解释 SFT 和普通 causal LM 的区别
- 能解释训练模板和推理模板为什么要一致
- 能让 20 条样本明显过拟合
- 能比较训练前后固定 eval prompts 的 JSON 合法率、intent 和 department 命中率

## Week 5：LoRA / PEFT

目标：

- 掌握参数高效微调

任务：

- 阅读 [docs/04-sft-lora.md](docs/04-sft-lora.md)
- 阅读 `Chinese-LLaMA-Alpaca/scripts/training`
- 用小模型做 LoRA 实验
- 保存 adapter
- 加载 adapter 推理

验收：

- 能解释 LoRA 的 `r`、`alpha`、`target_modules`
- 能区分 base model、adapter、merged model
- 能查看 trainable parameters

## Week 6：中文微调仓库和评估

目标：

- 能阅读中文 LLM 微调项目
- 建立自己的评估和排错流程

任务：

- 阅读 [docs/05-reference-repos.md](docs/05-reference-repos.md)
- 阅读 [docs/06-evaluation-debugging.md](docs/06-evaluation-debugging.md)
- 对照 BELLE / Chinese-LLaMA-Alpaca 的数据和训练脚本
- 固定 10 条 eval prompts 做实验比较

验收：

- 能判断一个仓库里的训练脚本负责什么
- 能解释数据质量如何影响微调
- 能写出下一轮实验计划

## 继续学习

完成 6 周后，本仓库把进阶主题继续拆成 08-10 三节可执行课程：

- [lessons/08-dpo-preference](lessons/08-dpo-preference): DPO，用 `prompt/chosen/rejected` 学偏好优化。
- [lessons/09-rlhf-reward](lessons/09-rlhf-reward): Reward / RLHF 概念，理解 reward model、reference model、KL 和 PPO 信号。
- [lessons/10-qlora-engineering](lessons/10-qlora-engineering): QLoRA / Training Engineering，理解量化、显存、CUDA/bitsandbytes 和 Mac/MPS 边界。

DeepSpeed、多模态微调、diffusion 微调暂时不作为本地第一阶段主线。先把小模型 SFT + LoRA + DPO 跑扎实。进入更大训练前，先用 [docs/07-training-methods-guide.md](docs/07-training-methods-guide.md) 判断你是在解决数据格式、参数更新、偏好优化、显存约束，还是部署压缩问题。
