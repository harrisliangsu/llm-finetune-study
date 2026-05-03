# 01. 微调地图

微调不是一个单一动作，而是一组训练策略。

## 1. Full Fine-tuning

训练全部模型参数。

优点：

- 理论上表达能力最强
- 对任务分布变化大的场景更充分

缺点：

- 显存和算力成本高
- 容易灾难性遗忘
- checkpoint 大，不利于多版本管理

本地建议：

- 只在小模型上学习
- 不建议本地 full fine-tune 7B 以上模型

## 2. Feature Extraction

冻结大部分模型，只训练一个小分类头或任务头。

适合：

- 文本分类
- 情感分析
- 简单意图识别
- embedding 后接传统模型

学习价值：

- 理解预训练模型如何作为特征提取器
- 理解 encoder 模型输出、pooling、分类头

## 3. Supervised Fine-tuning, SFT

用监督数据训练模型按照目标格式回答。

典型数据：

```json
{
  "instruction": "解释什么是 LoRA",
  "input": "",
  "output": "LoRA 是一种参数高效微调方法..."
}
```

核心问题：

- prompt 怎么拼
- response 从哪里开始
- 哪些 token 计算 loss
- 多轮对话如何套 chat template
- 训练后如何保持通用能力

## 4. LoRA

LoRA 冻结原模型参数，只在部分线性层旁边加低秩矩阵。

常见参数：

- `r`：低秩维度，越大可训练能力越强，显存也更高
- `lora_alpha`：缩放系数
- `lora_dropout`：LoRA 分支 dropout
- `target_modules`：加 LoRA 的层，例如 q_proj、v_proj、k_proj、o_proj

学习重点：

- 为什么只训练少量参数也能改变模型行为
- adapter 如何保存、加载、合并
- LoRA 和 full fine-tune 的效果/成本差异

## 5. QLoRA

QLoRA 通常是：

- 基座模型低比特量化加载
- 冻结基座
- 训练 LoRA adapter

学习重点：

- 量化是为了省显存，不是为了让训练更简单
- CUDA 生态依赖更重
- 本地 Mac 上不要把 QLoRA 当第一实践目标

## 6. Continued Pretraining

也叫继续预训练或领域继续训练。

训练数据通常是大规模无标注文本，目标仍然是 language modeling。

适合：

- 法律、医疗、金融、代码等领域语料注入
- 让模型熟悉领域词汇和表达

风险：

- 数据质量差会污染模型
- 成本高
- 单靠继续预训练不一定让模型更会“听指令”

## 7. Preference Optimization

包括 DPO、PPO、RLHF 等。

核心思想：

- 不只告诉模型标准答案
- 还告诉模型哪个回答更好

本地建议：

- 先理解 SFT
- 再读 DPO/PPO 代码
- 本地只做极小样本实验

## 推荐掌握顺序

1. 分类微调
2. causal LM 微调
3. SFT 数据格式
4. LoRA
5. adapter 保存/合并
6. 中文指令数据评估
7. QLoRA
8. DPO/RLHF
9. DeepSpeed/分布式训练

