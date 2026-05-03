# 04. SFT 和 LoRA

SFT 和 LoRA 是学习 LLM 微调的核心组合。

## SFT 到底训练什么

SFT 的目标不是让模型“记住问题”，而是让模型在给定上下文下生成目标回答。

例如：

```text
### Instruction:
解释什么是梯度累积

### Response:
梯度累积是在显存不足时，把多个 mini-batch 的梯度累加后再更新参数的方法。
```

训练时通常希望：

- prompt 部分不计算 loss
- response 部分计算 loss

否则模型会把用户问题、模板、回答都当成同等目标学习。

## LoRA 的训练对象

LoRA 通常训练的是注意力或 MLP 中的部分线性层旁路参数。

常见 target modules：

- `q_proj`
- `k_proj`
- `v_proj`
- `o_proj`
- `gate_proj`
- `up_proj`
- `down_proj`

不同模型命名不同，不能盲抄。

## LoRA 实验顺序

### 1. 先跑小模型 SFT

目标：

- 数据格式正确
- loss 能下降
- 20 条样本能过拟合

### 2. 只训练 LoRA adapter

检查：

- trainable parameters 是否明显少于总参数
- checkpoint 中是否主要保存 adapter
- 加载 adapter 后输出是否变化

### 3. 调参

优先调：

- learning rate
- max length
- batch size
- gradient accumulation
- LoRA rank `r`
- target modules

### 4. 合并和推理

需要理解两种模式：

- 保留 base model + adapter
- merge adapter 到 base model

学习阶段建议先保留 adapter，便于比较多个实验。

## LoRA 常见误区

### 误区 1：LoRA 一定不如 full fine-tune

不一定。数据少、任务明确时，LoRA 通常更稳、更便宜。

### 误区 2：rank 越大越好

rank 大会增加可训练参数，也更容易过拟合。

### 误区 3：只看 loss

SFT 必须看样例输出。格式、语气、事实性、拒答边界都要人工检查。

### 误区 4：忽略模板

训练模板和推理模板不一致，会严重影响效果。

## 本地建议

本地学习阶段优先使用：

- 小模型
- 小数据
- 短 max length
- 少量 steps
- 明确的 eval prompts

目标不是拿到最强模型，而是能解释：

- 为什么这样拼 prompt
- 为什么只训练回答 token
- LoRA 参数加在哪些层
- adapter 如何保存、加载和合并

