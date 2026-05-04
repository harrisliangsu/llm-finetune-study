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

## PEFT 方法地图

PEFT 是 Parameter-Efficient Fine-Tuning，意思是参数高效微调。它不是单一算法，而是一组“冻结大部分 base model，只训练少量新增或选定参数”的方法。LoRA 是其中最常用、最适合作为本地学习起点的一种。

| 方法 | 核心想法 | 适合先学吗 |
|---|---|---|
| LoRA | 在目标线性层旁边加低秩 A/B 矩阵，只训练 adapter delta | 是，当前课程主线 |
| AdaLoRA | 动态调整不同层的 rank，把参数预算分给更重要的层 | LoRA 后再学 |
| IA3 | 训练少量缩放向量，调节 attention/FFN 激活 | LoRA 后可了解 |
| Prompt Tuning | 训练 soft prompt，不改模型主体权重 | 可了解 |
| Prefix Tuning | 在每层 attention 前加入可训练 prefix key/value | 可了解 |
| P-Tuning | 用可训练 prompt 表示或 prompt encoder 引导模型 | 可了解 |
| LoHa / LoKr | LoRA 的 Hadamard/Kronecker 分解变体 | 进阶了解 |
| OFT / BOFT | 用正交变换方式调整权重表示 | 进阶了解 |
| X-LoRA | 用门控方式组合多个 LoRA adapter | 多 adapter 后再看 |
| LayerNorm Tuning | 只训练 LayerNorm 等极少参数 | 了解即可 |

学习顺序建议：先掌握 LoRA 的保存、加载、推理；再学 AdaLoRA/IA3；最后看 Prompt/Prefix/P-Tuning 和多 adapter 组合。

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

## 当前进度

已经完成的本地执行课：

- Lesson 02: `AutoTokenizer` 和 SFT labels，见 [lessons/02-tokenizer/report.md](../lessons/02-tokenizer/report.md)
- Lesson 03: batch、collator、padding，见 [lessons/03-batching/report.md](../lessons/03-batching/report.md)
- Lesson 04: `Trainer` 最小训练闭环，见 [lessons/04-trainer/report.md](../lessons/04-trainer/report.md)
- Lesson 05: LoRA adapter 训练，见 [lessons/05-lora/report.md](../lessons/05-lora/report.md)
- Lesson 06: PEFT + 真实 Hugging Face 小模型 LoRA，见 [lessons/06-peft-lora/report.md](../lessons/06-peft-lora/report.md)

Lesson 05 已经进入 LoRA，但用的是本地 tiny model，不下载真实 LLM。Lesson 06 使用 `Qwen/Qwen2.5-0.5B-Instruct` 跑通真实 PEFT LoRA。继续往后学习前，需要确认这些事：

1. 能解释 `decode(labels != -100)` 为什么只看到 assistant 回答。
2. 能解释 batch 的三组字段：`input_ids`、`attention_mask`、`labels`。
3. 能解释 train loss 下降但 eval loss 上升为什么通常是过拟合信号。
4. 能解释 LoRA 为什么只训练 `lora_A/lora_B`，以及 adapter 为什么能单独保存。
5. 能解释 `LoraConfig`、`get_peft_model`、`PeftModel.from_pretrained` 分别负责什么。
