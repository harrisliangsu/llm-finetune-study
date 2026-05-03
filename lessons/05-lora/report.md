# Lesson 05: LoRA Adapter 微调

## 是否下载模型

本课没有下载模型。我们继续使用 Lesson 04 的 tiny causal LM 作为 base model，
只在 `lm_head` 上插入 LoRA A/B 低秩矩阵。真实 LLM LoRA 后续需要下载 base model，
但学习 LoRA 机制不需要一开始就下载大模型。

## 本课执行结果

- train 样本数: 4
- validation 样本数: 1
- max_length: 96
- max_steps: 40
- tokenizer vocab size: 256
- target module: `lm_head`
- LoRA rank `r`: 4
- LoRA alpha: 8
- trainable params: 1408
- total params: 89280
- trainable ratio: 1.5771%
- eval loss before LoRA training: 5.551877498626709
- train loss: 5.074154376983643
- eval loss after LoRA training: 5.733215808868408
- LoRA B norm after training: 5.5681471824646
- adapter path: `outputs/lesson05-lora/lora_adapter.pt`

## 第 1 条训练样本的 label 检查

```text
梯度累积是在显存不足时，把多个 mini-batch 的梯度累加起来，再执行一次参数更新的方法。<eos>
```

## 固定 prompt 生成对比

Base model 输出：

```text
### Instruction:
解释什么是梯度累积

### Response:
SFT��h。< loss loss�I������ Instructio是让�#��#ponspons参数时��t�t�t�t�t� 的t�t�t�t�t
```

LoRA 训练后输出：

```text
### Instruction:
解释什么是梯度累积

### Response:
训练�训练���，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，
```

重新加载 adapter 后输出：

```text
### Instruction:
解释什么是梯度累积

### Response:
训练�训练���，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，，
```

## 每一步的作用、输入、输出

| 步骤 | 作用 | 输入 | 输出 |
|---|---|---|---|
| freeze base | 冻结原模型参数，避免 full fine-tune | tiny causal LM | base 参数 `requires_grad=False` |
| attach LoRA | 在目标线性层旁边加 A/B 低秩矩阵 | `lm_head`, rank=4, alpha=8 | 只有 LoRA 参数可训练 |
| count params | 验证参数高效微调 | model.parameters() | trainable ratio 1.5771% |
| Trainer.train() | 只更新 LoRA A/B | tokenized SFT dataset | train/eval loss |
| save adapter | 只保存 LoRA 权重 | `lora_A`, `lora_B` | adapter-only checkpoint |
| load adapter | 把 adapter 加载到新 base 上 | fresh base + adapter file | 可复用的 LoRA 模型 |

## 你要理解的关键点

1. LoRA 不是复制一份模型重训，而是在目标线性层上学习一个低秩增量。
2. base model 参数被冻结，训练成本来自少量 `lora_A/lora_B`。
3. `r` 控制低秩瓶颈大小，`alpha/r` 控制 LoRA 增量缩放。
4. adapter checkpoint 只保存 LoRA 参数，不保存完整 base model。
5. 真实 LLM 里 target modules 通常是 `q_proj/v_proj/o_proj` 等注意力线性层；本课用 `lm_head` 是为了本地看清机制。

## 下一步

下一课可以把自定义 LoRA 换成 `peft`，并用一个真正的小型 Hugging Face causal LM 做 adapter 训练。
