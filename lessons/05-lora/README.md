# Lesson 05: LoRA Adapter 微调

## 是否需要下载模型

本课不下载模型。

为了先学清楚 LoRA 机制，我们继续使用 Lesson 04 的 tiny causal LM 作为 base model，只在 `lm_head` 上插入 LoRA A/B 低秩矩阵。

真实 LLM LoRA 后续需要下载 base model，但一开始不需要。先把 adapter 的训练、保存、加载吃透更重要。

## 本课目标

理解参数高效微调的核心结构：

```text
frozen base model + trainable low-rank adapter = LoRA fine-tuning
```

本课执行链路：

```text
tiny causal LM -> freeze base -> attach LoRA -> train adapter -> save adapter -> load adapter
```

## 运行

```bash
.venv/bin/python lessons/05-lora/run.py
```

## 输入

- `examples/sample_sft.jsonl`
- Lesson 02 生成的本地 tokenizer
- Lesson 04 同结构 tiny causal LM

## 输出

- [report.md](report.md)
- [index.html](index.html)
- 本课 datasets 缓存：`lessons/05-lora/outputs/hf-cache`
- adapter-only checkpoint: `lessons/05-lora/outputs/lora/lora_adapter.pt`

## 本课关键结果

- target module: `lm_head`
- LoRA rank `r`: 4
- LoRA alpha: 8
- trainable params: 1,408
- total params: 89,280
- trainable ratio: 1.5771%
- eval loss before LoRA training: 5.5519
- train loss: 5.0742
- eval loss after LoRA training: 5.7332

## 你必须理解的点

1. LoRA 训练的是低秩增量，不是完整 base model。
2. base 参数被冻结，只有 `lora_A/lora_B` 更新。
3. `r` 越大，可训练参数越多，表达能力越强，也更容易过拟合。
4. `alpha/r` 是 LoRA 分支的缩放系数。
5. adapter checkpoint 只保存 adapter，不保存完整模型。

## 专有词解释 + 简短例子

### LoRA

作用：LoRA 是一种参数高效微调方法。它不直接改完整模型权重，而是在目标线性层旁边加一条很小的低秩分支，训练这条分支来学“增量”。

例子：本课不更新 `lm_head.weight`，只训练 `lora_A` 和 `lora_B`，前向结果变成 `lm_head(x) + LoRA_delta(x)`。

### rank / r

作用：`r` 决定 LoRA 低秩矩阵中间维度的大小。`r` 越大，adapter 参数越多，表达能力越强；`r` 越小，训练更省显存和存储。

例子：本课 `r=4`，所以 `lora_A` 把 hidden size 96 压到 4，`lora_B` 再从 4 映射回 vocab size 256。

### alpha / scaling

作用：`alpha` 控制 LoRA 分支输出的放大程度，实际缩放通常是 `alpha / r`。它影响 adapter delta 加到 base 输出上时的力度。

例子：本课 `alpha=8`、`r=4`，所以 scaling 是 `8 / 4 = 2`，LoRA 分支输出会乘以 2 再加到原始 `lm_head` 输出上。

### adapter

作用：adapter 是挂在 base model 上的小型可训练模块。它承担新任务的调整能力，base model 仍然负责原有语言能力。

例子：本课的 adapter 就是 `lm_head` 旁边的 `lora_A/lora_B`，它只有 1,408 个可训练参数。

### frozen base

作用：frozen base 指 base model 参数被冻结，不参与 optimizer 更新。这样训练成本低，也避免把原模型整体改坏。

例子：`TinyCausalLM` 的原始参数设置为 `requires_grad=False`，训练时梯度只更新 LoRA adapter。

### delta

作用：delta 是 LoRA 学到的“权重变化量”或“输出修正量”。LoRA 不替换原始权重，而是在原输出上叠加 delta。

例子：原来输出是 `xW^T + b`，LoRA 后是 `xW^T + b + (xA^T B^T) * scaling`，最后一项就是 adapter 产生的 delta。

### state dict

作用：state dict 是 PyTorch 保存模型参数的字典，key 是参数名，value 是张量。保存和加载 adapter 时，本质上就是读写这些张量。

例子：adapter checkpoint 里会有类似 `lora_A.weight`、`lora_B.weight` 的参数，而不是完整 tiny model 的所有参数。

### adapter-only checkpoint

作用：adapter-only checkpoint 只保存 adapter 参数，不保存 base model。它体积小，但加载时必须配合同结构的 base model。

例子：本课输出 `lessons/05-lora/outputs/lora/lora_adapter.pt`，重新推理时需要先创建 fresh `TinyCausalLM`，再加载这个 adapter。

## 验收

你应该能解释：

- 为什么 trainable params 只有 1.5771%。
- 为什么 adapter 可以单独保存。
- 为什么重新加载 adapter 后输出应和训练后 LoRA 输出一致。
- 真实 LLM 中为什么 target modules 常见是 `q_proj/v_proj/o_proj`，而本课用 `lm_head`。
