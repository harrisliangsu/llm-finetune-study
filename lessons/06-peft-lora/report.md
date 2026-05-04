# Lesson 06: PEFT + 真实 Hugging Face 小模型 LoRA

## 本机配置和模型选择

- system: Darwin
- machine: arm64
- memory: 32 GB
- MPS available: True
- selected model: `Qwen/Qwen2.5-0.5B-Instruct`

选择 `Qwen/Qwen2.5-0.5B-Instruct` 的原因：它是 0.5B 级别的真实中文/多语 instruct causal LM，
在 Apple M2 Max + 32GB 内存上适合做很短的 LoRA 学习实验，比 `tiny-gpt2`
更接近真实中文 LLM 微调工程。

## 本课执行结果

- train 样本数: 4
- validation 样本数: 1
- max_length: 128
- max_steps: 6
- tokenizer vocab size: 151665
- target modules: `['q_proj', 'v_proj']`
- LoRA rank `r`: 8
- LoRA alpha: 16
- trainable params: 540672
- total params: 494573440
- trainable ratio: 0.1093%
- eval loss before training: 4.92194128036499
- train loss: 3.520887096722921
- eval loss after training: 4.761502742767334
- adapter dir: `lessons/06-peft-lora/outputs/adapter`

## 第 1 条训练样本的 label 检查

```text
梯度累积是在显存不足时，把多个 mini-batch 的梯度累加起来，再执行一次参数更新的方法。<|im_end|>
```

## 固定 prompt 生成对比

Base model 输出：

```text
### Instruction:
解释什么是梯度累积

### Response:
梯度累积是一种在机器学习和深度学习中常用的计算方法，用于评估模型的性能。它通过将每个样本的输出值与预测值进行比较，并计算它们之间的差异来实现。

具体来说，对于一个输入数据集，我们首先对每一个样本应用一个函数（称为激活函数），然后将这些函数的结果相加得到一个累积值。这个累积值可以
```

LoRA 训练后输出：

```text
### Instruction:
解释什么是梯度累积

### Response:
梯度累积是指在机器学习中，通过计算每个样本的损失函数对所有样本的累积和。这个累积值可以用来评估模型的性能，并指导后续的优化过程。

具体来说，假设我们有一个分类问题，我们的目标是将一个类别的标签分配给一个数据点。在这个例子中，我们可以使用一个损失函数来衡量我们的预测是否正确。例如，
```

重新加载 adapter 后输出：

```text
### Instruction:
解释什么是梯度累积

### Response:
梯度累积是指在机器学习中，通过计算每个样本的损失函数对所有样本的累积和。这个累积值可以用来评估模型的性能，并指导后续的优化过程。

具体来说，假设我们有一个分类问题，我们的目标是将一个类别的标签分配给一个数据点。在这个例子中，我们可以使用一个损失函数来衡量我们的预测是否正确。例如，
```

## 每一步的作用、输入、输出

| 步骤 | 作用 | 输入 | 输出 |
|---|---|---|---|
| `AutoTokenizer.from_pretrained` | 下载/加载真实模型 tokenizer | `Qwen/Qwen2.5-0.5B-Instruct` | Qwen tokenizer |
| `AutoModelForCausalLM.from_pretrained` | 下载/加载真实 causal LM | `Qwen/Qwen2.5-0.5B-Instruct` | base model |
| `LoraConfig` | 描述 LoRA 插入位置和超参 | target modules, r, alpha | PEFT 配置 |
| `get_peft_model` | 把 base model 包成 LoRA model | base model + config | trainable adapter |
| `Trainer.train` | 只训练 LoRA adapter | tokenized dataset | train/eval loss |
| `save_pretrained` | 保存 adapter-only checkpoint | PEFT model | adapter dir |
| `PeftModel.from_pretrained` | 把 adapter 加载到 fresh base | base model + adapter dir | 可推理 LoRA model |

## PEFT 还有哪些具体方法

PEFT 不是单一算法，而是一组参数高效微调方法。当前课程只执行 LoRA，因为它最常见，也最容易观察 adapter 的保存、加载和推理效果。

| 方法 | 核心想法 | 学习优先级 |
|---|---|---|
| LoRA | 在目标线性层旁边加低秩 A/B 矩阵，只训练 adapter delta | 最高 |
| AdaLoRA | 动态调整不同层的 rank，把参数预算给更重要的层 | 中 |
| IA3 | 学习少量缩放向量，调节 attention/FFN 激活 | 中 |
| Prompt Tuning | 冻结模型，只训练 soft prompt 向量 | 中 |
| Prefix Tuning | 给每层 attention 加可训练 prefix key/value | 中 |
| P-Tuning | 用可训练 prompt 表示或 prompt encoder 引导模型 | 中 |
| LoHa / LoKr | LoRA 的 Hadamard/Kronecker 分解变体 | 低 |
| OFT / BOFT | 用正交变换方式调整权重表示 | 低 |
| X-LoRA | 用门控方式组合多个 LoRA adapter | 低 |
| LayerNorm Tuning | 只训练 LayerNorm 等极少参数 | 低 |

建议顺序：LoRA -> AdaLoRA / IA3 -> Prompt / Prefix / P-Tuning -> 多 adapter 组合。

## 和 Lesson 05 的区别

- Lesson 05 手写 LoRA，目标是看清 A/B 低秩矩阵。
- Lesson 06 使用 PEFT，目标是学习真实工程接口。
- Lesson 05 不下载模型；Lesson 06 下载真实 Hugging Face base model。
- Lesson 05 把 LoRA 插在 `lm_head`；Lesson 06 插在 Qwen attention 的 `q_proj/v_proj`。

## 下一步

下一课可以做真实 adapter 管理：比较多个 adapter、加载不同 checkpoint、固定 eval prompts 做训练前后对比。
