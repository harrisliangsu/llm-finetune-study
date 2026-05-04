# Lesson 04: Trainer 最小训练闭环

## 本课执行结果

- model: `sshleifer/tiny-gpt2`
- system: Darwin
- machine: arm64
- memory: 32 GB
- MPS available: True
- train 样本数: 4
- validation 样本数: 1
- max_length: 96
- max_steps: 20
- tokenizer vocab size: 50257
- trainable params: 102714
- total params: 102714
- trainable ratio: 100.0000%
- eval loss before training: 10.826223373413086
- train loss: 10.81447525024414
- eval loss after training: 10.822507858276367

本课不再手写 tiny model，而是从 Hugging Face 下载 `sshleifer/tiny-gpt2`。
它很小，适合快速看懂 `Trainer` 怎么组织 model、dataset、collator、loss、eval 和 checkpoint。

## 第 1 条训练样本的 label 检查

`decode(labels != -100)`:

```text
梯度累积是在显存不足时，把多个 mini-batch 的梯度累加起来，再执行一
```

这一步确认只有 answer 部分参与 loss，prompt 区域是 `-100`。

## 固定 prompt 训练前后生成

Base 输出：

```text
### Instruction:
解释什么是梯度累积

### Response:
 factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors
```

训练后输出：

```text
### Instruction:
解释什么是梯度累积

### Response:
 factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors factors
```

## 每一步的作用、输入、输出

| 步骤 | 作用 | 输入 | 输出 |
|---|---|---|---|
| `AutoTokenizer.from_pretrained` | 从 HF 加载 tokenizer | `sshleifer/tiny-gpt2` | token/id 映射 |
| `AutoModelForCausalLM.from_pretrained` | 从 HF 加载 causal LM | `sshleifer/tiny-gpt2` | 可训练 base model |
| 构造 tokenized dataset | 把 SFT JSONL 变成训练字段 | JSONL + tokenizer | `input_ids/attention_mask/labels` |
| `with_format("torch")` | 让 Dataset 返回 tensor | tokenized DatasetDict | Trainer 可消费的 split |
| `TrainingArguments` | 固定训练超参和输出目录 | batch、steps、lr、seed | 可复现实验设置 |
| `Trainer.evaluate()` | 训练前后在 validation 上测 loss | eval split | eval loss |
| `Trainer.train()` | 执行 forward、loss、backward、optimizer step | model + train split | train loss 和更新后的权重 |

## 你要理解的关键点

1. `Trainer` 本质是训练循环封装，不替你决定数据是否正确。
2. `labels` 仍然是核心，loss 只从非 `-100` 的位置来。
3. 本课模型很小，目标是验证训练闭环，不追求中文回答质量。
4. 后面换成 Qwen + LoRA 时，数据字段和 Trainer 闭环仍是同一套。

## 下一步

Lesson 05 继续手写 LoRA 机制；Lesson 06/07 使用真实 Hugging Face Qwen 模型执行 PEFT/SFT。
