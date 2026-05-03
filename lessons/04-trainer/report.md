# Lesson 04: Trainer 最小训练闭环

## 本课执行结果

- train 样本数: 4
- validation 样本数: 1
- max_length: 96
- max_steps: 30
- tokenizer vocab size: 256
- tiny model trainable params: 87872
- tiny model total params: 87872
- eval loss before training: 5.551877498626709
- train loss: 3.7272937138875326
- eval loss after training: 5.935162544250488
- 观察: train loss 下降但 eval loss 上升，这是 4 条训练样本上开始过拟合的信号。

## 第 1 条训练样本的 label 检查

`decode(labels != -100)`:

```text
梯度累积是在显存不足时，把多个 mini-batch 的梯度累加起来，再执行一次参数更新的方法。<eos>
```

## 训练后固定 prompt 生成

```text
### Instruction:
解释什么是梯度累积

### Response:
I ame tning toe-tuning.<eos>
```

## 每一步的作用、输入、输出

| 步骤 | 作用 | 输入 | 输出 |
|---|---|---|---|
| 构造 tokenizer | 本地加载真实 `AutoTokenizer` | `lessons/02-tokenizer/outputs/local-sft-tokenizer` | token/id 映射 |
| 构造 tokenized dataset | 复用 Lesson 02 的 SFT labels | JSONL + tokenizer | `input_ids/attention_mask/labels` |
| `with_format("torch")` | 让 Dataset 取样时返回 torch tensor | tokenized DatasetDict | Trainer 可消费的 split |
| tiny causal LM | 提供最小可训练模型 | vocab size | logits/loss |
| `TrainingArguments` | 固定训练超参和输出目录 | batch、steps、lr、seed | 可复现实验设置 |
| `Trainer.train()` | 执行反向传播和参数更新 | model + train_dataset | train loss 和训练状态 |
| `Trainer.evaluate()` | 用 validation 观察训练效果 | eval_dataset | eval loss |

## 你要理解的关键点

1. `Trainer` 不神秘，本质是把 model、dataset、collator、optimizer、评估循环组织起来。
2. `labels` 仍然是核心，loss 只从非 `-100` 的位置来。
3. 本课的 tiny model 不是可用 LLM，只是为了把训练闭环在本地跑通。
4. `eval_loss` 是验证集上的下一个 token 预测损失，不等于回答质量。
5. 后面换成真实 Transformer/LoRA 时，数据字段和 Trainer 闭环仍然是同一套。
6. 训练后生成的英文混杂输出不是失败，而是在提醒你：tiny model + 5 条样本只能验证流程，不能期待语言能力。

## 下一步

下一课可以进入 LoRA：冻结 base model，只训练少量 adapter 参数，并学习 adapter 保存、加载和合并。
