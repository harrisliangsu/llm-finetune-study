# Lesson 04: Trainer 最小训练闭环

## 本课目标

用本地 tiny causal LM 跑通 `transformers.Trainer` 的完整训练闭环：

```text
tokenized dataset -> torch tensors -> model forward -> loss -> backward -> optimizer step -> eval
```

## 运行

```bash
.venv/bin/python lessons/04-trainer/run.py
```

## 输入

- Lesson 02/03 同一套 SFT labels
- tiny causal LM
- `TrainingArguments`

## 输出

- [report.md](report.md)
- [index.html](index.html)
- 本地训练输出目录：`outputs/lesson04-trainer`

## 本课关键结果

- train 样本数: 4
- validation 样本数: 1
- tiny model trainable params: 87,872
- eval loss before training: 5.5519
- train loss: 3.7273
- eval loss after training: 5.9352

## 怎么理解结果

训练 loss 下降，说明参数确实被更新，模型在训练样本上更会预测 answer token。

验证 loss 上升，说明样本太少，模型开始记训练集，泛化变差。这是小样本过拟合信号。

固定 prompt 输出混杂英文，不代表流程失败。本课的模型太小、数据太少，只用于验证训练闭环，不用于获得真实语言能力。

## 验收

你应该能解释：

- `Trainer` 需要哪些输入对象。
- `TrainingArguments` 控制了哪些训练行为。
- 为什么 eval loss 和生成质量不是同一个东西。
- 为什么进入 LoRA 前必须先把这个闭环看懂。
