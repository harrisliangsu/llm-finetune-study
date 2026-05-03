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
- 本课 datasets 缓存：`lessons/04-trainer/outputs/hf-cache`
- 本地训练输出目录：`lessons/04-trainer/outputs/trainer`

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

## 专有词解释 + 简短例子

### Trainer

`Trainer` 是 Hugging Face Transformers 提供的训练调度器。它的作用是把 dataset、model、TrainingArguments、评估逻辑接起来，自动执行 forward、loss、backward、optimizer step、日志和保存。

例子：本课调用 `Trainer.train()` 后，训练闭环会跑 30 step，并输出 `train_loss=3.7273`。

### forward

`forward` 是模型从输入计算输出的前向过程。它的作用是用当前参数把 `input_ids` 转成 `logits`，如果传入 `labels`，通常还会直接算出 `loss`。

例子：把一个 `[batch_size, sequence_length]` 的 batch 送进 tiny causal LM，模型返回每个位置对 256 个词表 token 的预测分数。

### logits

`logits` 是模型还没有经过 softmax 的原始预测分数。它的作用是表达模型对每个候选 token 的偏好，分数越高，模型越倾向预测那个 token。

例子：某个位置的 logits 里，token 42 的分数是 3.1，token 7 的分数是 0.4，说明模型当前更倾向选 token 42。

### loss

loss 是模型预测和正确答案之间的差距。它的作用是给训练一个可优化的目标，loss 越低，表示模型在当前数据上的预测越接近 labels。

例子：本课训练前 `eval loss=5.5519`，训练后训练集 `train loss=3.7273`，说明模型在训练样本上更会预测 answer token。

### cross entropy

cross entropy 是分类预测里常用的 loss。它的作用是惩罚模型没有把概率放到正确 token 上，正确 token 概率越低，loss 越大。

例子：下一个 token 正确答案是“累”。如果模型只给“累”很低概率，cross entropy 会很高；如果模型把大部分概率给“累”，cross entropy 会变低。

### backward

`backward` 是从 loss 反向计算梯度的过程。它的作用是告诉每个可训练参数：为了让 loss 下降，参数应该往哪个方向调整。

例子：`loss.backward()` 后，`embedding.weight.grad`、`lm_head.weight.grad` 等参数梯度会被填上数值。

### gradient

gradient 是 loss 对参数的变化方向和大小。它的作用是指导 optimizer 更新参数，没有 gradient，optimizer 不知道参数该怎么动。

例子：某个权重的 gradient 是正数，optimizer 可能会把这个权重往负方向调一点；gradient 绝对值越大，说明这个参数对当前 loss 更敏感。

### optimizer

optimizer 是根据 gradient 更新参数的算法。它的作用是把“应该怎么变”的梯度信号转成实际的参数更新。

例子：本课使用 AdamW。每次 optimizer step 会读取参数的 `.grad`，结合学习率计算更新量，然后修改模型权重。

### learning rate

learning rate 是每次参数更新的步子大小。它的作用是控制训练速度和稳定性，太大可能震荡或发散，太小可能学得很慢。

例子：本课 `lr=5e-3`。如果 loss 明显乱跳，可以尝试把 learning rate 调小；如果 loss 几乎不变，可能需要检查学习率或数据设置。

### epoch

epoch 是完整看完一遍训练集的次数。它的作用是描述训练数据被重复学习了多少轮。

例子：如果训练集有 4 条样本，batch size 是 2，那么跑完 2 个 batch 就相当于 1 个 epoch。

### overfitting

overfitting 是模型把训练集记得更好，但对没见过的数据表现变差。它的作用是提醒你不要只看训练 loss，还要看验证集和真实任务效果。

例子：本课训练 loss 下降，但 eval loss 从 `5.5519` 上升到 `5.9352`，这是小样本过拟合信号。

### checkpoint

checkpoint 是训练过程中保存下来的模型状态。它的作用是让你可以恢复训练、回滚到某一步，或者加载某个中间版本做评估。

例子：Trainer 的输出目录是 `lessons/04-trainer/outputs/trainer`。真实训练中通常会按 step 保存 checkpoint，例如 `checkpoint-100`。

## 验收

你应该能解释：

- `Trainer` 需要哪些输入对象。
- `TrainingArguments` 控制了哪些训练行为。
- 为什么 eval loss 和生成质量不是同一个东西。
- 为什么进入 LoRA 前必须先把这个闭环看懂。
