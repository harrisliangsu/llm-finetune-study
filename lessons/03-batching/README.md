# Lesson 03: Batch、Collator 和 Padding

## 本课目标

理解训练时不是一条样本一条样本直接进入模型，而是先由 collator 合成 batch。

```text
list[dict] -> collator -> dict[array]
```

## 运行

```bash
.venv/bin/python lessons/03-batching/run.py
```

## 输入

- Lesson 02 生成的 tokenizer
- `examples/sample_sft.jsonl`
- tokenized train split 中的 2 条样本

## 输出

- [report.md](report.md)
- [index.html](index.html)
- 本课 datasets 缓存：`lessons/03-batching/outputs/hf-cache`

## 本课关键结果

```text
input_ids.shape      = (2, 96)
attention_mask.shape = (2, 96)
labels.shape         = (2, 96)
effective batch size = 8
```

## 你必须理解的点

1. batch 的第一维是样本数量，第二维是序列长度。
2. padding 让同一个 batch 内的序列等长。
3. `attention_mask` 控制哪些位置是有效 token。
4. `labels=-100` 控制哪些位置参与 loss。
5. `effective batch size = micro_batch_size * gradient_accumulation_steps * num_devices`。

## 专有词解释 + 简短例子

### batch

batch 是一次送进模型的一组样本。它的作用是让模型并行处理多条数据，提高训练效率，并让 loss 基于一组样本的平均表现来计算。

例子：本课把 2 条 tokenized 样本合成一个 batch，所以 `input_ids.shape = (2, 96)`。这里的 `2` 是样本数，`96` 是每条样本的 token 长度。

### collator

collator 是把多条样本整理成 batch 的函数。它的作用是把 `list[dict]` 变成模型更容易消费的 `dict[array]`，同时处理 padding、labels 等字段。

例子：原始输入像 `[{"input_ids": [...]}, {"input_ids": [...]}]`，经过 `sft_numpy_collator` 后变成 `{"input_ids": array([[...], [...]]), "attention_mask": array(...)}`。

### padding

padding 是在较短序列后面补特殊 token，让同一个 batch 内所有序列等长。它的作用是让多条样本能堆成一个矩阵，否则长度不同的列表无法直接组成 `[batch_size, sequence_length]` 张量。

例子：一条样本有 70 个 token，另一条有 96 个 token。如果本课固定长度是 96，第一条样本后面会补 26 个 padding token。

### attention_mask

`attention_mask` 是告诉模型哪些位置是真实 token、哪些位置是 padding 的标记。它的作用是避免模型把补出来的 padding 当成真实内容去注意。

例子：`[1, 1, 1, 0, 0]` 表示前三个位置是真实 token，后两个位置是 padding。模型计算 attention 时应该忽略后两个位置。

### micro batch

micro batch 是一次 forward/backward 实际放进显存或内存的小 batch。它的作用是控制单次计算占用的显存，避免 batch 太大导致内存不够。

例子：`micro_batch_size=2` 表示每次只把 2 条样本送进模型。即使你希望一次更新等价于看 8 条样本，也可以先分 4 次小批量处理。

### gradient accumulation

gradient accumulation 是多次 micro batch 先累积梯度，攒够次数后再做一次 optimizer step。它的作用是在显存较小的机器上模拟更大的更新批量。

例子：`micro_batch_size=2`，`gradient_accumulation_steps=4`，模型会处理 4 个 micro batch 后再更新一次参数，相当于一次更新看过 8 条样本。

### effective batch size

effective batch size 是一次参数更新真正对应的样本数量。它的作用是帮助你判断训练稳定性和学习率设置，而不只看单次 forward 的 micro batch。

例子：本课 `micro_batch_size=2`，`gradient_accumulation_steps=4`，`num_devices=1`，所以 `effective batch size = 2 * 4 * 1 = 8`。

## 验收

你应该能解释：

- 为什么本课是 `(2, 96)`。
- 为什么 `attention_mask` 和 `labels` 解决的是不同问题。
- 为什么 gradient accumulation 能模拟更大的更新批量。
