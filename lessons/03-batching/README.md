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

## 验收

你应该能解释：

- 为什么本课是 `(2, 96)`。
- 为什么 `attention_mask` 和 `labels` 解决的是不同问题。
- 为什么 gradient accumulation 能模拟更大的更新批量。
