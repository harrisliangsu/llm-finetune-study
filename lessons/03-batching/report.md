# Lesson 03: Batch、Collator 和 Padding

## 本课执行结果

- train 样本数: 4
- validation 样本数: 1
- max_length: 96
- micro batch size: 2
- gradient_accumulation_steps: 4
- effective batch size: 8
- batch `input_ids.shape`: (2, 96)
- batch `attention_mask.shape`: (2, 96)
- batch `labels.shape`: (2, 96)
- 第 1 条样本参与 loss 的 token 数: 58

## batch 中的三个核心张量

```text
input_ids      -> (2, 96)
attention_mask -> (2, 96)
labels         -> (2, 96)
```

第一维是 batch 维度，第二维是 sequence length。本课是 `2 x 96`。

## 第一条样本的局部预览

`input_ids[0][:32]`:

```text
[109, 149, 6, 72, 252, 112, 244, 72, 72, 109, 146, 6, 72, 244, 112, 248, 111, 62, 134, 97, 107, 86, 69, 54, 51, 250, 115, 249, 168, 98, 230, 223]
```

`attention_mask[0][:32]`:

```text
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

`labels[0][:32]`:

```text
[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 244, 112, 248, 111, 62, 134, 97, 107, 86, 69, 54, 51, 250, 115, 249, 168, 98, 230, 223]
```

`decode(labels[0] != -100)`:

```text
梯度累积是在显存不足时，把多个 mini-batch 的梯度累加起来，再执行一次参数更新的方法。<eos>
```

## 每一步的作用、输入、输出

| 步骤 | 作用 | 输入 | 输出 |
|---|---|---|---|
| `map(tokenize_fn)` | 把原始文本行变成定长训练字段 | `instruction/input/output` | 每条样本都有 `input_ids/attention_mask/labels` |
| `select(range(batch_size))` | 取一个 micro batch 做演示 | tokenized train split | 2 条样本列表 |
| `collator(features)` | 把样本列表堆叠成数组 | list of dict | dict of numpy arrays |
| `gradient_accumulation_steps` | 多个 micro batch 后再更新参数 | micro batch loss | 更大的 effective batch |

## 你要理解的关键点

1. dataset 里是一条条样本，训练时必须 collate 成 batch。
2. batch 的形状通常是 `[batch_size, sequence_length]`。
3. padding 让同一个 batch 中的样本长度一致。
4. `attention_mask` 让模型知道哪些位置是 padding。
5. `labels=-100` 和 padding 是两件事：前者控制 loss，后者控制有效 token。

## 下一步

Lesson 04 会把这个 batch 喂给 `Trainer`，用一个本地 tiny causal LM 跑最小训练闭环。
