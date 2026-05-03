# Lesson 01: Hugging Face Datasets 微调数据管线

## 本课执行结果

- 原始样本数: 5
- 字段: ['instruction', 'input', 'output']
- train 样本数: 4
- validation 样本数: 1
- tokenized 字段: ['input_ids', 'attention_mask', 'labels', 'prompt_length', 'answer_length']
- max_length: 192
- toy tokenizer vocab size: 160

## 你要理解的关键点

1. `load_dataset("json", data_files=..., split="train")` 把 JSONL 读成一个 `Dataset`。
2. `train_test_split(test_size=..., seed=42)` 生成可复现的 train/validation。
3. `filter()` 用来删除空回答、异常样本、过长样本等。
4. `map(..., batched=True)` 是微调数据预处理主干，适合批量 tokenization。
5. SFT 的 `labels` 不等于无脑复制 `input_ids`，prompt 区域应该被 `-100` mask。
6. decode 非 `-100` labels 时，应该只看到 answer 区域。

## SFT labels 检查

- prompt 被 mask 的 token 数: 42
- answer 参与 loss 的 token 数: 49
- labels 非 `-100` 解码结果:

```text
梯度累积是在显存不足时，把多个 mini-batch 的梯度累加起来，再执行一次参数更新的方法。<eos>
```

## 下一步

进入 Lesson 02 时，把 toy tokenizer 换成真实 `AutoTokenizer`，
然后继续检查 `decode(input_ids)` 和 `decode(labels != -100)` 是否符合预期。
