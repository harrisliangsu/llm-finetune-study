# Lesson 01: Hugging Face Datasets 微调数据管线

## 本课目标

把本地 `examples/sample_sft.jsonl` 读成 Hugging Face `Dataset`，并完成微调前最基础的数据变换：

```text
JSONL -> Dataset -> train/validation split -> filter -> map -> formatted arrays
```

## 运行

```bash
.venv/bin/python lessons/01-datasets/inspect_dataset.py examples/sample_sft.jsonl
.venv/bin/python lessons/01-datasets/run.py
```

## 输入

- `examples/sample_sft.jsonl`
- 字段：`instruction`、`input`、`output`

## 输出

- [report.md](report.md)
- [index.html](index.html)

## 你必须理解的点

1. `load_dataset("json")` 把普通 JSONL 变成可执行数据算子的 `Dataset`。
2. `train_test_split(test_size=0.2, seed=42)` 让随机切分可复现。
3. `filter()` 是数据质量控制入口。
4. `map(..., batched=True)` 是微调预处理主干。
5. `labels=-100` 表示该位置不参与 loss。

## 验收

你应该能解释：

- 为什么本课输出是 4 条 train、1 条 validation。
- 为什么 prompt token 被 mask。
- 为什么 `decode(labels != -100)` 只能看到回答。
