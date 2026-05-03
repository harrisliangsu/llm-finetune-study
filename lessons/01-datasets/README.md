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
- 本课 datasets 缓存：`lessons/01-datasets/outputs/hf-cache`

## 你必须理解的点

1. `load_dataset("json")` 把普通 JSONL 变成可执行数据算子的 `Dataset`。
2. `train_test_split(test_size=0.2, seed=42)` 让随机切分可复现。
3. `filter()` 是数据质量控制入口。
4. `map(..., batched=True)` 是微调预处理主干。
5. `labels=-100` 表示该位置不参与 loss。

## 专有词解释 + 简短例子

### Dataset

作用：Hugging Face `Dataset` 可以理解成一张带 schema 的数据表。它不是普通 list，而是支持按行取样、按列查看、`filter()`、`map()`、缓存和格式转换的数据对象。

例子：`dataset[0]` 返回第一条样本，例如 `{"instruction": "解释什么是梯度累积", "input": "", "output": "..."}`；`dataset["output"]` 返回整列回答。

### JSONL

作用：JSON Lines 格式，一行就是一条 JSON 样本。它适合训练数据，因为可以按行追加、抽查和流式读取，不需要一次把整个大数组读进内存。

例子：`examples/sample_sft.jsonl` 里每一行都有 `instruction`、`input`、`output` 三个字段，`load_dataset("json")` 会把它读成 Dataset。

### split / train / validation

作用：split 是数据切分。`train` 用来更新模型参数，`validation` 用来观察模型在未训练样本上的效果，避免只看训练集 loss 自我欺骗。

例子：本课 5 条样本执行 `train_test_split(test_size=0.2, seed=42)` 后，得到 4 条 `train` 和 1 条 `validation`。

### filter

作用：`filter()` 用来删除不合格样本，是训练前的数据质量入口。常见过滤对象包括空回答、太短样本、超长样本、乱码、重复数据和错误标签。

例子：`dataset.filter(lambda row: bool(row["output"].strip()))` 会保留有回答的样本，删掉 `output` 为空的样本。

### map

作用：`map()` 用来把原始字段转换成训练字段，是微调预处理主干。SFT 中它通常负责拼 prompt、tokenize、构造 `input_ids`、`attention_mask` 和 `labels`。

例子：`dataset.map(tokenize_batch, batched=True)` 可以把一批 `instruction/input/output` 转成同长度的 token 数组。

### seed

作用：seed 是随机数种子，用来让随机切分、shuffle 等操作可复现。实验对比时，固定 seed 能避免“只是数据切分不同”造成的误判。

例子：同一份 5 条样本都用 `seed=42` 切分，下一次运行仍然应该得到相同的 train/validation 分配。

### Arrow cache

作用：`datasets` 底层使用 Apache Arrow 存储和缓存处理后的数据。缓存能让重复 `map()` 或重新加载更快，也能避免每次都从原始 JSONL 重新解析。

例子：本课把缓存放到 `lessons/01-datasets/outputs/hf-cache`，这样课程产物不会散落到系统默认缓存目录。

### labels = -100

作用：`-100` 是训练 loss 的忽略标记。SFT 里 prompt 只是上下文，不应该被模型“背诵”，所以 prompt 对应的 label 设为 `-100`；answer 对应的 label 保留 token id，参与 loss。

例子：如果 prompt 有 42 个 token、answer 有 49 个 token，那么 labels 类似 `[-100, ...42 个..., 23, 45, ...49 个 answer id...]`。decode 非 `-100` 部分时应该只看到回答。

## 验收

你应该能解释：

- 为什么本课输出是 4 条 train、1 条 validation。
- 为什么 prompt token 被 mask。
- 为什么 `decode(labels != -100)` 只能看到回答。
