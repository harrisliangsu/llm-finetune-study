# Lesson 02: AutoTokenizer 和 SFT Labels

## 本课目标

把 Lesson 01 的 toy tokenizer 换成真实 Hugging Face tokenizer 加载路径，同时保持本地离线可复现。

```text
prompt + answer -> AutoTokenizer -> input_ids / attention_mask / labels
```

## 运行

```bash
.venv/bin/python lessons/02-tokenizer/run.py
```

## 输入

- `examples/sample_sft.jsonl`
- 本地生成的 tokenizer 目录：`lessons/02-tokenizer/outputs/local-sft-tokenizer`

## 输出

- [report.md](report.md)
- [index.html](index.html)
- 本课 datasets 缓存：`lessons/02-tokenizer/outputs/hf-cache`

## 本课关键结果

- tokenizer vocab size: 256
- prompt token 数: 13
- answer token 数: 58
- prompt mask 的 `-100` 数量: 13
- padding ignore 的 `-100` 数量: 25
- 参与 loss 的 token 数: 58

## 你必须理解的点

1. `input_ids` 是模型真正看到的整数序列。
2. `attention_mask=1` 表示真实 token，`0` 表示 padding。
3. `labels=-100` 的位置被 loss 函数忽略。
4. SFT 中通常只让 assistant answer 参与 loss。

## 验收

你应该能解释下面这行为什么只包含回答：

```text
decode(labels != -100)
```
