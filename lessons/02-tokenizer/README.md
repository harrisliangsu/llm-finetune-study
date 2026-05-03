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

## 专有词解释 + 简短例子

### tokenizer

作用：tokenizer 负责把人类可读的文本变成模型可处理的 token id，也负责把 token id decode 回文本。模型不直接读中文字符串，而是读整数序列。

例子：`AutoTokenizer.from_pretrained(local_dir)` 加载本课本地 tokenizer；`tokenizer("解释什么是梯度累积")` 会输出 `input_ids`。

### token

作用：token 是模型处理文本的最小片段，可能是一个字、一个词、一个子词、标点或特殊符号。训练和推理时，模型都是逐 token 预测。

例子：`"梯度累积"` 不一定等于 4 个 token，真实切法由 tokenizer 的词表和规则决定。

### vocab

作用：vocab 是 tokenizer 的词表，记录 token 和整数 id 的映射。vocab size 决定模型输入嵌入表需要支持多少种 token id。

例子：本课本地 tokenizer 的 vocab size 是 256，表示它最多有 256 个基础 token id。

### input_ids

作用：`input_ids` 是 token 对应的整数序列，也是模型真正看到的输入。文本必须先变成这些 id，才能进入 embedding 层。

例子：`"### Response:"` 经过 tokenizer 后可能变成 `[41, 41, 41, 88, ...]` 这样的整数列表。

### attention_mask

作用：`attention_mask` 标记哪些位置是真实 token，哪些位置是 padding。真实 token 通常是 `1`，补齐出来的位置是 `0`，让模型不要把 padding 当正文看。

例子：`input_ids` 被补齐到 96 长度时，前 71 个真实 token 的 mask 是 `1`，后 25 个 padding 的 mask 是 `0`。

### SFT labels

作用：SFT labels 告诉模型哪些 token 要参与 loss。通常 prompt 区域设成 `-100`，answer 区域保留 token id，让模型学习“看到指令后如何回答”。

例子：本课 prompt token 数是 13，所以 labels 前 13 个都是 `-100`；answer 的 58 个 token 保留 id 并参与 loss。

### prompt / response

作用：prompt 是给模型看的问题、指令和上下文；response 是希望模型学会生成的回答。SFT 会把两者拼成一条训练文本，但 loss 主要放在 response 上。

例子：`### Instruction:\n解释什么是梯度累积\n\n### Response:` 是 prompt，`梯度累积是在...` 是 response。

### padding / truncation

作用：padding 把短序列补到统一长度，方便批量训练；truncation 把过长序列截断，防止超过模型最大长度。两者都会影响训练样本是否完整。

例子：`max_length=96` 时，不足 96 的样本会补 pad；超过 96 的样本会被截断，如果 answer 被截掉，模型就学不到完整回答。

### EOS

作用：EOS 是 end-of-sequence，表示一段回答结束。SFT 中给 response 末尾加 EOS，可以教模型在合适位置停止生成。

例子：回答会被拼成 `梯度累积是在...的方法。<eos>`，decode 非 `-100` labels 时末尾应该能看到 `<eos>`。

## 验收

你应该能解释下面这行为什么只包含回答：

```text
decode(labels != -100)
```
