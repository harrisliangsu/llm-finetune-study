# 02. 数据和 Tokenization

微调中最常见的问题不在模型，而在数据。

## 分类微调数据

典型字段：

```json
{
  "text": "这个产品体验很好",
  "label": 1
}
```

训练目标：

- 输入文本
- 输出类别
- loss 只来自分类头

需要检查：

- label 是否从 0 开始
- 类别是否极度不平衡
- train/validation 是否泄漏
- 文本是否过长被截断

## 生成式微调数据

典型字段：

```json
{
  "prompt": "写一句关于春天的诗",
  "completion": "春风入旧巷，花影落新茶。"
}
```

训练目标：

- 模型根据 prompt 生成 completion
- 需要明确 completion 部分是否计算 loss

## 指令微调数据

Alpaca 风格：

```json
{
  "instruction": "把下面句子翻译成英文",
  "input": "我喜欢机器学习。",
  "output": "I like machine learning."
}
```

Chat 风格：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个简洁的助手。"},
    {"role": "user", "content": "什么是 LoRA？"},
    {"role": "assistant", "content": "LoRA 是一种参数高效微调方法。"}
  ]
}
```

学习重点：

- `instruction/input/output` 怎么拼成 prompt
- `messages` 怎么套 chat template
- assistant 回答前的 token 是否参与 loss
- 多轮对话中是否只训练 assistant token

## Tokenization 需要理解的字段

### input_ids

token id 序列，模型真正看到的是这个。

### attention_mask

告诉模型哪些位置是真实 token，哪些是 padding。

### labels

训练目标。对 causal LM 来说，通常和 `input_ids` 同长度。

常见做法：

- prompt 部分 label 设置为 `-100`
- answer 部分 label 保留真实 token id
- `-100` 表示该位置不参与 loss

## 最容易踩的坑

### 1. pad token 没设置

很多 causal LM 没有显式 pad token，可以先用 eos token 作为 pad token。

### 2. max length 太小

如果 response 经常被截断，模型学不到完整回答。

### 3. labels 全是 -100

训练 loss 会异常，模型不会学到任何内容。

### 4. prompt 和 response 分隔不清

模型会学到混乱格式，推理时输出不可控。

### 5. 验证集和训练集重复

eval loss 看起来好，但泛化无意义。

## 数据检查清单

每次训练前抽 5 条样本打印：

- 原始样本
- 拼接后的 prompt
- tokenized 长度
- decode(input_ids)
- decode(labels 中非 -100 的部分)

如果这一步看不懂，不要开始训练。

