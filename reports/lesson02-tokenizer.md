# Lesson 02: AutoTokenizer 和 SFT labels

## 本课执行结果

- tokenizer class: `TokenizersBackend`
- tokenizer vocab size: 256
- pad token/id: `<pad>` / 0
- eos token/id: `<eos>` / 1
- unk token/id: `<unk>` / 2
- max_length: 96
- prompt token 数: 13
- answer token 数: 58
- labels 中 prompt mask 的 `-100` 数量: 13
- labels 中 padding ignore 的 `-100` 数量: 25
- labels 中 `-100` 总数量: 38
- labels 中参与 loss 的 token 数: 58

## 实际输入样本

```json
{
  "instruction": "解释什么是梯度累积",
  "input": "",
  "output": "梯度累积是在显存不足时，把多个 mini-batch 的梯度累加起来，再执行一次参数更新的方法。"
}
```

## prompt 拼接结果

```text
### Instruction:
解释什么是梯度累积

### Response:
```

## tokenizer 输出

`input_ids` 前 40 个:

```text
[109, 149, 6, 72, 252, 112, 244, 72, 72, 109, 146, 6, 72, 244, 112, 248, 111, 62, 134, 97, 107, 86, 69, 54, 51, 250, 115, 249, 168, 98, 230, 223, 150, 24, 4, 18, 17, 33, 19, 23] ...
```

`attention_mask` 前 40 个:

```text
[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ...
```

`labels` 前 40 个:

```text
[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 244, 112, 248, 111, 62, 134, 97, 107, 86, 69, 54, 51, 250, 115, 249, 168, 98, 230, 223, 150, 24, 4, 18, 17, 33, 19, 23] ...
```

## decode 检查

`decode(input_ids)` 会看到 prompt + answer + padding 相关 token：

```text
### Instruction:
解释什么是梯度累积

### Response:
梯度累积是在显存不足时，把多个 mini-batch 的梯度累加起来，再执行一次参数更新的方法。<eos><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>
```

`decode(labels != -100)` 只能看到回答：

```text
梯度累积是在显存不足时，把多个 mini-batch 的梯度累加起来，再执行一次参数更新的方法。<eos>
```

## 你要理解的关键点

1. `AutoTokenizer.from_pretrained(local_dir)` 不一定要联网，目录里有 tokenizer 文件就能加载。
2. `input_ids` 是模型真正接收的整数序列，文本只是人类可读的中间形态。
3. `attention_mask=1` 表示真实 token，`attention_mask=0` 表示 padding。
4. `labels=-100` 的位置会被 loss 函数忽略，所以 prompt 不参与 SFT loss。
5. 每次进入训练前都要 decode 两次：`input_ids` 和 `labels != -100`。

## 下一步

Lesson 03 会把多条 tokenized 样本合成 batch，解释 batch 维度、padding、collator 和 effective batch size。
