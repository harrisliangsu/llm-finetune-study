# 03. Transformers Trainer 闭环

学习微调时，`transformers` 的价值不只是 API，而是它把训练闭环标准化了。

## 最小训练闭环

1. 加载数据集
2. 加载 tokenizer
3. tokenize 数据
4. 加载模型
5. 定义 `TrainingArguments`
6. 创建 `Trainer`
7. `trainer.train()`
8. `trainer.evaluate()`
9. 保存 checkpoint
10. 加载模型推理

## 文本分类微调

适合第一步做，因为它容易验证。

重点理解：

- encoder 模型如何接分类头
- `num_labels`
- accuracy / F1
- train/eval split
- 过拟合小样本测试

建议先跑：

- `examples/pytorch/text-classification`
- 小数据集
- 小模型

## Causal LM 微调

适合第二步做，因为它接近 LLM 微调。

重点理解：

- causal mask
- 下一个 token 预测
- labels 和 input_ids 的关系
- prompt/response 拼接
- eval loss 和生成效果的差异

建议先跑：

- `examples/pytorch/language-modeling`
- `--line_by_line`
- 小模型和小语料

## 关键参数

### learning_rate

LoRA/SFT 常见学习率可能比 full fine-tune 更高，但不要机械套用。

判断方式：

- loss 不降：可能太小、数据错、参数没训练
- loss 爆炸：可能太大、精度问题、数据异常

### per_device_train_batch_size

受显存限制。小机器上 batch size 小很正常。

### gradient_accumulation_steps

用多步累积模拟更大的 effective batch size。

effective batch size:

```text
per_device_train_batch_size * gradient_accumulation_steps * num_devices
```

### num_train_epochs

小数据集上 epoch 太多容易过拟合。

### max_steps

学习阶段可以用 `max_steps` 限制成本。

### eval_strategy

建议固定间隔评估，不要只看最终结果。

### save_strategy

保存 checkpoint 方便回滚和比较。

## 你应该记录什么

每个实验至少记录：

- 模型名
- 数据版本
- 样本数量
- max length
- learning rate
- batch size
- gradient accumulation
- epoch 或 max steps
- train loss / eval loss
- 3 到 5 条生成样例
- checkpoint 路径

## 何时进入 LoRA

当你能解释下面问题时，再进入 LoRA：

- 数据是如何变成 input_ids 的
- labels 中哪些位置参与 loss
- 为什么 eval loss 降不代表回答一定好
- checkpoint 如何保存和恢复
- 一个 20 条样本过拟合实验应该长什么样

