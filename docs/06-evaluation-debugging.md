# 06. 评估和排错

微调不是只看 train loss。

## 必看的信号

### train loss

说明模型是否在训练集上学习。

问题：

- loss 不降：数据、labels、学习率、可训练参数都要查
- loss 降太快：可能数据太少或泄漏

### eval loss

说明验证集上的平均 token 预测质量。

注意：

- eval loss 低不代表回答一定符合人类偏好
- 验证集重复会让 eval loss 虚高质量

### 人工样例

每次训练后固定一组 prompts：

- 格式类问题
- 领域类问题
- 边界类问题
- 未见过的泛化问题
- 需要拒答的问题

比较：

- base model 输出
- fine-tuned model 输出
- 不同 checkpoint 输出

## 最小排错流程

### 1. 打印 tokenized 样本

检查：

- `decode(input_ids)`
- `decode(labels != -100)`
- answer 是否完整
- eos 是否存在

### 2. 20 条样本过拟合

如果小样本都过拟合不了，通常是训练管线错误。

### 3. 检查 trainable parameters

LoRA 场景必须确认可训练参数不是 0。

### 4. 固定随机种子

比较实验时，尽量固定 seed。

### 5. 一次只改一个变量

不要同时改数据、学习率、模板、模型和 LoRA rank。

## 常见问题

### 模型只会复读 prompt

可能原因：

- prompt 部分参与了太多 loss
- answer 太短或格式不清
- 推理模板和训练模板不一致

### 模型输出乱码或无法停止

可能原因：

- eos token 设置错误
- tokenizer 和模型不匹配
- max_new_tokens / stop words 设置不合理

### loss 正常但效果差

可能原因：

- 数据质量差
- instruction 太泛
- output 风格不稳定
- 评估 prompt 和训练分布不一致

### 训练显存不够

优先尝试：

- 减小 `per_device_train_batch_size`
- 增大 `gradient_accumulation_steps`
- 缩短 `max_length`
- 使用 LoRA
- 使用 mixed precision
- 换更小模型

## 实验记录模板

```markdown
## Experiment

- Date:
- Base model:
- Dataset:
- Samples:
- Task:
- Template:
- Max length:
- Trainable params:
- LR:
- Batch size:
- Gradient accumulation:
- Epochs/steps:
- Train loss:
- Eval loss:
- Checkpoint:

### Eval prompts

1.
2.
3.

### Observations

-

### Next change

-
```

