# Examples

这里不直接放大模型权重或大型数据，而是定义本地练习顺序。

## 练习 1：文本分类微调

目标：

- 跑通 `datasets -> tokenizer -> model -> Trainer -> eval`
- 理解分类头和指标

建议：

- 小数据集
- 小模型
- 先跑 100 到 500 step

你需要能回答：

- label 如何映射
- max length 如何影响结果
- eval accuracy/F1 怎么看

## 已落地脚本

先执行 datasets 数据管线课：

```bash
.venv/bin/python scripts/lesson01_datasets_pipeline.py
```

它会生成：

```text
reports/lesson01-datasets.md
```

这一步的目标不是训练模型，而是确认你已经理解 `load_dataset`、`train_test_split`、`filter`、`map(batched=True)` 和 SFT `labels=-100`。

## 练习 2：causal LM 微调

目标：

- 理解下一个 token 预测
- 理解 labels 和 input_ids 的关系
- 学会看生成样例

建议：

- 使用极小文本语料
- 先让模型在 20 条样本上过拟合
- 保存 checkpoint 后重新加载生成

你需要能回答：

- prompt 部分是否参与 loss
- eos token 是否正确
- eval loss 和输出质量有什么差异

## 练习 3：指令微调 SFT

目标：

- 把 `instruction/input/output` 转成训练文本
- 只训练 answer token
- 固定 eval prompts 做对比

样例数据：

```json
{"instruction": "解释什么是 LoRA", "input": "", "output": "LoRA 是一种参数高效微调方法，通过训练低秩矩阵来改变模型行为。"}
{"instruction": "把下面句子翻译成英文", "input": "我正在学习大模型微调。", "output": "I am learning LLM fine-tuning."}
```

你需要能回答：

- SFT 和普通 causal LM 微调有什么不同
- 为什么要 mask prompt loss
- 推理模板为什么要和训练模板一致

## 练习 4：LoRA

目标：

- 冻结 base model
- 只训练 adapter
- 保存和加载 adapter

你需要能回答：

- trainable parameters 占比是多少
- LoRA 加在哪些模块
- adapter 和 merged model 的区别

## 练习 5：中文微调数据复盘

目标：

- 阅读 BELLE / Chinese-LLaMA-Alpaca 的数据格式
- 自己构造 50 条高质量中文指令数据
- 比较训练前后输出

重点：

- 数据质量比数据规模更早影响效果
- 指令、输入、回答风格要稳定
- 评估 prompts 要固定
