# Lesson 07: SFT Baseline

本课补齐独立 SFT 基线。目标是让你看清：同一批固定 eval prompts 在训练前后如何变化，为什么 prompt 区域不参与 loss，以及模型如何从监督答案里学习严格 JSON 输出。

## 为什么用这个场景

Lesson 06 用通用概念解释做 LoRA 链路验证，训练前后效果不明显。本课改成客服工单路由到严格 JSON，因为效果可以被直接检查：

- 是否输出合法 JSON
- 是否包含 `intent`、`priority`、`department`、`summary`
- `intent` / `department` 是否选对
- 是否输出 JSON 以外的废话

## 模型选择

默认模型是：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

原因：

- 它是 Hugging Face 上的真实 instruct causal LM，不是本地手写模型。
- 0.5B 级别适合当前本机 32GB 内存 + MPS 做短步数 LoRA/SFT。
- 支持中文输入，更适合客服工单场景。

如果机器更弱，`run.py --model-name auto` 会退回更小的 Hugging Face tiny causal LM，保证课程仍可执行；但要观察中文结构化输出效果，建议用默认 Qwen。

## 自包含数据

训练数据：

```text
lessons/07-sft-baseline/data/train.jsonl
```

固定评估 prompt：

```text
lessons/07-sft-baseline/data/eval_prompts.jsonl
```

所有运行产物都写在：

```text
lessons/07-sft-baseline/outputs/
```

包括：

- `outputs/hf-cache/`: Hugging Face 模型和 tokenizer 缓存
- `outputs/adapter/`: 训练后的 PEFT adapter
- `outputs/generations/before.jsonl`: 训练前固定 prompt 输出
- `outputs/generations/after.jsonl`: 训练后固定 prompt 输出
- `outputs/generations/loaded.jsonl`: 重新加载 adapter 后输出
- `outputs/metrics.json`: loss、格式指标和模型配置

## 执行

常规执行：

```bash
.venv/bin/python lessons/07-sft-baseline/run.py
```

配合可视化页面演示：

```bash
.venv/bin/python visualizer/serve.py
.venv/bin/python lessons/07-sft-baseline/run.py --trace-delay 0.5
```

快速冒烟测试：

```bash
.venv/bin/python lessons/07-sft-baseline/run.py --max-steps 1 --max-new-tokens 16
```

## 输入输出

| 步骤 | 输入 | 输出 | 作用 |
|---|---|---|---|
| 选择模型 | 本机配置、`--model-name` | Hugging Face model id | 保证默认模型适合本地执行 |
| 加载 tokenizer/base | model id | tokenizer + base model | 使用真实 HF 模型进入训练 |
| 构造 SFT 数据 | `data/train.jsonl` | `input_ids/attention_mask/labels` | 把业务字段变成训练字段 |
| 训练前生成 | `eval_prompts.jsonl` + base model | `before.jsonl` | 建立行为基线 |
| 挂 LoRA | base model + `r/alpha/target_modules` | PEFT model | 冻结 base，只训练 adapter |
| Trainer SFT | tokenized train/validation | loss、adapter 更新 | 用标准 JSON answer 监督训练 |
| 训练后生成 | 同一批 eval prompts | `after.jsonl` | 观察格式和路由效果变化 |
| 保存/加载 adapter | PEFT model、adapter dir | `loaded.jsonl` | 验证部署时 base + adapter 路径 |

## 关键概念

SFT: Supervised Fine-Tuning，用标准答案教模型在某类输入下按目标格式回答。

loss mask: prompt 的 `labels` 设置为 `-100`，不计算 loss；answer token 保留真实 token id，参与 loss。

LoRA: 本课为了让 Qwen 在本地可训练，使用 LoRA 做参数高效 SFT。训练目标仍是 SFT，参数更新方式是 PEFT/LoRA。

adapter: LoRA 训练出的增量权重。保存时不会复制完整 base model，部署或推理时需要同一个 base model 加载 adapter。

## 验收点

完成本课后，你应该能解释：

- 为什么这个任务比“解释概念”更适合看训练前后效果。
- 为什么要同时看 `extractable JSON` 和 `strict JSON-only`：前者只是能抽出 JSON，后者才表示模型学会只输出目标结构。
- 为什么短步数 LoRA/SFT 往往先改善格式，再改善 intent/department 这类业务标签精确率。
- 为什么 SFT 不等于 PEFT，LoRA 只是本课的参数更新方式。
- 为什么 eval prompts 必须固定，才能比较训练前后。
- 为什么 adapter 需要和 base model 一起加载。
