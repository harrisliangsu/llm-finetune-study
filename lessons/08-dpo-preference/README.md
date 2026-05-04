# Lesson 08: DPO Preference Optimization

本课实现一个不依赖 `trl` 的最小 DPO 训练循环：用真实 Hugging Face causal LM 作为 base，挂 LoRA adapter 作为 policy，并用同一个 PEFT 模型在 `disable_adapter()` 下计算 reference log probability。

## 本课学什么

DPO 的输入不是 SFT 的单条标准答案，而是偏好对：

```text
prompt + chosen
prompt + rejected
```

目标是让 policy 相对 reference 更偏向 `chosen`，同时不需要显式训练 reward model，也不需要 PPO rollout。

## 模型选择

默认执行：

```bash
.venv/bin/python lessons/08-dpo-preference/run.py
```

`--model-name auto` 会调用课程已有的本地模型策略。当前 32GB Mac 默认选择：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

如果只是做冒烟测试，可以减少步数：

```bash
.venv/bin/python lessons/08-dpo-preference/run.py --quick --max-new-tokens 24
```

## 数据和输出

数据只放在本课目录：

- `data/preferences.jsonl`: DPO chosen/rejected 偏好对
- `data/eval_prompts.jsonl`: 固定生成 prompt

运行产物只放在本课目录：

- `outputs/adapter/`: LoRA adapter
- `outputs/generations/before.jsonl`
- `outputs/generations/after.jsonl`
- `outputs/metrics.json`
- `outputs/hf-cache/`

同时，`VisualTrace` 会写实时 trace，并归档到：

```text
visualizer/traces/08-dpo-preference.json
```

## 关键公式

对一条偏好样本，先计算 completion log probability：

```text
pi_logratio  = log pi(chosen | prompt) - log pi(rejected | prompt)
ref_logratio = log ref(chosen | prompt) - log ref(rejected | prompt)
loss = -log sigmoid(beta * (pi_logratio - ref_logratio))
```

其中：

- `policy`: base model + LoRA adapter
- `reference`: 同一个 base model 禁用 adapter
- `beta`: 控制偏好更新强度

## 观察指标

重点看 `report.md` 里的：

- DPO loss 是否下降
- reward margin 是否从接近 0 变为正数
- preference accuracy 是否提升
- 固定 prompt 的生成是否更少越权承诺、索要敏感信息、跳过核验

短步数本地课程不追求业务效果显著，只要求你能看懂 DPO 的数据路径、log probability 计算和 policy/reference 对比。
