# Lesson 08: DPO Preference Optimization

## 本课目标

本课用客服回复偏好对实现一个最小 DPO。它不依赖 `trl`，只用 `transformers`、`torch` 和 `peft`：

- policy = base model + LoRA adapter
- reference = 同一个 base model 禁用 adapter
- chosen/rejected completion 都只在回答 token 上计算 log probability
- DPO loss 直接用 `-logsigmoid(beta * (pi_logratio - ref_logratio))`

## 本机和模型选择

- system: Darwin
- machine: arm64
- memory: 32 GB
- MPS available: True
- selected model: `Qwen/Qwen2.5-0.5B-Instruct`
- device: `mps`
- HF cache: `lessons/08-dpo-preference/outputs/hf-cache`

## 数据和训练配置

- preference data: `lessons/08-dpo-preference/data/preferences.jsonl`
- eval prompts: `lessons/08-dpo-preference/data/eval_prompts.jsonl`
- preference rows used: 4
- eval prompts: 2
- max_length: 128
- max_steps: 1
- beta: 0.1
- learning rate: 2e-05
- target modules: `['q_proj', 'v_proj']`
- LoRA rank `r`: 8
- LoRA alpha: 16
- trainable params: 540672
- total params: 494573440
- trainable ratio: 0.1093%
- adapter dir: `lessons/08-dpo-preference/outputs/adapter`

## 偏好指标

| 指标 | DPO 前 | DPO 后 |
|---|---:|---:|
| DPO loss | 0.6931 | 0.6789 |
| reward margin | 0.0000 | 0.0289 |
| preference accuracy | 0.00% | 100.00% |
| reference chosen-rejected logp gap | -102.8661 | -102.8661 |

`reference chosen-rejected logp gap` 是 base model 本来对 chosen/rejected 的偏好。DPO 真正优化的是 policy 相对 reference 的 reward margin。

## 第一步和最后一步训练日志

| 字段 | 第一步 | 最后一步 |
|---|---:|---:|
| loss | 0.6931 | 0.6931 |
| reward_margin | 0.0000 | 0.0000 |
| policy_logratio | -92.8131 | -92.8131 |
| ref_logratio | -92.8131 | -92.8131 |

## 固定 prompt 生成示例

输入：

```text
用户说：我买的课程突然不能看了，今天必须解决。
```

DPO 前输出：

````text
您好！感谢您的反馈。我们已经收到您关于课程购买的问题，并已
````

DPO 后输出：

````text
您好！感谢您的反馈。我们已经收到您关于课程购买的问题，并已
````

## 每一步的作用、输入、输出

| 步骤 | 作用 | 输入 | 输出 |
|---|---|---|---|
| 选择模型 | 根据本机配置选择真实 HF 模型 | local config + `--model-name` | `Qwen/Qwen2.5-0.5B-Instruct` |
| 加载 tokenizer/base | 使用 HF 真实加载路径 | model id | tokenizer + base causal LM |
| 挂 LoRA | policy 只训练 adapter | target modules/r/alpha | PEFT policy |
| 构造偏好 batch | prompt/chosen/rejected tokenization | preference JSONL | chosen/rejected tensors |
| 训练前评估 | 计算 DPO loss 和 margin | policy/reference logps | before metrics |
| 手写 DPO 更新 | `-logsigmoid(beta * delta)` | one pair per step | LoRA 梯度更新 |
| 训练后评估 | 同一批偏好对复测 | updated policy | after metrics |
| 固定 prompt 生成 | 对比 base/reference 和 policy | eval prompts | before/after JSONL |
| 保存 adapter | 输出 adapter-only checkpoint | PEFT policy | `outputs/adapter/` |

## 产物

- `outputs/generations/before.jsonl`
- `outputs/generations/after.jsonl`
- `outputs/preference_scores_before.jsonl`
- `outputs/preference_scores_after.jsonl`
- `outputs/metrics.json`
- `outputs/adapter/`
- `visualizer/traces/08-dpo-preference.json`
