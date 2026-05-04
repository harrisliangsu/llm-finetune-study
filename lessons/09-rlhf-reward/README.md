# Lesson 09: Reward / RLHF Concept

本课把 RLHF 的核心对象放进一个本地可执行的小模拟里：

- `reward model`: 从同一 prompt 下的 `chosen/rejected` 偏好对学习回答分数。
- `reference model`: 冻结的基线分布，用来计算 KL 约束。
- `policy`: 当前要优化的回答分布。
- `KL`: 限制 policy 不要为了刷 reward 过度偏离 reference。
- `PPO`: 用 clipped ratio 控制每次策略更新幅度。

它不做完整大模型 RLHF。脚本会默认从 Hugging Face 加载 `sshleifer/tiny-gpt2` 给候选回答算 completion mean-token logprob，再在候选回答级别运行 PPO/KL。这样能看清算法结构，又不会下载或训练重型模型。

注意：本课的 reward model 是教学用轻量模型，特征来自 prompt + response 的文本规则。真实 RLHF 通常用 transformer reward model 编码完整 `(prompt, answer)`。

## 数据

```text
lessons/09-rlhf-reward/data/preference_pairs.jsonl
lessons/09-rlhf-reward/data/eval_candidates.jsonl
```

`preference_pairs.jsonl` 是 reward model 的训练数据。每行包含同一个 prompt 下的 `chosen` 和 `rejected`。

`eval_candidates.jsonl` 是 PPO 模拟的固定候选集。每个 prompt 有 3 个候选回答，其中一个标记为 `expected_best`，用于观察 policy 是否把概率移向更好的回答。

## 运行

快速执行：

```bash
.venv/bin/python lessons/09-rlhf-reward/run.py --quick
```

完整默认执行：

```bash
.venv/bin/python lessons/09-rlhf-reward/run.py
```

演示 trace 事件节奏：

```bash
.venv/bin/python lessons/09-rlhf-reward/run.py --quick --trace-delay 0.5
```

## 输出

所有运行产物都写入：

```text
lessons/09-rlhf-reward/outputs/
```

主要文件：

- `outputs/reward_weights.json`: reward model 的线性特征权重。
- `outputs/reference_scores.json`: reference model 对每个候选回答的 length-normalized mean-token logprob 和候选概率。
- `outputs/policy_before.jsonl`: PPO 前的候选分布。
- `outputs/policy_after.jsonl`: PPO 后的候选分布。
- `outputs/ppo_history.jsonl`: 每步 PPO loss、expected reward、KL。
- `outputs/metrics.json`: 汇总指标。
- `outputs/trace.json`: 本课自己的 trace events。
- `../../visualizer/traces/09-rlhf-reward.json`: 给可视化页面课程下拉框读取的归档 trace。

## 关键验收点

完成本课后，你应该能解释：

- reward model 为什么是偏好预测器，不是绝对质量真理。
- reference model 为什么通常来自 SFT 后冻结模型。
- policy 和 reference 的区别：一个被更新，一个只做约束锚点。
- KL penalty 为什么能缓解 reward hacking 和分布漂移。
- PPO 的 ratio、clip range、advantage 分别控制什么；本课固定更新前候选分布作为 `pi_old`，让 clipped ratio 在多步更新中可观察。
- 为什么本课用候选级模拟，而真实 RLHF 会在 token 序列上采样并更新语言模型参数。
