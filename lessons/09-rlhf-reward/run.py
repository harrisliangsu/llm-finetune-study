#!/usr/bin/env python3
"""Lesson 09: reward model and tiny RLHF/PPO simulation.

This lesson intentionally avoids full RLHF training. It uses a real tiny
Hugging Face causal LM to produce reference log probabilities, then runs a
small preference-trained reward model and a candidate-level PPO update with KL
control.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

LESSON_DIR = Path(__file__).resolve().parent
LESSON_OUTPUTS = LESSON_DIR / "outputs"
os.environ["HF_HOME"] = str(LESSON_OUTPUTS / "hf-cache")
os.environ["HF_DATASETS_CACHE"] = str(LESSON_OUTPUTS / "hf-cache" / "datasets")
os.environ["HF_XET_CACHE"] = str(LESSON_OUTPUTS / "hf-cache" / "xet")

from lessons.common.hf_model_policy import DEFAULT_TINY_CAUSAL_LM, detect_local_config
from lessons.common.lesson_common import resolve_project_path
from lessons.common.visual_trace import VisualTrace

FEATURE_NAMES = [
    "length_norm",
    "specificity",
    "stepwise",
    "clarifying",
    "privacy_guard",
    "safety_refusal",
    "strict_json",
    "evaluation_terms",
    "prompt_overlap",
    "unsafe_assistance",
    "vague_or_overconfident",
]


def require_stack():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError as exc:
        raise SystemExit(
            "Lesson 09 requires torch and transformers. Install project requirements first."
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, set_seed


class LessonTrace(VisualTrace):
    """VisualTrace with fixed lesson metadata plus local output trace path."""

    def __init__(self, path: str | Path, delay: float = 0.0) -> None:
        super().__init__("09-rlhf-reward", "Lesson 09 · Reward / RLHF Concept", path, delay)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in resolve_project_path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_reference_model_name(requested: str) -> str:
    if requested == "auto":
        return DEFAULT_TINY_CAUSAL_LM
    return requested


def load_hf_reference_model(AutoModelForCausalLM, AutoTokenizer, model_name: str):  # noqa: ANN001
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True)
    model.eval()
    return tokenizer, model


def response_features(text: str, tokenizer, prompt: str = "") -> list[float]:
    stripped = text.strip()
    lower = f"{prompt}\n{stripped}".lower()
    token_count = len(tokenizer.encode(stripped, add_special_tokens=False))
    length_norm = min(token_count / 96.0, 1.0)
    specificity_terms = [
        "trace",
        "p95",
        "p99",
        "json",
        "sql",
        "参数化",
        "审批",
        "备份",
        "错误码",
        "rollout",
        "测试",
        "指标",
        "审计",
    ]
    step_terms = ["先", "再", "然后", "步骤", "1.", "2.", "检查", "确认"]
    clarify_terms = ["确认", "补齐", "提供", "期望", "实际", "复现"]
    privacy_terms = ["脱敏", "最小权限", "token", "密钥", "手机号", "审计", "轮换", "不要传播"]
    refusal_terms = ["不能", "不要", "不建议", "避免", "不要直接", "不能直接"]
    eval_terms = ["评估", "指标", "成功率", "trace", "回归", "复核", "kl", "reward"]
    unsafe_terms = ["rm -rf", "sudo", "delete pod --all", "完整日志", "完整个人信息", "不用查", "直接运行"]
    vague_terms = ["应该没问题", "看起来", "重启一下", "感觉不错", "一定是", "不用查"]

    def count_terms(terms: list[str]) -> float:
        return min(sum(1 for term in terms if term.lower() in lower) / 3.0, 1.0)

    prompt_token_ids = set(tokenizer.encode(prompt, add_special_tokens=False))
    response_token_ids = set(tokenizer.encode(stripped, add_special_tokens=False))
    prompt_overlap = 0.0
    if prompt_token_ids and response_token_ids:
        prompt_overlap = min(len(prompt_token_ids & response_token_ids) / max(1, min(len(prompt_token_ids), len(response_token_ids))), 1.0)

    return [
        length_norm,
        count_terms(specificity_terms),
        count_terms(step_terms),
        count_terms(clarify_terms),
        count_terms(privacy_terms),
        count_terms(refusal_terms),
        1.0 if stripped.startswith("{") and stripped.endswith("}") else 0.0,
        count_terms(eval_terms),
        prompt_overlap,
        count_terms(unsafe_terms),
        count_terms(vague_terms),
    ]


@dataclass
class RewardBundle:
    model: Any
    mean: Any
    std: Any
    feature_names: list[str]

    def tensorize(self, torch, rows: list[list[float]]):
        values = torch.tensor(rows, dtype=torch.float32)
        return (values - self.mean) / self.std

    def score_texts(self, torch, tokenizer, texts: list[str], prompts: list[str] | None = None):
        with torch.no_grad():
            prompts = prompts or [""] * len(texts)
            features = [response_features(text, tokenizer, prompts[index]) for index, text in enumerate(texts)]
            return self.model(self.tensorize(torch, features)).squeeze(-1)


def train_reward_model(torch, pairs: list[dict[str, Any]], tokenizer, epochs: int, lr: float) -> tuple[RewardBundle, dict[str, float]]:
    chosen_features = [response_features(row["chosen"], tokenizer, row.get("prompt", "")) for row in pairs]
    rejected_features = [response_features(row["rejected"], tokenizer, row.get("prompt", "")) for row in pairs]
    all_features = torch.tensor(chosen_features + rejected_features, dtype=torch.float32)
    mean = all_features.mean(dim=0, keepdim=True)
    std = all_features.std(dim=0, keepdim=True).clamp_min(0.05)

    chosen = (torch.tensor(chosen_features, dtype=torch.float32) - mean) / std
    rejected = (torch.tensor(rejected_features, dtype=torch.float32) - mean) / std
    model = torch.nn.Linear(len(FEATURE_NAMES), 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    history = []
    for epoch in range(epochs):
        chosen_scores = model(chosen).squeeze(-1)
        rejected_scores = model(rejected).squeeze(-1)
        margins = chosen_scores - rejected_scores
        loss = -torch.nn.functional.logsigmoid(margins).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 0 or epoch == epochs - 1 or (epoch + 1) % max(1, epochs // 4) == 0:
            history.append({"epoch": epoch + 1, "loss": float(loss.detach().item()), "margin": float(margins.mean().detach().item())})

    with torch.no_grad():
        chosen_scores = model(chosen).squeeze(-1)
        rejected_scores = model(rejected).squeeze(-1)
        margins = chosen_scores - rejected_scores
        accuracy = float((margins > 0).float().mean().item())
        mean_margin = float(margins.mean().item())
        final_loss = float((-torch.nn.functional.logsigmoid(margins).mean()).item())

    bundle = RewardBundle(model=model, mean=mean, std=std, feature_names=FEATURE_NAMES)
    metrics = {
        "preference_accuracy": accuracy,
        "mean_reward_margin": mean_margin,
        "final_reward_loss": final_loss,
        "epochs": epochs,
        "history": history,
    }
    return bundle, metrics


def sequence_logprob(torch, model, tokenizer, prompt: str, completion: str, max_length: int) -> float:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion, add_special_tokens=False) + [tokenizer.eos_token_id]
    ids = (prompt_ids + completion_ids)[-max_length:]
    prompt_tokens_kept = max(0, len(ids) - len(completion_ids))
    if len(ids) < 2:
        return 0.0

    input_ids = torch.tensor([ids[:-1]], dtype=torch.long)
    target_ids = torch.tensor(ids[1:], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    start = max(0, prompt_tokens_kept - 1)
    selected = log_probs[start:].gather(1, target_ids[start:].unsqueeze(1)).squeeze(1)
    return float(selected.mean().item())


def build_reference_tables(torch, model, tokenizer, eval_rows: list[dict[str, Any]], max_length: int, temperature: float) -> list[dict[str, Any]]:
    tables = []
    for row in eval_rows:
        logps = [
            sequence_logprob(torch, model, tokenizer, row["prompt"], candidate["text"], max_length)
            for candidate in row["candidates"]
        ]
        logits = torch.tensor(logps, dtype=torch.float32) / max(temperature, 1e-6)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        tables.append(
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "candidate_ids": [candidate["id"] for candidate in row["candidates"]],
                "reference_mean_token_logps": logps,
                "reference_logps_note": "length-normalized mean token logprob for the completion, not raw summed sequence logprob",
                "reference_probs": torch.exp(log_probs).tolist(),
            }
        )
    return tables


def policy_snapshot(torch, eval_rows: list[dict[str, Any]], policy_logits, ref_log_probs, rewards) -> list[dict[str, Any]]:
    rows = []
    for i, row in enumerate(eval_rows):
        log_probs = torch.nn.functional.log_softmax(policy_logits[i], dim=-1)
        probs = torch.exp(log_probs)
        kl = torch.sum(probs * (log_probs - ref_log_probs[i]))
        expected_reward = torch.sum(probs * rewards[i])
        best_index = int(torch.argmax(probs).item())
        rows.append(
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "expected_reward": float(expected_reward.detach().item()),
                "kl_to_reference": float(kl.detach().item()),
                "top_candidate": row["candidates"][best_index]["id"],
                "top_candidate_expected_best": bool(row["candidates"][best_index].get("expected_best", False)),
                "candidates": [
                    {
                        "id": candidate["id"],
                        "probability": float(probs[j].detach().item()),
                        "reward": float(rewards[i][j].detach().item()),
                        "reference_probability": float(torch.exp(ref_log_probs[i][j]).detach().item()),
                        "expected_best": bool(candidate.get("expected_best", False)),
                        "text": candidate["text"],
                    }
                    for j, candidate in enumerate(row["candidates"])
                ],
            }
        )
    return rows


def aggregate_policy(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"top1_expected_best_rate": 0.0, "mean_expected_reward": 0.0, "mean_kl": 0.0}
    return {
        "top1_expected_best_rate": sum(1 for row in rows if row["top_candidate_expected_best"]) / len(rows),
        "mean_expected_reward": sum(row["expected_reward"] for row in rows) / len(rows),
        "mean_kl": sum(row["kl_to_reference"] for row in rows) / len(rows),
    }


def run_ppo(torch, ref_log_probs, rewards, steps: int, lr: float, clip_range: float, beta: float) -> tuple[Any, list[dict[str, float]]]:
    policy_logits = torch.nn.Parameter(ref_log_probs.detach().clone())
    optimizer = torch.optim.AdamW([policy_logits], lr=lr)
    history = []
    with torch.no_grad():
        old_log_probs = torch.nn.functional.log_softmax(policy_logits.detach(), dim=-1)
        old_probs = torch.exp(old_log_probs)
        baseline = torch.sum(old_probs * rewards, dim=-1, keepdim=True)
        advantages = rewards - baseline

    for step in range(steps):
        log_probs = torch.nn.functional.log_softmax(policy_logits, dim=-1)
        probs = torch.exp(log_probs)
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
        ppo_gain = torch.sum(old_probs * surrogate, dim=-1).mean()
        kl = torch.sum(probs * (log_probs - ref_log_probs), dim=-1).mean()
        loss = -ppo_gain + beta * kl

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            new_log_probs = torch.nn.functional.log_softmax(policy_logits, dim=-1)
            new_probs = torch.exp(new_log_probs)
            expected_reward = torch.sum(new_probs * rewards, dim=-1).mean()
            current_kl = torch.sum(new_probs * (new_log_probs - ref_log_probs), dim=-1).mean()
            clip_fraction = ((ratio < 1.0 - clip_range) | (ratio > 1.0 + clip_range)).float().mean()
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach().item()),
                    "ppo_gain": float(ppo_gain.detach().item()),
                    "mean_expected_reward": float(expected_reward.item()),
                    "mean_kl": float(current_kl.item()),
                    "clip_fraction": float(clip_fraction.item()),
                }
            )

    return policy_logits.detach(), history


def write_reward_weights(path: Path, bundle: RewardBundle) -> None:
    weights = bundle.model.weight.detach().cpu().squeeze(0).tolist()
    bias = float(bundle.model.bias.detach().cpu().item())
    write_json(
        path,
        {
            "feature_names": FEATURE_NAMES,
            "weights": dict(zip(FEATURE_NAMES, weights)),
            "bias": bias,
            "feature_mean": bundle.mean.squeeze(0).detach().cpu().tolist(),
            "feature_std": bundle.std.squeeze(0).detach().cpu().tolist(),
        },
    )


def write_report(
    report_path: Path,
    machine: dict[str, Any],
    args,
    tokenizer,
    model_name: str,
    reward_metrics: dict[str, Any],
    before_metrics: dict[str, float],
    after_metrics: dict[str, float],
    ppo_history: list[dict[str, float]],
    first_before: dict[str, Any],
    first_after: dict[str, Any],
) -> None:
    report = dedent(
        f"""
        # Lesson 09: Reward / RLHF Concept

        ## 本课目标

        本课讲 RLHF 中最容易混淆的五个对象：

        - reward model: 从同一 prompt 下的 chosen/rejected 偏好对学习一个标量评分函数 `r(x, y)`。
        - reference model: 通常是 SFT 后冻结的模型 `pi_ref`，用于限制 RL 更新不要偏离太远。
        - policy: 当前要优化的模型 `pi_theta`，RLHF/PPO 更新的就是它。
        - KL penalty: `KL(pi_theta || pi_ref)`，防止模型为了刷 reward 变成奇怪、啰嗦或不安全的分布。
        - PPO: 用 clipped ratio 控制每次 policy update 的步长，避免一次更新把策略推飞。

        本地执行不训练完整 LLM。脚本用真实 Hugging Face tiny causal LM 给候选回答算 reference mean-token logprob，
        再在候选回答分布上跑一个可观察的 PPO/KL 模拟。这样能保留 RLHF 的核心结构，同时不会做重型训练。

        ## 本机与本地 HF 路径

        - system: {machine["system"]}
        - machine: {machine["machine"]}
        - memory: {machine["memory_gb"]} GB
        - MPS available: {machine["mps_available"]}
        - reference model: `{model_name}`
        - tokenizer vocab size: {len(tokenizer)}
        - HF cache: `lessons/09-rlhf-reward/outputs/hf-cache`

        ## 数据与输出

        - preference data: `{args.preferences}`
        - eval candidates: `{args.eval_candidates}`
        - outputs dir: `{args.output_dir}`
        - trace: `{args.trace}`

        ## Reward Model 结果

        - preference pair 数: {reward_metrics["pair_count"]}
        - reward epochs: {reward_metrics["epochs"]}
        - preference accuracy: {reward_metrics["preference_accuracy"]:.2%}
        - mean chosen-rejected margin: {reward_metrics["mean_reward_margin"]:.4f}
        - final Bradley-Terry loss: {reward_metrics["final_reward_loss"]:.4f}

        本课的 reward model 是教学用的轻量线性模型，特征来自 prompt + response 的文本规则；真实系统通常用 transformer reward model 对完整 `(prompt, answer)` 编码。
        Reward model 不是“真理函数”。它只是把人类偏好压缩成一个标量。如果偏好数据有偏、reward 特征不够或模型被 policy 钻空子，
        高 reward 不一定代表高质量，所以后面必须有 KL、离线评估和人工复核。

        ## PPO/KL 模拟结果

        | 指标 | PPO 前 policy | PPO 后 policy |
        |---|---:|---:|
        | top-1 expected-best rate | {before_metrics["top1_expected_best_rate"]:.2%} | {after_metrics["top1_expected_best_rate"]:.2%} |
        | mean expected reward | {before_metrics["mean_expected_reward"]:.4f} | {after_metrics["mean_expected_reward"]:.4f} |
        | mean KL to reference | {before_metrics["mean_kl"]:.4f} | {after_metrics["mean_kl"]:.4f} |

        最后一步 PPO history:

        ```json
        {json.dumps(ppo_history[-1] if ppo_history else {}, ensure_ascii=False, indent=2)}
        ```

        ## 第一条 eval prompt 的策略变化

        Prompt:

        ```text
        {first_before.get("prompt", "")}
        ```

        PPO 前 top candidate: `{first_before.get("top_candidate")}`

        PPO 后 top candidate: `{first_after.get("top_candidate")}`

        PPO 后候选分布:

        ```json
        {json.dumps(first_after.get("candidates", []), ensure_ascii=False, indent=2)}
        ```

        ## 开发者级理解

        RLHF 的训练链路通常是：

        1. 先做 SFT，得到一个会遵守任务格式的初始 policy。
        2. 冻结一份 SFT model 作为 reference model。
        3. 收集同一 prompt 下多个回答的人类偏好，形成 chosen/rejected。
        4. 训练 reward model，让 `r(prompt, chosen) > r(prompt, rejected)`。
        5. 用 PPO 或类似算法更新 policy，让高 reward 回答概率上升。
        6. 同时加入 KL penalty，约束 policy 不要离 reference 太远。
        7. 用离线评估、红队样本和人工复核检查 reward hacking、安全性、格式稳定性和真实任务成功率。

        PPO 里最关键的量是 probability ratio：

        ```text
        ratio = pi_theta(action | prompt) / pi_old(action | prompt)
        objective = min(ratio * advantage, clip(ratio, 1-eps, 1+eps) * advantage) - beta * KL
        ```

        `advantage` 表示某个候选回答比旧 policy 的平均水平好多少。本课把旧 policy 固定为更新前的候选分布，因此 clipped ratio 会在多步更新中约束 policy 漂移；
        真实 PPO 会按 rollout 批次定期刷新 `pi_old`。本课在每个 prompt 的候选集合上精确计算分布；
        真实 RLHF 会从语言模型采样 token 序列，再对整段回答打 reward。

        ## 产物

        - `outputs/reward_weights.json`
        - `outputs/reference_scores.json`
        - `outputs/policy_before.jsonl`
        - `outputs/policy_after.jsonl`
        - `outputs/ppo_history.jsonl`
        - `outputs/metrics.json`
        - `outputs/trace.json`
        """
    ).strip()
    report = "\n".join(line[8:] if line.startswith("        ") else line for line in report.splitlines())
    report_path.write_text(report + "\n", encoding="utf-8")


def write_index(index_path: Path) -> None:
    html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Lesson 09 · Reward / RLHF Concept</title>
  <style>
    body { margin:0; background:#0b0d12; color:#eef2ff; font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif; line-height:1.65; }
    main { max-width:1120px; margin:0 auto; padding:44px 22px 80px; }
    h1 { font-size:40px; margin:0 0 10px; }
    h2 { margin-top:34px; color:#a7f3d0; }
    p, li { color:#aeb9cc; }
    code, pre { font-family:"SF Mono",Menlo,Consolas,monospace; }
    pre { overflow:auto; padding:16px; border:1px solid rgba(148,163,184,.2); border-radius:8px; background:#10151f; color:#e0f2fe; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:14px; margin:22px 0; }
    .card { border:1px solid rgba(148,163,184,.2); border-radius:8px; background:#111827; padding:16px; }
    .card strong { display:block; color:#f8fafc; margin-bottom:6px; }
    .flow { display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:10px; }
    .step { border:1px solid rgba(16,185,129,.28); border-radius:8px; background:#0f1b17; padding:12px; }
    .step span { display:block; color:#34d399; font-size:12px; font-weight:800; }
    a { color:#67e8f9; }
  </style>
</head>
<body>
  <main>
    <h1>Lesson 09 · Reward / RLHF Concept</h1>
    <p>用一个小而完整的候选回答模拟，把 reward model、reference model、policy、KL 和 PPO 放到同一条可执行链路里。</p>
    <div class="grid">
      <div class="card"><strong>偏好数据</strong><code>data/preference_pairs.jsonl</code><br>chosen/rejected 训练 reward model。</div>
      <div class="card"><strong>评估候选</strong><code>data/eval_candidates.jsonl</code><br>每个 prompt 三个候选回答。</div>
      <div class="card"><strong>HF 加载</strong><code>sshleifer/tiny-gpt2</code><br>真实 Hugging Face tiny causal LM。</div>
      <div class="card"><strong>产物</strong><code>outputs/</code><br>reward、policy、PPO history、trace。</div>
    </div>
    <h2>执行命令</h2>
    <pre>.venv/bin/python lessons/09-rlhf-reward/run.py --quick
.venv/bin/python lessons/09-rlhf-reward/run.py --trace-delay 0.5</pre>
    <h2>流程</h2>
    <div class="flow">
      <div class="step"><span>01</span>加载偏好对</div>
      <div class="step"><span>02</span>加载 HF tokenizer/model</div>
      <div class="step"><span>03</span>训练 reward model</div>
      <div class="step"><span>04</span>计算 reference 分布</div>
      <div class="step"><span>05</span>初始化 policy</div>
      <div class="step"><span>06</span>PPO + KL 更新</div>
      <div class="step"><span>07</span>输出评估报告</div>
    </div>
    <h2>关键概念</h2>
    <ul>
      <li><strong>Reward model</strong>: 把同一 prompt 下的偏好对压缩成回答分数；本课用 prompt + response 规则特征做教学模型。</li>
      <li><strong>Reference model</strong>: 冻结基线，提供 KL 约束的锚点。</li>
      <li><strong>Policy</strong>: 当前被优化的回答分布。</li>
      <li><strong>KL</strong>: 控制 policy 不要为了刷 reward 过度偏离 reference。</li>
      <li><strong>PPO</strong>: 用 clipped ratio 控制每次策略更新幅度；本课固定更新前候选分布作为 pi_old。</li>
    </ul>
    <h2>看什么结果</h2>
    <ul>
      <li>打开 <a href="report.md">report.md</a> 看 PPO 前后指标。</li>
      <li>打开 <a href="outputs/policy_before.jsonl">policy_before.jsonl</a> 和 <a href="outputs/policy_after.jsonl">policy_after.jsonl</a> 看候选概率变化。</li>
      <li>打开 <a href="outputs/trace.json">outputs/trace.json</a> 看每一步 trace event。</li>
    </ul>
  </main>
</body>
</html>
"""
    index_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="auto")
    parser.add_argument("--preferences", default="lessons/09-rlhf-reward/data/preference_pairs.jsonl")
    parser.add_argument("--eval-candidates", default="lessons/09-rlhf-reward/data/eval_candidates.jsonl")
    parser.add_argument("--output-dir", default="lessons/09-rlhf-reward/outputs")
    parser.add_argument("--report", default="lessons/09-rlhf-reward/report.md")
    parser.add_argument("--index", default="lessons/09-rlhf-reward/index.html")
    parser.add_argument("--trace", default="lessons/09-rlhf-reward/outputs/trace.json")
    parser.add_argument("--trace-delay", type=float, default=0.0)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reward-epochs", type=int, default=80)
    parser.add_argument("--ppo-steps", type=int, default=12)
    parser.add_argument("--reward-lr", type=float, default=0.08)
    parser.add_argument("--ppo-lr", type=float, default=0.35)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.08)
    parser.add_argument("--reference-temperature", type=float, default=2.0)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        args.reward_epochs = min(args.reward_epochs, 30)
        args.ppo_steps = min(args.ppo_steps, 5)
        args.max_length = min(args.max_length, 128)

    torch, AutoModelForCausalLM, AutoTokenizer, set_seed = require_stack()
    set_seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = resolve_project_path(args.output_dir)
    os.environ["HF_HOME"] = str(output_dir / "hf-cache")
    os.environ["HF_DATASETS_CACHE"] = str(output_dir / "hf-cache" / "datasets")
    os.environ["HF_XET_CACHE"] = str(output_dir / "hf-cache" / "xet")
    report_path = resolve_project_path(args.report)
    index_path = resolve_project_path(args.index)
    for path in [output_dir, report_path.parent, index_path.parent]:
        path.mkdir(parents=True, exist_ok=True)

    trace = LessonTrace(args.trace, args.trace_delay)
    machine = detect_local_config(torch)
    pairs = read_jsonl(args.preferences)
    eval_rows = read_jsonl(args.eval_candidates)

    print("Step 1: load preference and eval data")
    trace.event(
        "load preference data",
        "data",
        "读取 chosen/rejected 偏好对和固定 eval candidate 集合。",
        inputs={"preferences": args.preferences, "eval_candidates": args.eval_candidates},
        outputs={"preference_pairs": len(pairs), "eval_prompts": len(eval_rows)},
        sample=pairs[0],
    )

    print("Step 2: load real HF tokenizer and tiny reference model")
    model_name = resolve_reference_model_name(args.model_name)
    tokenizer, reference_model = load_hf_reference_model(AutoModelForCausalLM, AutoTokenizer, model_name)
    trace.event(
        "load HF tokenizer and reference model",
        "model",
        "通过 AutoTokenizer 和 AutoModelForCausalLM.from_pretrained 从 Hugging Face 加载真实 tiny causal LM。",
        inputs={"requested_model": args.model_name, "model_name": model_name},
        outputs={"tokenizer_vocab_size": len(tokenizer), "model_type": reference_model.config.model_type},
        metrics={"memory_gb": machine["memory_gb"], "mps_available": machine["mps_available"]},
    )

    print("Step 3: train reward model from preferences")
    reward_bundle, reward_metrics = train_reward_model(torch, pairs, tokenizer, args.reward_epochs, args.reward_lr)
    reward_metrics["pair_count"] = len(pairs)
    write_reward_weights(output_dir / "reward_weights.json", reward_bundle)
    trace.event(
        "train reward model",
        "reward",
        "用 Bradley-Terry / pairwise logistic loss 让 chosen 分数高于 rejected。",
        inputs={"feature_names": FEATURE_NAMES, "epochs": args.reward_epochs, "learning_rate": args.reward_lr},
        outputs={"reward_weights": output_dir / "reward_weights.json"},
        metrics={key: value for key, value in reward_metrics.items() if key != "history"},
        sample={"history": reward_metrics["history"]},
    )

    print("Step 4: compute reference probabilities for candidates")
    reference_tables = build_reference_tables(
        torch,
        reference_model,
        tokenizer,
        eval_rows,
        args.max_length,
        args.reference_temperature,
    )
    write_json(output_dir / "reference_scores.json", {"rows": reference_tables})
    ref_log_probs = torch.log(torch.tensor([row["reference_probs"] for row in reference_tables], dtype=torch.float32))
    rewards = torch.stack(
        [
            reward_bundle.score_texts(
                torch,
                tokenizer,
                [candidate["text"] for candidate in row["candidates"]],
                [row["prompt"]] * len(row["candidates"]),
            )
            for row in eval_rows
        ]
    )
    trace.event(
        "score candidates with reference and reward",
        "eval",
        "reference model 给候选回答算 completion mean-token logprob，reward model 给同一批 prompt/answer 候选打偏好分。",
        inputs={"candidate_prompts": len(eval_rows), "reference_temperature": args.reference_temperature},
        outputs={"reference_scores": output_dir / "reference_scores.json"},
        metrics={"mean_reward": float(rewards.mean().item()), "mean_reference_entropy": float((-(torch.exp(ref_log_probs) * ref_log_probs).sum(dim=-1)).mean().item())},
        sample=reference_tables[0],
    )

    print("Step 5: run candidate-level PPO with KL control")
    before_rows = policy_snapshot(torch, eval_rows, ref_log_probs.clone(), ref_log_probs, rewards)
    before_metrics = aggregate_policy(before_rows)
    policy_logits, ppo_history = run_ppo(
        torch,
        ref_log_probs,
        rewards.detach(),
        args.ppo_steps,
        args.ppo_lr,
        args.clip_range,
        args.beta,
    )
    after_rows = policy_snapshot(torch, eval_rows, policy_logits, ref_log_probs, rewards)
    after_metrics = aggregate_policy(after_rows)
    write_jsonl(output_dir / "policy_before.jsonl", before_rows)
    write_jsonl(output_dir / "policy_after.jsonl", after_rows)
    write_jsonl(output_dir / "ppo_history.jsonl", ppo_history)
    trace.event(
        "ppo update policy",
        "rl",
        "在候选回答分布上执行 PPO clipped objective，并加入 KL(pi_policy || pi_ref) 惩罚。",
        inputs={"ppo_steps": args.ppo_steps, "clip_range": args.clip_range, "beta": args.beta, "learning_rate": args.ppo_lr},
        outputs={
            "policy_before": output_dir / "policy_before.jsonl",
            "policy_after": output_dir / "policy_after.jsonl",
            "ppo_history": output_dir / "ppo_history.jsonl",
        },
        metrics={"before": before_metrics, "after": after_metrics},
        sample={"last_step": ppo_history[-1] if ppo_history else {}},
    )

    metrics = {
        "reward": {key: value for key, value in reward_metrics.items() if key != "history"},
        "policy_before": before_metrics,
        "policy_after": after_metrics,
        "ppo": {
            "steps": args.ppo_steps,
            "clip_range": args.clip_range,
            "beta": args.beta,
            "learning_rate": args.ppo_lr,
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    write_index(index_path)
    write_report(
        report_path,
        machine,
        args,
        tokenizer,
        model_name,
        reward_metrics,
        before_metrics,
        after_metrics,
        ppo_history,
        before_rows[0] if before_rows else {},
        after_rows[0] if after_rows else {},
    )
    trace.finish("Lesson 09 完成：reward model、reference、policy、KL 和 PPO 模拟都已生成本地可检查产物。", metrics)

    print("Reward preference accuracy:", f"{reward_metrics['preference_accuracy']:.2%}")
    print("Policy top-1 expected-best before:", f"{before_metrics['top1_expected_best_rate']:.2%}")
    print("Policy top-1 expected-best after:", f"{after_metrics['top1_expected_best_rate']:.2%}")
    print("Report written:", report_path.relative_to(PROJECT_ROOT))
    print("Trace written:", resolve_project_path(args.trace).relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
