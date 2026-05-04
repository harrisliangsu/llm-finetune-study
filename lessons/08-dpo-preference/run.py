#!/usr/bin/env python3
"""Lesson 08: manual DPO preference optimization without TRL."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
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

from lessons.common.hf_model_policy import detect_local_config, infer_lora_target_modules, resolve_model_name
from lessons.common.lesson_common import build_prompt, pad, resolve_project_path
from lessons.common.visual_trace import VisualTrace


def require_training_stack():
    try:
        import torch
        import torch.nn.functional as F
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    except ImportError as exc:
        raise SystemExit(
            "Lesson 08 requires torch, transformers, accelerate, and peft. "
            "Install the repository requirements in .venv before running this lesson."
        ) from exc

    return torch, F, LoraConfig, TaskType, get_peft_model, AutoModelForCausalLM, AutoTokenizer, set_seed


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line in resolve_project_path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def display_path(path: Path) -> Path | str:
    try:
        return path.relative_to(resolve_project_path("."))
    except ValueError:
        return str(path)


def clean_text(value: Any) -> str:
    return "\n".join(str(value).splitlines()).strip()


def choose_device(torch, requested: str):  # noqa: ANN001
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_tokenizer(AutoTokenizer, model_name: str):  # noqa: ANN001
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


def load_base_model(torch, AutoModelForCausalLM, model_name: str):  # noqa: ANN001
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    return model


def encode_completion(tokenizer, prompt: str, completion: str, max_length: int) -> dict[str, list[int]]:  # noqa: ANN001
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    completion_ids = tokenizer.encode(completion.strip(), add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        completion_ids = completion_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids

    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        prompt_drop = min(overflow, max(0, len(prompt_ids) - 8))
        prompt_ids = prompt_ids[prompt_drop:]
        labels = labels[prompt_drop:]
        input_ids = input_ids[prompt_drop:]
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    attention_mask = [1] * len(input_ids)
    completion_tokens = sum(1 for token_id in labels if token_id != -100)
    return {
        "input_ids": pad(input_ids, max_length, tokenizer.pad_token_id),
        "attention_mask": pad(attention_mask, max_length, 0),
        "labels": pad(labels, max_length, -100),
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": completion_tokens,
    }


def build_preference_features(tokenizer, rows: list[dict[str, Any]], max_length: int) -> list[dict[str, Any]]:  # noqa: ANN001
    features = []
    for row in rows:
        prompt = build_prompt(row["instruction"], row.get("input", ""))
        chosen = encode_completion(tokenizer, prompt, row["chosen"], max_length)
        rejected = encode_completion(tokenizer, prompt, row["rejected"], max_length)
        features.append(
            {
                "id": row["id"],
                "prompt": prompt,
                "chosen_text": row["chosen"],
                "rejected_text": row["rejected"],
                "criteria": row.get("criteria", ""),
                "chosen": chosen,
                "rejected": rejected,
            }
        )
    return features


def tensorize(torch, feature: dict[str, Any], device):  # noqa: ANN001
    chosen = feature["chosen"]
    rejected = feature["rejected"]
    return {
        "input_ids": torch.tensor([chosen["input_ids"], rejected["input_ids"]], dtype=torch.long, device=device),
        "attention_mask": torch.tensor(
            [chosen["attention_mask"], rejected["attention_mask"]], dtype=torch.long, device=device
        ),
        "labels": torch.tensor([chosen["labels"], rejected["labels"]], dtype=torch.long, device=device),
    }


def sequence_logps(torch, F, model, batch: dict[str, Any]) -> Any:  # noqa: ANN001
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = outputs.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)
    token_logps = F.log_softmax(logits, dim=-1).gather(dim=-1, index=safe_labels.unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask).sum(dim=-1)


def reference_logps(torch, F, model, batch: dict[str, Any]):  # noqa: ANN001
    was_training = model.training
    model.eval()
    with torch.no_grad():
        if hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                logps = sequence_logps(torch, F, model, batch)
        else:
            logps = sequence_logps(torch, F, model, batch)
    if was_training:
        model.train()
    return logps


def dpo_step(torch, F, model, batch: dict[str, Any], beta: float) -> dict[str, Any]:  # noqa: ANN001
    policy_logps = sequence_logps(torch, F, model, batch)
    ref_logps = reference_logps(torch, F, model, batch)
    policy_logratio = policy_logps[0] - policy_logps[1]
    ref_logratio = ref_logps[0] - ref_logps[1]
    logits = beta * (policy_logratio - ref_logratio)
    loss = -F.logsigmoid(logits)
    chosen_reward = beta * (policy_logps[0].detach() - ref_logps[0])
    rejected_reward = beta * (policy_logps[1].detach() - ref_logps[1])
    return {
        "loss": loss,
        "policy_chosen_logp": policy_logps[0].detach(),
        "policy_rejected_logp": policy_logps[1].detach(),
        "ref_chosen_logp": ref_logps[0],
        "ref_rejected_logp": ref_logps[1],
        "policy_logratio": policy_logratio.detach(),
        "ref_logratio": ref_logratio,
        "chosen_reward": chosen_reward,
        "rejected_reward": rejected_reward,
        "reward_margin": chosen_reward - rejected_reward,
        "preference_correct": (chosen_reward > rejected_reward).float(),
    }


def evaluate_preferences(torch, F, model, features: list[dict[str, Any]], beta: float, device) -> dict[str, Any]:  # noqa: ANN001
    rows = []
    losses = []
    margins = []
    ref_gaps = []
    correct = []
    was_training = model.training
    model.eval()
    for feature in features:
        batch = tensorize(torch, feature, device)
        with torch.no_grad():
            result = dpo_step(torch, F, model, batch, beta)
        row = {
            "id": feature["id"],
            "loss": float(result["loss"].detach().cpu()),
            "policy_chosen_logp": float(result["policy_chosen_logp"].detach().cpu()),
            "policy_rejected_logp": float(result["policy_rejected_logp"].detach().cpu()),
            "ref_chosen_logp": float(result["ref_chosen_logp"].detach().cpu()),
            "ref_rejected_logp": float(result["ref_rejected_logp"].detach().cpu()),
            "reward_margin": float(result["reward_margin"].detach().cpu()),
            "reference_preference_gap": float(result["ref_logratio"].detach().cpu()),
            "preference_correct": bool(result["preference_correct"].detach().cpu().item()),
        }
        rows.append(row)
        losses.append(row["loss"])
        margins.append(row["reward_margin"])
        ref_gaps.append(row["reference_preference_gap"])
        correct.append(1.0 if row["preference_correct"] else 0.0)
    if was_training:
        model.train()
    count = max(1, len(rows))
    return {
        "loss": sum(losses) / count,
        "reward_margin": sum(margins) / count,
        "reference_preference_gap": sum(ref_gaps) / count,
        "preference_accuracy": sum(correct) / count,
        "rows": rows,
    }


def greedy_generate(torch, model, tokenizer, prompt: str, max_new_tokens: int, disable_adapter: bool = False) -> str:  # noqa: ANN001
    model.eval()
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    context = model.disable_adapter() if disable_adapter and hasattr(model, "disable_adapter") else nullcontext()
    with torch.no_grad(), context:
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(generated[0].detach().cpu().tolist(), skip_special_tokens=True)


class nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, traceback):  # noqa: ANN001
        return False


def strip_prompt(generated: str) -> str:
    marker = "### Response:"
    if marker in generated:
        return generated.split(marker, 1)[1].strip()
    return generated.strip()


def generate_eval_rows(torch, model, tokenizer, prompts: list[dict[str, Any]], max_new_tokens: int, disable_adapter: bool) -> list[dict[str, Any]]:  # noqa: ANN001
    rows = []
    for item in prompts:
        prompt = build_prompt(item["instruction"], item.get("input", ""))
        generated = greedy_generate(torch, model, tokenizer, prompt, max_new_tokens, disable_adapter=disable_adapter)
        rows.append({**item, "prompt": prompt, "generated": generated, "answer": strip_prompt(generated)})
    return rows


def count_trainable_parameters(model) -> tuple[int, int]:  # noqa: ANN001
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def finite_metric(value: float) -> float:
    if math.isfinite(value):
        return value
    return 0.0


def write_report(
    report_path: Path,
    model_name: str,
    machine: dict[str, Any],
    device: str,
    data_path: str,
    eval_path: str,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    max_steps: int,
    max_length: int,
    beta: float,
    learning_rate: float,
    target_modules: list[str],
    rank: int,
    alpha: int,
    trainable_params: int,
    total_params: int,
    before_eval: dict[str, Any],
    after_eval: dict[str, Any],
    step_logs: list[dict[str, Any]],
    before_generations: list[dict[str, Any]],
    after_generations: list[dict[str, Any]],
    adapter_dir: Path,
) -> None:
    first_before = before_generations[0] if before_generations else {}
    first_after = after_generations[0] if after_generations else {}
    first_step = step_logs[0] if step_logs else {}
    last_step = step_logs[-1] if step_logs else {}
    report = dedent(
        f"""
        # Lesson 08: DPO Preference Optimization

        ## 本课目标

        本课用客服回复偏好对实现一个最小 DPO。它不依赖 `trl`，只用 `transformers`、`torch` 和 `peft`：

        - policy = base model + LoRA adapter
        - reference = 同一个 base model 禁用 adapter
        - chosen/rejected completion 都只在回答 token 上计算 log probability
        - DPO loss 直接用 `-logsigmoid(beta * (pi_logratio - ref_logratio))`

        ## 本机和模型选择

        - system: {machine["system"]}
        - machine: {machine["machine"]}
        - memory: {machine["memory_gb"]} GB
        - MPS available: {machine["mps_available"]}
        - selected model: `{model_name}`
        - device: `{device}`
        - HF cache: `lessons/08-dpo-preference/outputs/hf-cache`

        ## 数据和训练配置

        - preference data: `{data_path}`
        - eval prompts: `{eval_path}`
        - preference rows used: {len(train_rows)}
        - eval prompts: {len(eval_rows)}
        - max_length: {max_length}
        - max_steps: {max_steps}
        - beta: {beta}
        - learning rate: {learning_rate}
        - target modules: `{target_modules}`
        - LoRA rank `r`: {rank}
        - LoRA alpha: {alpha}
        - trainable params: {trainable_params}
        - total params: {total_params}
        - trainable ratio: {trainable_params / total_params:.4%}
        - adapter dir: `{display_path(adapter_dir)}`

        ## 偏好指标

        | 指标 | DPO 前 | DPO 后 |
        |---|---:|---:|
        | DPO loss | {before_eval["loss"]:.4f} | {after_eval["loss"]:.4f} |
        | reward margin | {before_eval["reward_margin"]:.4f} | {after_eval["reward_margin"]:.4f} |
        | preference accuracy | {before_eval["preference_accuracy"]:.2%} | {after_eval["preference_accuracy"]:.2%} |
        | reference chosen-rejected logp gap | {before_eval["reference_preference_gap"]:.4f} | {after_eval["reference_preference_gap"]:.4f} |

        `reference chosen-rejected logp gap` 是 base model 本来对 chosen/rejected 的偏好。DPO 真正优化的是 policy 相对 reference 的 reward margin。

        ## 第一步和最后一步训练日志

        | 字段 | 第一步 | 最后一步 |
        |---|---:|---:|
        | loss | {finite_metric(float(first_step.get("loss", 0.0))):.4f} | {finite_metric(float(last_step.get("loss", 0.0))):.4f} |
        | reward_margin | {finite_metric(float(first_step.get("reward_margin", 0.0))):.4f} | {finite_metric(float(last_step.get("reward_margin", 0.0))):.4f} |
        | policy_logratio | {finite_metric(float(first_step.get("policy_logratio", 0.0))):.4f} | {finite_metric(float(last_step.get("policy_logratio", 0.0))):.4f} |
        | ref_logratio | {finite_metric(float(first_step.get("ref_logratio", 0.0))):.4f} | {finite_metric(float(last_step.get("ref_logratio", 0.0))):.4f} |

        ## 固定 prompt 生成示例

        输入：

        ```text
        {first_before.get("input", "-")}
        ```

        DPO 前输出：

        ````text
        {clean_text(first_before.get("answer", "-"))}
        ````

        DPO 后输出：

        ````text
        {clean_text(first_after.get("answer", "-"))}
        ````

        ## 每一步的作用、输入、输出

        | 步骤 | 作用 | 输入 | 输出 |
        |---|---|---|---|
        | 选择模型 | 根据本机配置选择真实 HF 模型 | local config + `--model-name` | `{model_name}` |
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
  <title>Lesson 08 · DPO Preference Optimization</title>
  <style>
    body { margin:0; background:#0a0c10; color:#f4f7fb; font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif; line-height:1.65; }
    main { max-width:1120px; margin:0 auto; padding:48px 22px 80px; }
    h1 { font-size:40px; margin:0 0 10px; }
    h2 { margin-top:34px; color:#bef264; }
    p, li { color:#aeb8c8; }
    code, pre { font-family:"SF Mono",Menlo,Consolas,monospace; }
    pre { overflow:auto; padding:16px; border:1px solid rgba(174,184,200,.2); border-radius:8px; background:#10151d; color:#e5edff; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:14px; margin:22px 0; }
    .card { border:1px solid rgba(174,184,200,.2); border-radius:8px; background:#111820; padding:16px; }
    .card strong { display:block; margin-bottom:6px; color:#ffffff; }
    .flow { display:grid; grid-template-columns:repeat(auto-fit,minmax(128px,1fr)); gap:10px; }
    .step { border:1px solid rgba(190,242,100,.24); border-radius:8px; padding:12px; background:#101a14; }
    .step span { display:block; color:#bef264; font-size:12px; font-weight:800; letter-spacing:.08em; }
    a { color:#bef264; }
  </style>
</head>
<body>
  <main>
    <h1>Lesson 08 · DPO Preference Optimization</h1>
    <p>目标：不用 TRL，在本地用 transformers + torch + PEFT 手写一个最小 DPO 偏好优化循环。</p>
    <div class="grid">
      <div class="card"><strong>偏好数据</strong><code>data/preferences.jsonl</code><br>chosen / rejected 成对回答。</div>
      <div class="card"><strong>模型</strong><code>--model-name auto</code><br>本机默认 Qwen 0.5B。</div>
      <div class="card"><strong>Reference</strong><code>disable_adapter()</code><br>同一 base 禁用 LoRA。</div>
      <div class="card"><strong>产物</strong><code>outputs/</code><br>adapter、metrics、generations。</div>
    </div>
    <h2>执行命令</h2>
    <pre>.venv/bin/python lessons/08-dpo-preference/run.py --quick --trace-delay 0.2</pre>
    <h2>DPO 流程</h2>
    <div class="flow">
      <div class="step"><span>01</span>选择 HF 模型</div>
      <div class="step"><span>02</span>加载 tokenizer/base</div>
      <div class="step"><span>03</span>挂 LoRA policy</div>
      <div class="step"><span>04</span>构造偏好 batch</div>
      <div class="step"><span>05</span>计算 ref logps</div>
      <div class="step"><span>06</span>计算 policy logps</div>
      <div class="step"><span>07</span>DPO loss 更新</div>
      <div class="step"><span>08</span>生成并写报告</div>
    </div>
    <h2>看什么结果</h2>
    <ul>
      <li>打开 <a href="report.md">report.md</a> 看 reward margin 和 preference accuracy。</li>
      <li>对比 <a href="outputs/generations/before.jsonl">before.jsonl</a> 与 <a href="outputs/generations/after.jsonl">after.jsonl</a>。</li>
      <li>用 visualizer 查看 <code>visualizer/traces/08-dpo-preference.json</code> 的 DPO step 事件。</li>
    </ul>
  </main>
</body>
</html>
"""
    index_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="auto")
    parser.add_argument("--data", default="lessons/08-dpo-preference/data/preferences.jsonl")
    parser.add_argument("--eval-prompts", default="lessons/08-dpo-preference/data/eval_prompts.jsonl")
    parser.add_argument("--report", default="lessons/08-dpo-preference/report.md")
    parser.add_argument("--index", default="lessons/08-dpo-preference/index.html")
    parser.add_argument("--adapter-dir", default="lessons/08-dpo-preference/outputs/adapter")
    parser.add_argument("--generation-dir", default="lessons/08-dpo-preference/outputs/generations")
    parser.add_argument("--max-length", type=int, default=160)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--trace", default="visualizer/traces/live.json")
    parser.add_argument("--trace-delay", type=float, default=0.0)
    args = parser.parse_args()

    if args.quick:
        args.max_steps = min(args.max_steps, 2)
        args.max_new_tokens = min(args.max_new_tokens, 32)
        args.max_length = min(args.max_length, 128)

    torch, F, LoraConfig, TaskType, get_peft_model, AutoModelForCausalLM, AutoTokenizer, set_seed = require_training_stack()
    set_seed(42)

    report_path = resolve_project_path(args.report)
    index_path = resolve_project_path(args.index)
    adapter_dir = resolve_project_path(args.adapter_dir)
    generation_dir = resolve_project_path(args.generation_dir)
    for path in [LESSON_OUTPUTS, report_path.parent, index_path.parent, adapter_dir, generation_dir]:
        path.mkdir(parents=True, exist_ok=True)

    machine = detect_local_config(torch)
    model_name = resolve_model_name(args.model_name, machine)
    device = choose_device(torch, args.device)
    trace = VisualTrace("08-dpo-preference", "Lesson 08 · DPO Preference Optimization", args.trace, args.trace_delay)

    trace.event(
        "select model",
        "setup",
        "根据本机配置选择真实 Hugging Face causal LM；默认 auto 在当前 Mac 上选择 Qwen 0.5B。",
        inputs={"requested_model": args.model_name, "machine": machine, "quick": args.quick},
        outputs={"model_name": model_name, "device": str(device), "hf_cache": os.environ["HF_HOME"]},
    )

    print("Step 1: load tokenizer and base model")
    tokenizer = load_tokenizer(AutoTokenizer, model_name)
    base_model = load_base_model(torch, AutoModelForCausalLM, model_name)
    base_model.to(device)
    trace.event(
        "load tokenizer and base model",
        "model",
        "通过 AutoTokenizer/AutoModelForCausalLM 加载真实 HF 模型，后续不使用手写模型。",
        inputs={"model_name": model_name},
        outputs={"tokenizer_vocab_size": len(tokenizer), "pad_token_id": tokenizer.pad_token_id},
        model={"base": model_name, "device": str(device), "adapter": "none"},
    )

    print("Step 2: attach LoRA policy")
    target_modules = infer_lora_target_modules(model_name)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    model.config.use_cache = False
    trainable_params, total_params = count_trainable_parameters(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    trace.event(
        "attach LoRA policy",
        "model",
        "policy 是 base + LoRA；reference 用同一个 PEFT 模型禁用 adapter，避免加载第二份 0.5B 模型。",
        inputs={"target_modules": target_modules, "rank": args.rank, "alpha": args.alpha, "dropout": args.dropout},
        outputs={"trainable_params": trainable_params, "total_params": total_params},
        model={"policy": "base + trainable LoRA", "reference": "base with adapter disabled"},
    )

    print("Step 3: load preference data")
    all_rows = read_jsonl(args.data)
    eval_prompts = read_jsonl(args.eval_prompts)
    if args.quick:
        all_rows = all_rows[:4]
        eval_prompts = eval_prompts[:2]
    features = build_preference_features(tokenizer, all_rows, args.max_length)
    trace.event(
        "build preference batches",
        "data",
        "把每条样本编码成 prompt+chosen 和 prompt+rejected。prompt token 的 label=-100，只统计 completion logps。",
        inputs={"data": args.data, "eval_prompts": args.eval_prompts, "max_length": args.max_length},
        outputs={"preference_rows": len(features), "eval_prompts": len(eval_prompts)},
        tensors=[
            {"name": "chosen input_ids", "shape": [args.max_length]},
            {"name": "rejected input_ids", "shape": [args.max_length]},
            {"name": "completion labels", "shape": [args.max_length]},
        ],
        sample={
            "id": features[0]["id"],
            "chosen": features[0]["chosen_text"],
            "rejected": features[0]["rejected_text"],
            "criteria": features[0]["criteria"],
        },
    )

    print("Step 4: evaluate and generate before DPO")
    before_eval = evaluate_preferences(torch, F, model, features, args.beta, device)
    write_jsonl(LESSON_OUTPUTS / "preference_scores_before.jsonl", before_eval["rows"])
    before_generations = generate_eval_rows(
        torch, model, tokenizer, eval_prompts, args.max_new_tokens, disable_adapter=True
    )
    write_jsonl(generation_dir / "before.jsonl", before_generations)
    trace.event(
        "evaluate before DPO",
        "eval",
        "adapter 初始时 policy 近似 reference，reward margin 通常接近 0。",
        outputs={
            "preference_scores": LESSON_OUTPUTS / "preference_scores_before.jsonl",
            "generations": generation_dir / "before.jsonl",
        },
        metrics={
            "loss": before_eval["loss"],
            "reward_margin": before_eval["reward_margin"],
            "preference_accuracy": before_eval["preference_accuracy"],
            "reference_preference_gap": before_eval["reference_preference_gap"],
        },
    )

    print("Step 5: train manual DPO loop")
    step_logs: list[dict[str, Any]] = []
    model.train()
    for step in range(args.max_steps):
        feature = features[step % len(features)]
        batch = tensorize(torch, feature, device)
        optimizer.zero_grad(set_to_none=True)
        result = dpo_step(torch, F, model, batch, args.beta)
        result["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        log = {
            "step": step + 1,
            "sample_id": feature["id"],
            "loss": float(result["loss"].detach().cpu()),
            "policy_logratio": float(result["policy_logratio"].detach().cpu()),
            "ref_logratio": float(result["ref_logratio"].detach().cpu()),
            "reward_margin": float(result["reward_margin"].detach().cpu()),
            "chosen_reward": float(result["chosen_reward"].detach().cpu()),
            "rejected_reward": float(result["rejected_reward"].detach().cpu()),
            "preference_correct": bool(result["preference_correct"].detach().cpu().item()),
        }
        step_logs.append(log)
        print(
            f"step {log['step']:02d} loss={log['loss']:.4f} "
            f"margin={log['reward_margin']:.4f} sample={log['sample_id']}"
        )
        trace.event(
            f"DPO step {step + 1}",
            "train",
            "手写 DPO 更新：policy chosen/rejected logratio 相对 reference logratio 的差越大，loss 越低。",
            inputs={"sample_id": feature["id"], "beta": args.beta, "learning_rate": args.learning_rate},
            outputs={
                "loss": log["loss"],
                "policy_logratio": log["policy_logratio"],
                "ref_logratio": log["ref_logratio"],
                "reward_margin": log["reward_margin"],
            },
            metrics=log,
        )

    print("Step 6: evaluate and generate after DPO")
    after_eval = evaluate_preferences(torch, F, model, features, args.beta, device)
    write_jsonl(LESSON_OUTPUTS / "preference_scores_after.jsonl", after_eval["rows"])
    after_generations = generate_eval_rows(
        torch, model, tokenizer, eval_prompts, args.max_new_tokens, disable_adapter=False
    )
    write_jsonl(generation_dir / "after.jsonl", after_generations)
    trace.event(
        "evaluate after DPO",
        "eval",
        "训练后重新计算整批偏好对的 reward margin，并生成同一批固定 prompts。",
        outputs={
            "preference_scores": LESSON_OUTPUTS / "preference_scores_after.jsonl",
            "generations": generation_dir / "after.jsonl",
        },
        metrics={
            "loss": after_eval["loss"],
            "reward_margin": after_eval["reward_margin"],
            "preference_accuracy": after_eval["preference_accuracy"],
            "reference_preference_gap": after_eval["reference_preference_gap"],
        },
    )

    print("Step 7: save outputs")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    metrics = {
        "model_name": model_name,
        "machine": machine,
        "device": str(device),
        "quick": args.quick,
        "max_steps": args.max_steps,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "target_modules": target_modules,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "before": {key: value for key, value in before_eval.items() if key != "rows"},
        "after": {key: value for key, value in after_eval.items() if key != "rows"},
        "steps": step_logs,
    }
    (LESSON_OUTPUTS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(
        report_path,
        model_name,
        machine,
        str(device),
        args.data,
        args.eval_prompts,
        all_rows,
        eval_prompts,
        args.max_steps,
        args.max_length,
        args.beta,
        args.learning_rate,
        target_modules,
        args.rank,
        args.alpha,
        trainable_params,
        total_params,
        before_eval,
        after_eval,
        step_logs,
        before_generations,
        after_generations,
        adapter_dir,
    )
    write_index(index_path)
    trace.event(
        "save adapter and reports",
        "checkpoint",
        "保存 LoRA adapter、tokenizer、metrics、report 和 generation JSONL。",
        outputs={
            "adapter_dir": adapter_dir,
            "metrics": LESSON_OUTPUTS / "metrics.json",
            "report": report_path,
            "index": index_path,
        },
        model={"adapter_saved": True, "base_saved": False},
    )
    trace.finish(
        "Lesson 08 完成：manual DPO objective、偏好指标、固定 prompt 生成和 adapter 保存都已写出。",
        metrics={
            "loss_after": after_eval["loss"],
            "reward_margin_after": after_eval["reward_margin"],
            "preference_accuracy_after": after_eval["preference_accuracy"],
        },
    )
    print(f"Report written: {display_path(report_path)}")


if __name__ == "__main__":
    main()
