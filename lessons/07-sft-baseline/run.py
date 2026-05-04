#!/usr/bin/env python3
"""Lesson 07: SFT baseline on a strict JSON ticket routing task."""

from __future__ import annotations

import argparse
import inspect
import json
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
from lessons.common.lesson_common import (
    build_prompt,
    decode_learned_labels,
    load_sft_splits,
    make_sft_tokenize_fn,
    read_sft_records,
    resolve_project_path,
)
from lessons.common.visual_trace import VisualTrace, make_trainer_trace_callback


def require_training_stack():
    try:
        import torch
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator, set_seed
    except ImportError as exc:
        raise SystemExit(
            "Lesson 07 requires torch, transformers, datasets, accelerate, and peft. "
            "Install them in .venv before running this lesson."
        ) from exc

    return (
        torch,
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
        set_seed,
    )


def make_training_arguments(TrainingArguments, **kwargs):
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
    return TrainingArguments(**{key: value for key, value in kwargs.items() if key in parameters})


def load_tokenizer(AutoTokenizer, model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(torch, AutoModelForCausalLM, model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    return model


def prepare_dataset(tokenizer, data_path: str, max_length: int):
    splits = load_sft_splits(data_path, test_size=0.25, seed=42)
    tokenized = splits.map(
        make_sft_tokenize_fn(tokenizer, max_length),
        batched=True,
        remove_columns=splits["train"].column_names,
    )
    trainer_dataset = tokenized.remove_columns(["prompt_length", "answer_length"]).with_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )
    return splits, tokenized, trainer_dataset


def trainable_parameter_counts(model) -> tuple[int, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    return trainable, total


def display_path(path: Path) -> Path | str:
    try:
        return path.relative_to(resolve_project_path("."))
    except ValueError:
        return str(path)


def clean_multiline(value: Any) -> str:
    return "\n".join(line.rstrip() for line in str(value).splitlines())


def load_eval_prompts(path: str | Path) -> list[dict[str, str]]:
    rows = []
    for line in resolve_project_path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def prompt_for_ticket(text: str) -> str:
    return build_prompt(
        "把用户工单路由成严格 JSON，字段为 intent、priority、department、summary。不要输出 JSON 以外的文字。",
        text,
    )


def greedy_generate(torch, model, tokenizer, prompt: str, max_new_tokens: int = 96) -> str:
    model.eval()
    device = next(model.parameters()).device
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(generated[0].detach().cpu().tolist(), skip_special_tokens=True)


def strip_prompt(generated: str) -> str:
    marker = "### Response:"
    if marker in generated:
        return generated.split(marker, 1)[1].strip()
    return generated.strip()


def try_parse_json(text: str) -> dict[str, Any] | None:
    content = strip_prompt(text)
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        parsed = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def is_strict_json_only(text: str) -> bool:
    content = strip_prompt(text).strip()
    if not (content.startswith("{") and content.endswith("}")):
        return False
    return try_parse_json(content) is not None


def score_generation(generated: str, expected: dict[str, str]) -> dict[str, Any]:
    parsed = try_parse_json(generated)
    required = {"intent", "priority", "department", "summary"}
    return {
        "valid_json": parsed is not None,
        "strict_json_only": is_strict_json_only(generated),
        "has_required_fields": bool(parsed and required.issubset(parsed.keys())),
        "intent_match": bool(parsed and parsed.get("intent") == expected.get("expected_intent")),
        "department_match": bool(parsed and parsed.get("department") == expected.get("expected_department")),
        "parsed": parsed,
    }


def evaluate_prompts(torch, model, tokenizer, prompts: list[dict[str, str]], max_new_tokens: int) -> list[dict[str, Any]]:
    rows = []
    for item in prompts:
        prompt = prompt_for_ticket(item["input"])
        generated = greedy_generate(torch, model, tokenizer, prompt, max_new_tokens=max_new_tokens)
        rows.append(
            {
                **item,
                "prompt": prompt,
                "generated": generated,
                "answer": strip_prompt(generated),
                "score": score_generation(generated, item),
            }
        )
    return rows


def aggregate_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "valid_json_rate": 0.0,
            "strict_json_only_rate": 0.0,
            "required_fields_rate": 0.0,
            "intent_match_rate": 0.0,
            "department_match_rate": 0.0,
        }
    return {
        "valid_json_rate": sum(1 for row in rows if row["score"]["valid_json"]) / len(rows),
        "strict_json_only_rate": sum(1 for row in rows if row["score"]["strict_json_only"]) / len(rows),
        "required_fields_rate": sum(1 for row in rows if row["score"]["has_required_fields"]) / len(rows),
        "intent_match_rate": sum(1 for row in rows if row["score"]["intent_match"]) / len(rows),
        "department_match_rate": sum(1 for row in rows if row["score"]["department_match"]) / len(rows),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_report(
    report_path: Path,
    model_name: str,
    machine: dict,
    data_path: str,
    eval_path: str,
    raw_rows: list[dict[str, str]],
    splits,
    tokenized,
    target_modules: list[str],
    rank: int,
    alpha: int,
    trainable_params: int,
    total_params: int,
    eval_before: dict,
    train_metrics: dict,
    eval_after: dict,
    before_scores: dict[str, float],
    after_scores: dict[str, float],
    adapter_dir: Path,
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
    loaded_rows: list[dict[str, Any]],
    sample_label_text: str,
    max_steps: int,
    max_length: int,
) -> None:
    before_example = before_rows[0] if before_rows else {}
    after_example = after_rows[0] if after_rows else {}
    loaded_example = loaded_rows[0] if loaded_rows else {}
    report = dedent(
        f"""
        # Lesson 07: SFT Baseline

        ## 本课目标

        本课用“客服工单 -> 严格 JSON 路由”做独立 SFT baseline。它比通用概念解释更适合观察训练效果，因为可以直接检查：

        - 是否输出合法 JSON
        - 是否包含 `intent/priority/department/summary`
        - `intent` 和 `department` 是否和固定 eval prompt 的期望一致
        - 是否输出 JSON 以外的废话

        ## 本机和模型选择

        - system: {machine["system"]}
        - machine: {machine["machine"]}
        - memory: {machine["memory_gb"]} GB
        - MPS available: {machine["mps_available"]}
        - selected model: `{model_name}`
        - HF cache: `lessons/07-sft-baseline/outputs/hf-cache`

        选择 `{model_name}` 的原因：本机 32GB + MPS 可以承受 0.5B 级模型短步数 LoRA/SFT。课程默认从 Hugging Face 下载真实模型，不再自己生成模型。

        ## 数据和输出

        - train data: `{data_path}`
        - eval prompts: `{eval_path}`
        - raw rows: {len(raw_rows)}
        - train rows: {len(splits["train"])}
        - validation rows: {len(splits["validation"])}
        - max_length: {max_length}
        - max_steps: {max_steps}
        - target modules: `{target_modules}`
        - LoRA rank `r`: {rank}
        - LoRA alpha: {alpha}
        - trainable params: {trainable_params}
        - total params: {total_params}
        - trainable ratio: {trainable_params / total_params:.4%}
        - eval loss before training: {eval_before.get("eval_loss")}
        - train loss: {train_metrics.get("train_loss")}
        - eval loss after training: {eval_after.get("eval_loss")}
        - adapter dir: `{adapter_dir}`

        ## 固定 eval prompts 的格式指标

        | 指标 | 训练前 | 训练后 |
        |---|---:|---:|
        | extractable JSON rate | {before_scores["valid_json_rate"]:.2%} | {after_scores["valid_json_rate"]:.2%} |
        | strict JSON-only rate | {before_scores["strict_json_only_rate"]:.2%} | {after_scores["strict_json_only_rate"]:.2%} |
        | required fields rate | {before_scores["required_fields_rate"]:.2%} | {after_scores["required_fields_rate"]:.2%} |
        | intent match rate | {before_scores["intent_match_rate"]:.2%} | {after_scores["intent_match_rate"]:.2%} |
        | department match rate | {before_scores["department_match_rate"]:.2%} | {after_scores["department_match_rate"]:.2%} |

        这组指标要分开看：

        - `extractable JSON rate` 说明输出里能不能抽出一个 JSON 对象；base model 本来就可能做到。
        - `strict JSON-only rate` 说明输出是否只剩 JSON，没有 Markdown、解释文字或继续补写下一段 prompt；这是本课最直观的 SFT 效果。
        - `intent/department match rate` 是更难的业务分类指标。短步数 LoRA/SFT 先学会格式约束，精确 taxonomy 还需要更明确的标签集合、更高质量数据或更多训练步数。

        ## 第 1 条训练样本的 label 检查

        下面是 `labels != -100` 解码后的目标回答，确认 prompt 没有参与 loss，只有 JSON answer 被学习：

        ```text
        {sample_label_text}
        ```

        > 报告生成时完整 label 内容写在 trace 的 `build SFT dataset` 事件中；学习时重点看 `labels = -100` 只 mask prompt，answer token 才计算 loss。

        ## 固定 prompt 对比示例

        输入：

        ```text
        {before_example.get("input", "-")}
        ```

        训练前输出：

        ````text
        {clean_multiline(before_example.get("answer", "-"))}
        ````

        训练后输出：

        ````text
        {clean_multiline(after_example.get("answer", "-"))}
        ````

        重新加载 adapter 后输出：

        ````text
        {clean_multiline(loaded_example.get("answer", "-"))}
        ````

        ## 每一步的作用、输入、输出

        | 步骤 | 作用 | 输入 | 输出 |
        |---|---|---|---|
        | 选择模型 | 根据本机 32GB/MPS 选择真实 HF 模型 | local config | `{model_name}` |
        | 加载 tokenizer/base | 下载或读取 HF cache | model id | tokenizer + base model |
        | 构造 SFT dataset | 把 ticket 文本和 JSON answer 转成训练字段 | JSONL | `input_ids/attention_mask/labels` |
        | 训练前生成 | 固定 eval prompts 先跑 base | base model + prompts | before generations |
        | 挂 LoRA | 冻结 base，只训练 adapter | target modules/r/alpha | PEFT model |
        | Trainer train | 执行 SFT 参数更新 | PEFT model + tokenized data | loss/checkpoint |
        | 训练后生成 | 同一批 prompts 再跑训练后模型 | PEFT model + prompts | after generations |
        | 保存 adapter | 保存 adapter-only artifact | PEFT model | adapter dir |
        | 重新加载 adapter | 模拟部署路径 | fresh base + adapter | loaded generations |

        ## 产物

        - `outputs/generations/before.jsonl`
        - `outputs/generations/after.jsonl`
        - `outputs/generations/loaded.jsonl`
        - `outputs/metrics.json`
        - `outputs/adapter/`

        ## 下一步

        完成 SFT baseline 后，再进入 Lesson 08: DPO。DPO 不包含 SFT，它通常接在 SFT 后，用 chosen/rejected 偏好对继续优化模型偏好。
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
  <title>Lesson 07 · SFT Baseline</title>
  <style>
    body { margin:0; background:#090b10; color:#eef3ff; font-family:-apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif; line-height:1.65; }
    main { max-width:1120px; margin:0 auto; padding:48px 22px 80px; }
    h1 { font-size:42px; margin:0 0 10px; }
    h2 { margin-top:34px; color:#bae6fd; }
    p, li { color:#a8b3c7; }
    code, pre { font-family:"SF Mono",Menlo,Consolas,monospace; }
    pre { overflow:auto; padding:16px; border:1px solid rgba(148,163,184,.18); border-radius:8px; background:#0d131d; color:#dbeafe; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:14px; margin:22px 0; }
    .card { border:1px solid rgba(148,163,184,.18); border-radius:8px; background:#101722; padding:16px; }
    .card strong { display:block; margin-bottom:6px; color:#f8fafc; }
    .flow { display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:10px; }
    .step { border:1px solid rgba(56,189,248,.22); border-radius:8px; padding:12px; background:#0d1723; }
    .step span { display:block; color:#38bdf8; font-size:12px; font-weight:800; letter-spacing:.08em; }
    a { color:#7dd3fc; }
  </style>
</head>
<body>
  <main>
    <h1>Lesson 07 · SFT Baseline</h1>
    <p>目标：用真实 Hugging Face Qwen 0.5B + LoRA，在本地训练一个客服工单到严格 JSON 的 SFT baseline。</p>
    <div class="grid">
      <div class="card"><strong>数据</strong><code>data/train.jsonl</code><br>40 条 instruction/input/output。</div>
      <div class="card"><strong>评估</strong><code>data/eval_prompts.jsonl</code><br>固定 6 条 prompt，对比训练前后。</div>
      <div class="card"><strong>模型</strong><code>Qwen/Qwen2.5-0.5B-Instruct</code><br>按本机 32GB/MPS 选择。</div>
      <div class="card"><strong>产物</strong><code>outputs/adapter/</code><br>adapter、generations、metrics。</div>
    </div>
    <h2>执行命令</h2>
    <pre>.venv/bin/python lessons/07-sft-baseline/run.py --trace-delay 0.5</pre>
    <h2>训练流程</h2>
    <div class="flow">
      <div class="step"><span>01</span>选择 HF 模型</div>
      <div class="step"><span>02</span>加载 tokenizer/base</div>
      <div class="step"><span>03</span>构造 SFT labels</div>
      <div class="step"><span>04</span>训练前生成</div>
      <div class="step"><span>05</span>挂 LoRA adapter</div>
      <div class="step"><span>06</span>Trainer SFT</div>
      <div class="step"><span>07</span>训练后生成</div>
      <div class="step"><span>08</span>保存并重新加载 adapter</div>
    </div>
    <h2>关键概念</h2>
    <p><strong>SFT</strong> 是用标准答案训练模型。这里标准答案是严格 JSON。</p>
    <p><strong>loss mask</strong> 是让 prompt 的 labels 等于 <code>-100</code>，只让 answer token 计算 loss。</p>
    <p><strong>LoRA</strong> 是参数高效训练方式，本课为了在本地跑 Qwen，只训练 adapter，不更新 base model。</p>
    <p><strong>adapter</strong> 是训练后的增量权重，部署时用同一个 base model 加载 adapter。</p>
    <h2>看什么结果</h2>
    <ul>
      <li>打开 <a href="report.md">report.md</a> 看训练前后 JSON 格式指标。</li>
      <li>打开 <a href="outputs/generations/before.jsonl">before.jsonl</a> 和 <a href="outputs/generations/after.jsonl">after.jsonl</a> 看固定 prompt 输出。</li>
      <li>打开可视化页看 <code>visualizer/traces/07-sft-baseline.json</code> 的数据流和模型变化。</li>
    </ul>
  </main>
</body>
</html>
"""
    index_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="auto")
    parser.add_argument("--data", default="lessons/07-sft-baseline/data/train.jsonl")
    parser.add_argument("--eval-prompts", default="lessons/07-sft-baseline/data/eval_prompts.jsonl")
    parser.add_argument("--report", default="lessons/07-sft-baseline/report.md")
    parser.add_argument("--index", default="lessons/07-sft-baseline/index.html")
    parser.add_argument("--output-dir", default="lessons/07-sft-baseline/outputs/trainer")
    parser.add_argument("--adapter-dir", default="lessons/07-sft-baseline/outputs/adapter")
    parser.add_argument("--generation-dir", default="lessons/07-sft-baseline/outputs/generations")
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--trace", default="visualizer/traces/live.json")
    parser.add_argument("--trace-delay", type=float, default=0.0)
    args = parser.parse_args()

    (
        torch,
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        default_data_collator,
        set_seed,
    ) = require_training_stack()
    set_seed(42)

    machine = detect_local_config(torch)
    model_name = resolve_model_name(args.model_name, machine)

    report_path = resolve_project_path(args.report)
    index_path = resolve_project_path(args.index)
    output_dir = resolve_project_path(args.output_dir)
    adapter_dir = resolve_project_path(args.adapter_dir)
    generation_dir = resolve_project_path(args.generation_dir)
    for path in [report_path.parent, index_path.parent, output_dir, adapter_dir, generation_dir]:
        path.mkdir(parents=True, exist_ok=True)

    trace = VisualTrace("07-sft-baseline", "Lesson 07 · SFT Baseline", args.trace, args.trace_delay)
    raw_rows = read_sft_records(args.data)
    eval_prompts = load_eval_prompts(args.eval_prompts)

    print("Step 0: select local Hugging Face model")
    print("machine:", machine)
    print("model:", model_name)
    trace.event(
        "select model",
        "setup",
        "根据本机配置选择真实 Hugging Face instruct causal LM。默认 32GB/MPS 使用 Qwen 0.5B。",
        inputs={"requested_model": args.model_name, "machine": machine},
        outputs={"model_name": model_name, "hf_cache": os.environ["HF_HOME"]},
    )

    print("\nStep 1: load tokenizer and base model")
    tokenizer = load_tokenizer(AutoTokenizer, model_name)
    base_model = load_base_model(torch, AutoModelForCausalLM, model_name)
    trace.event(
        "load tokenizer and base model",
        "model",
        "从 Hugging Face 下载或读取缓存，加载 tokenizer 和 base model。",
        inputs={"model_name": model_name},
        outputs={"tokenizer_vocab_size": len(tokenizer), "pad_token_id": tokenizer.pad_token_id},
        model={"base": model_name, "adapter": "none", "cache": os.environ["HF_HOME"]},
    )

    print("\nStep 2: build SFT dataset")
    splits, tokenized, trainer_dataset = prepare_dataset(tokenizer, args.data, args.max_length)
    sample_label_text = decode_learned_labels(tokenizer, tokenized["train"][0]["labels"])
    trace.event(
        "build SFT dataset",
        "data",
        "把客服工单 JSONL 转成 SFT 训练字段。prompt 区域 labels=-100，JSON answer 参与 loss。",
        inputs={"data": args.data, "eval_prompts": args.eval_prompts, "max_length": args.max_length},
        outputs={"raw_rows": len(raw_rows), "train_rows": len(splits["train"]), "validation_rows": len(splits["validation"])},
        tensors=[
            {"name": "input_ids", "shape": [args.max_length]},
            {"name": "attention_mask", "shape": [args.max_length]},
            {"name": "labels", "shape": [args.max_length]},
        ],
        sample={"learned_label_text": sample_label_text, **raw_rows[0]},
    )

    print("\nStep 3: generate before SFT")
    before_rows = evaluate_prompts(torch, base_model, tokenizer, eval_prompts, args.max_new_tokens)
    before_scores = aggregate_scores(before_rows)
    write_jsonl(generation_dir / "before.jsonl", before_rows)
    trace.event(
        "generate before SFT",
        "generation",
        "固定 eval prompts 跑 base model，记录训练前输出和 JSON 指标。",
        inputs={"eval_prompts": args.eval_prompts},
        outputs={"generation_file": generation_dir / "before.jsonl", **before_scores, "first_answer": before_rows[0]["answer"]},
        metrics=before_scores,
    )

    target_modules = infer_lora_target_modules(model_name)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=target_modules,
        bias="none",
    )
    peft_model = get_peft_model(base_model, lora_config)
    peft_model.config.use_cache = False
    trainable_params, total_params = trainable_parameter_counts(peft_model)
    trace.event(
        "attach LoRA for SFT",
        "model",
        "为了本地可执行，本课用 LoRA 执行 SFT：base 冻结，只训练所选 target modules 上的 adapter。",
        inputs={"target_modules": target_modules, "rank": args.rank, "alpha": args.alpha, "dropout": args.dropout},
        outputs={"trainable_params": trainable_params, "total_params": total_params},
        model={
            "base": "frozen",
            "adapter": "trainable",
            "target_modules": target_modules,
            "trainable_ratio": f"{trainable_params / total_params:.4%}",
        },
    )

    training_args = make_training_arguments(
        TrainingArguments,
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=2,
        eval_steps=8,
        save_strategy="no",
        report_to=[],
        disable_tqdm=True,
        seed=42,
        data_seed=42,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=trainer_dataset["train"],
        eval_dataset=trainer_dataset["validation"],
        data_collator=default_data_collator,
        callbacks=[make_trainer_trace_callback(trace, "SFT Trainer")],
    )

    print("\nStep 4: evaluate before SFT")
    eval_before = trainer.evaluate()
    print(eval_before)
    trace.event(
        "evaluate before SFT",
        "eval",
        "训练前在 validation split 上测 loss。",
        outputs={"eval_loss": eval_before.get("eval_loss")},
        metrics=eval_before,
    )

    print("\nStep 5: train SFT LoRA adapter")
    train_output = trainer.train()
    print(train_output.metrics)
    trace.event(
        "train SFT adapter",
        "train",
        "Trainer 用客服工单 JSON 标准答案更新 LoRA adapter。",
        inputs={"max_steps": args.max_steps, "learning_rate": args.learning_rate},
        outputs={"train_loss": train_output.metrics.get("train_loss")},
        metrics=train_output.metrics,
        model={"base": "frozen", "updated": ["q_proj LoRA", "v_proj LoRA"]},
    )

    print("\nStep 6: evaluate after SFT")
    eval_after = trainer.evaluate()
    print(eval_after)
    trace.event(
        "evaluate after SFT",
        "eval",
        "训练后再次评估 validation loss。",
        outputs={"eval_loss": eval_after.get("eval_loss")},
        metrics=eval_after,
    )

    print("\nStep 7: generate after SFT")
    after_rows = evaluate_prompts(torch, peft_model, tokenizer, eval_prompts, args.max_new_tokens)
    after_scores = aggregate_scores(after_rows)
    write_jsonl(generation_dir / "after.jsonl", after_rows)
    trace.event(
        "generate after SFT",
        "generation",
        "同一批 fixed eval prompts 跑训练后模型，直接观察 JSON 格式和 intent/department。",
        inputs={"eval_prompts": args.eval_prompts},
        outputs={"generation_file": generation_dir / "after.jsonl", **after_scores, "first_answer": after_rows[0]["answer"]},
        metrics=after_scores,
    )

    print("\nStep 8: save adapter")
    peft_model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    trace.event(
        "save SFT adapter",
        "checkpoint",
        "保存 adapter-only checkpoint 和 tokenizer，保持课程输出自包含。",
        outputs={"adapter_dir": adapter_dir},
        model={"saved": ["adapter_config.json", "adapter_model.safetensors"], "base_saved": False},
    )

    print("\nStep 9: reload adapter and generate")
    fresh_base = load_base_model(torch, AutoModelForCausalLM, model_name)
    loaded_model = PeftModel.from_pretrained(fresh_base, str(adapter_dir))
    loaded_rows = evaluate_prompts(torch, loaded_model, tokenizer, eval_prompts, args.max_new_tokens)
    write_jsonl(generation_dir / "loaded.jsonl", loaded_rows)
    trace.event(
        "reload adapter and compare",
        "checkpoint",
        "重新加载 fresh base + adapter，验证部署路径和训练后输出路径一致。",
        inputs={"model_name": model_name, "adapter_dir": adapter_dir},
        outputs={"generation_file": generation_dir / "loaded.jsonl", "first_answer": loaded_rows[0]["answer"]},
        model={"base": "fresh", "adapter": "loaded"},
    )

    metrics = {
        "model_name": model_name,
        "machine": machine,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "eval_before": eval_before,
        "train": train_output.metrics,
        "eval_after": eval_after,
        "generation_before": before_scores,
        "generation_after": after_scores,
    }
    (LESSON_OUTPUTS / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_report(
        report_path,
        model_name,
        machine,
        args.data,
        args.eval_prompts,
        raw_rows,
        splits,
        tokenized,
        target_modules,
        args.rank,
        args.alpha,
        trainable_params,
        total_params,
        eval_before,
        train_output.metrics,
        eval_after,
        before_scores,
        after_scores,
        display_path(adapter_dir),
        before_rows,
        after_rows,
        loaded_rows,
        sample_label_text,
        args.max_steps,
        args.max_length,
    )
    write_index(index_path)

    trace.finish(
        "Lesson 07 完成：SFT baseline、训练前后固定 prompt 对比、adapter 保存加载都已可视化。",
        metrics={"valid_json_after": after_scores["valid_json_rate"], "intent_match_after": after_scores["intent_match_rate"]},
    )
    print(f"\nReport written: {display_path(report_path)}")


if __name__ == "__main__":
    main()
