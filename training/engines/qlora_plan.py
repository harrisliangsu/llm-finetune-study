#!/usr/bin/env python3
"""Studio QLoRA and training engineering planner.

This engine intentionally does not import bitsandbytes or launch CUDA training.
It turns the QLoRA engineering checklist into a local, executable planning
artifact: model policy, optional HF config metadata, memory budget estimates,
and clear Mac/MPS versus CUDA boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

LESSON_ID = "studio-qlora-plan"
LESSON_TITLE = "Studio · QLoRA / Training Engineering"
ENGINE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUTS = ENGINE_DIR / "outputs"
os.environ["HF_HOME"] = str(DEFAULT_OUTPUTS / "hf-cache")
os.environ["HF_DATASETS_CACHE"] = str(DEFAULT_OUTPUTS / "hf-cache" / "datasets")
os.environ["HF_XET_CACHE"] = str(DEFAULT_OUTPUTS / "hf-cache" / "xet")

from training.common.extra_args import known_extra_args
from training.common.hf_model_policy import detect_local_config, infer_lora_target_modules, resolve_model_name
from training.common.run_common import resolve_project_path
from training.common.visual_trace import VisualTrace


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    label: str
    params_b: float
    hidden_size: int
    num_layers: int
    vocab_size: int
    source: str
    why: str

    @property
    def params(self) -> int:
        return int(self.params_b * 1_000_000_000)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(resolve_project_path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def try_import_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def try_load_hf_metadata(model_id: str, local_files_only: bool) -> dict[str, Any]:
    """Load lightweight config/tokenizer metadata. Never downloads model weights."""

    metadata: dict[str, Any] = {
        "model_id": model_id,
        "available": False,
        "local_files_only": local_files_only,
        "error": None,
    }
    try:
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:
        metadata["error"] = f"transformers unavailable: {exc}"
        return metadata

    try:
        config = AutoConfig.from_pretrained(model_id, local_files_only=local_files_only, trust_remote_code=False)
        metadata.update(
            {
                "available": True,
                "architectures": getattr(config, "architectures", None),
                "model_type": getattr(config, "model_type", None),
                "hidden_size": getattr(config, "hidden_size", None) or getattr(config, "n_embd", None),
                "num_hidden_layers": getattr(config, "num_hidden_layers", None) or getattr(config, "n_layer", None),
                "num_attention_heads": getattr(config, "num_attention_heads", None) or getattr(config, "n_head", None),
                "vocab_size": getattr(config, "vocab_size", None),
                "torch_dtype": str(getattr(config, "torch_dtype", None)),
            }
        )
    except Exception as exc:  # noqa: BLE001 - metadata is optional by design.
        metadata["error"] = f"config unavailable: {exc}"
        return metadata

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            local_files_only=local_files_only,
            use_fast=True,
            trust_remote_code=False,
        )
        metadata["tokenizer_available"] = True
        metadata["tokenizer_vocab_size"] = len(tokenizer)
        metadata["model_max_length"] = getattr(tokenizer, "model_max_length", None)
    except Exception as exc:  # noqa: BLE001
        metadata["tokenizer_available"] = False
        metadata["tokenizer_error"] = str(exc)

    return metadata


def resolve_candidate_specs(
    config: dict[str, Any],
    selected_model_id: str,
    hf_metadata: dict[str, dict[str, Any]],
) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    candidate_by_id = {item["id"]: item for item in config["candidate_models"]}
    for item in config["candidate_models"]:
        model_id = selected_model_id if item["id"] == "auto" else item["id"]
        fallback = candidate_by_id.get(model_id, item)
        metadata = hf_metadata.get(model_id, {})
        hidden_size = metadata.get("hidden_size") or fallback["fallback_hidden_size"]
        num_layers = metadata.get("num_hidden_layers") or fallback["fallback_num_layers"]
        vocab_size = metadata.get("tokenizer_vocab_size") or metadata.get("vocab_size") or fallback["fallback_vocab_size"]
        source = "hf_config" if metadata.get("available") else "planning_config_fallback"
        if item["id"] == "auto" and fallback is not item:
            source = f"{source}:matched_{fallback['id']}"
        specs.append(
            ModelSpec(
                model_id=model_id,
                label=item["label"] if item["id"] != "auto" else f"{item['label']} -> {fallback['label']}",
                params_b=float(fallback["fallback_params_b"]),
                hidden_size=int(hidden_size),
                num_layers=int(num_layers),
                vocab_size=int(vocab_size),
                source=source,
                why=item["why"] if item["id"] != "auto" else f"{item['why']} Matched fallback: {fallback['why']}",
            )
        )
    return specs


def dedupe_specs_by_model_id(specs: list[ModelSpec]) -> list[ModelSpec]:
    unique: list[ModelSpec] = []
    seen: set[str] = set()
    for spec in specs:
        if spec.model_id in seen:
            continue
        unique.append(spec)
        seen.add(spec.model_id)
    return unique


def gb(bytes_value: float) -> float:
    return bytes_value / (1024**3)


def estimate_lora_params(spec: ModelSpec, rank: int, target_module_count: int) -> int:
    # For a square projection, LoRA trains A and B: rank * in + out * rank.
    # Most attention/MLP projections are close enough to hidden x hidden for planning.
    return int(spec.num_layers * target_module_count * (2 * rank * spec.hidden_size))


def estimate_activation_gb(
    spec: ModelSpec,
    seq_length: int,
    micro_batch: int,
    activation_multiplier: float,
    bytes_per_activation: float = 2.0,
) -> float:
    raw = micro_batch * seq_length * spec.hidden_size * spec.num_layers * bytes_per_activation
    return gb(raw * activation_multiplier)


def plan_memory(
    spec: ModelSpec,
    profile: dict[str, Any],
    defaults: dict[str, Any],
    seq_length: int,
    per_device_batch_size: int,
    grad_accum: int,
    rank: int,
    target_module_count: int,
    gradient_checkpointing: bool,
    local_memory_gb: float,
) -> dict[str, Any]:
    lora_params = estimate_lora_params(spec, rank, target_module_count)
    world_size = max(1, int(profile.get("world_size", 1) or 1))
    shard_parameters = bool(profile.get("shard_parameters", False))
    shard_gradients = bool(profile.get("shard_gradients", False))
    shard_optimizer = bool(profile.get("shard_optimizer", False))
    base_weight_gb_raw = gb(spec.params * float(profile["weight_bytes_per_param"]))
    lora_weight_gb_raw = gb(lora_params * 2)
    lora_grad_gb_raw = gb(lora_params * 2)
    lora_adam_gb_raw = gb(lora_params * 8)
    base_weight_gb = base_weight_gb_raw / world_size if shard_parameters else base_weight_gb_raw
    lora_weight_gb = lora_weight_gb_raw / world_size if shard_parameters else lora_weight_gb_raw
    lora_grad_gb = lora_grad_gb_raw / world_size if shard_gradients else lora_grad_gb_raw
    lora_adam_gb = lora_adam_gb_raw / world_size if shard_optimizer else lora_adam_gb_raw
    activation_multiplier = (
        float(defaults["checkpoint_activation_multiplier"])
        if gradient_checkpointing
        else float(defaults["activation_multiplier"])
    )
    activation_gb = estimate_activation_gb(
        spec,
        seq_length=seq_length,
        micro_batch=per_device_batch_size,
        activation_multiplier=activation_multiplier,
    )
    safety_margin_gb = float(defaults["safety_margin_gb"])
    peak_gb = base_weight_gb + lora_weight_gb + lora_grad_gb + lora_adam_gb + activation_gb + safety_margin_gb
    target_vram_gb = float(profile.get("target_vram_gb", 0) or 0)
    local_limit = local_memory_gb * float(profile.get("system_ram_headroom", 0.75))
    fit_limit = target_vram_gb or local_limit

    if profile["device_family"] == "mps_or_cpu" and profile["quantization"] != "none":
        recommendation = "invalid: bitsandbytes QLoRA is not treated as native MPS."
    elif profile["device_family"] == "mps_or_cpu":
        recommendation = "local planning only: use fp16/bf16/fp32 LoRA on MPS/CPU, or move QLoRA to CUDA."
    elif peak_gb <= fit_limit:
        recommendation = "likely fits, then validate with a one-step CUDA smoke run."
    elif profile["device_family"] == "cuda":
        recommendation = "too tight for one GPU; reduce sequence/batch/rank or move to sharding."
    elif profile["device_family"] == "cuda_distributed":
        recommendation = (
            f"distributed estimate only: assumes world_size={world_size} sharding for configured states; "
            "verify ZeRO/FSDP, PEFT, quantization, and save/load compatibility with a real cluster smoke run."
        )
    else:
        recommendation = "distributed planning case; verify compatibility with the exact trainer stack."

    return {
        "model_id": spec.model_id,
        "label": spec.label,
        "profile": profile["name"],
        "device_family": profile["device_family"],
        "quantization": profile["quantization"],
        "params_b": spec.params_b,
        "metadata_source": spec.source,
        "hidden_size": spec.hidden_size,
        "num_layers": spec.num_layers,
        "seq_length": seq_length,
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": grad_accum,
        "effective_batch_size_per_gpu": per_device_batch_size * grad_accum,
        "gradient_checkpointing": gradient_checkpointing,
        "world_size": world_size,
        "shard_parameters": shard_parameters,
        "shard_gradients": shard_gradients,
        "shard_optimizer": shard_optimizer,
        "lora_rank": rank,
        "target_module_count": target_module_count,
        "estimated_lora_params": lora_params,
        "base_weight_gb_raw": round(base_weight_gb_raw, 3),
        "base_weight_gb": round(base_weight_gb, 3),
        "lora_weight_gb": round(lora_weight_gb, 3),
        "lora_grad_gb": round(lora_grad_gb, 3),
        "lora_adam_gb": round(lora_adam_gb, 3),
        "activation_gb": round(activation_gb, 3),
        "safety_margin_gb": round(safety_margin_gb, 3),
        "estimated_peak_gb": round(peak_gb, 3),
        "fit_limit_gb": round(fit_limit, 3),
        "fits_limit": peak_gb <= fit_limit,
        "recommendation": recommendation,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_boundaries(machine: dict[str, Any]) -> dict[str, str]:
    has_mps = bool(machine.get("mps_available"))
    return {
        "qlora": "QLoRA = frozen base model loaded in 4-bit NF4 + small trainable LoRA adapters + paged optimizers.",
        "quantization": "Quantization reduces storage/compute dtype for frozen weights; it does not make optimizer or activations free.",
        "mac_mps": (
            "MPS is available, but this engine does not present bitsandbytes 4-bit QLoRA as native on Mac."
            if has_mps
            else "No MPS detected; local execution remains a CPU/Mac planning path, not QLoRA training."
        ),
        "cuda_bitsandbytes": "The common HF QLoRA path depends on CUDA-oriented bitsandbytes 4-bit kernels.",
        "gradient_checkpointing": "Checkpointing lowers activation memory by recomputing forward segments during backward.",
        "gradient_accumulation": "Accumulation raises effective batch size across microbatches but does not multiply peak activation memory.",
        "deepspeed": "DeepSpeed is a distributed training boundary for sharding states; it is not a Mac substitute for CUDA QLoRA kernels.",
    }


def write_report(
    report_path: Path,
    machine: dict[str, Any],
    selected_model: str,
    specs: list[ModelSpec],
    plans: list[dict[str, Any]],
    metadata: dict[str, Any],
    boundaries: dict[str, str],
    paths: dict[str, Path],
) -> None:
    selected_rows = [row for row in plans if row["model_id"] == selected_model]
    qlora_rows = [row for row in selected_rows if row["profile"] == "single_cuda_qlora_24gb"]
    mac_rows = [row for row in selected_rows if row["profile"] == "mac_mps_lora_planning"]
    qlora = qlora_rows[0] if qlora_rows else {}
    mac = mac_rows[0] if mac_rows else {}
    table_rows = "\n".join(
        f"| `{row['model_id']}` | `{row['profile']}` | {row['estimated_peak_gb']} | {row['fit_limit_gb']} | {row['fits_limit']} | {row['recommendation']} |"
        for row in plans
    )
    metadata_note = metadata.get(selected_model, {})
    spec_lines = "\n".join(
        f"- `{spec.model_id}`: {spec.params_b}B params, hidden={spec.hidden_size}, layers={spec.num_layers}, source={spec.source}"
        for spec in specs
    )
    report = dedent(
        f"""
        # Studio: QLoRA / Training Engineering

        ## Studio 定位

        这是本地可执行的训练工程规划器，不是 CUDA 训练脚本。它会用共享模型策略 helper 选择模型，尽量读取 Hugging Face config/tokenizer 元数据，然后估算不同训练路径的显存/内存预算。

        关键边界：Mac/MPS 可以做小模型 LoRA/SFT 学习，但本引擎不会把 bitsandbytes 4-bit QLoRA 伪装成 Mac 原生能力。标准 HF QLoRA 路径应该放到 CUDA 环境验证。

        ## 本机检测

        - system: {machine.get("system")}
        - machine: {machine.get("machine")}
        - processor: {machine.get("processor")}
        - memory: {machine.get("memory_gb")} GB
        - MPS built: {machine.get("mps_built")}
        - MPS available: {machine.get("mps_available")}
        - selected model by policy: `{selected_model}`
        - selected metadata available: {metadata_note.get("available")}
        - selected metadata error: `{metadata_note.get("error")}`

        ## 必须理解的概念

        - QLoRA: {boundaries["qlora"]}
        - Quantization: {boundaries["quantization"]}
        - CUDA/bitsandbytes: {boundaries["cuda_bitsandbytes"]}
        - Mac/MPS: {boundaries["mac_mps"]}
        - Gradient checkpointing: {boundaries["gradient_checkpointing"]}
        - Gradient accumulation: {boundaries["gradient_accumulation"]}
        - DeepSpeed: {boundaries["deepspeed"]}

        ## 当前策略模型预算

        - Mac/MPS local profile estimated peak: {mac.get("estimated_peak_gb")} GB
        - CUDA QLoRA 24GB profile estimated peak: {qlora.get("estimated_peak_gb")} GB
        - LoRA rank: {mac.get("lora_rank")}
        - sequence length: {mac.get("seq_length")}
        - per-device micro batch: {mac.get("per_device_batch_size")}
        - gradient accumulation steps: {mac.get("gradient_accumulation_steps")}
        - effective batch per GPU: {mac.get("effective_batch_size_per_gpu")}
        - gradient checkpointing: {mac.get("gradient_checkpointing")}

        ## 候选模型元数据

        {spec_lines}

        ## 预算表

        | model | profile | estimated peak GB | fit limit GB | fits | recommendation |
        |---|---:|---:|---:|---:|---|
        {table_rows}

        ## 估算公式

        这些数字用于训练方案初筛，不替代真实 profiler：

        ```text
        base_weight_gb = params * bytes_per_param
        qlora_nf4_double_quant ~= params * 0.56 bytes
        lora_params ~= layers * target_module_count * 2 * rank * hidden_size
        lora_train_state ~= lora_weights + lora_grads + AdamW moments
        activation_gb ~= batch * seq * hidden * layers * bytes * multiplier
        peak ~= base_weights + lora_train_state + activations + safety_margin
        ```

        梯度累积改变的是有效 batch，不是单个 microbatch 的激活峰值。要降显存峰值，优先调 `per_device_batch_size`、`seq_length`、gradient checkpointing、rank、target modules。

        ## DeepSpeed 边界

        DeepSpeed ZeRO-2/ZeRO-3 解决的是多 GPU 上 optimizer/gradient/parameter state 的切分。`multi_cuda_deepspeed_zero3` 只是按配置的 `world_size` 做 per-GPU 初筛估算，不替代真实 profiler。它不能让 Mac MPS 运行 bitsandbytes CUDA kernel，也不能自动消除 QLoRA 与 ZeRO/FSDP 的兼容性限制。工程上应该先用单 GPU one-step smoke test 验证模型加载、loss、保存，再扩展到 DeepSpeed。

        ## 产物

        - `{paths["plan_json"]}`
        - `{paths["plan_csv"]}`
        - `{paths["metadata_json"]}`
        - `{paths["trace_json"]}`
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
  <title>Studio · QLoRA / Training Engineering</title>
  <style>
    :root { --bg:#09100f; --panel:#111a18; --panel2:#15211f; --ink:#eef8f4; --muted:#9bb0aa; --line:rgba(238,248,244,.12); --green:#2dd4bf; --blue:#60a5fa; --gold:#fbbf24; --red:#fb7185; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif; line-height:1.65; }
    code, pre, .mono { font-family:"SF Mono",Menlo,Consolas,monospace; }
    .nav { position:sticky; top:0; display:flex; justify-content:space-between; align-items:center; min-height:54px; padding:0 22px; background:rgba(9,16,15,.88); border-bottom:1px solid var(--line); backdrop-filter:blur(16px); z-index:2; }
    .brand { font-weight:900; } .brand span { color:var(--green); }
    .nav a { color:var(--muted); text-decoration:none; margin-left:16px; font-size:13px; font-weight:800; }
    main { max-width:1120px; margin:0 auto; padding:58px 22px 82px; }
    h1 { margin:0 0 14px; font-size:48px; line-height:1.08; letter-spacing:0; } h1 span, h2 span { color:var(--green); }
    h2 { margin:0 0 12px; font-size:28px; line-height:1.2; }
    p, li { color:var(--muted); }
    .lead { font-size:17px; max-width:880px; }
    .badge { display:inline-flex; gap:8px; align-items:center; padding:6px 12px; border:1px solid var(--line); border-radius:999px; color:var(--muted); font-size:12px; font-weight:900; margin-bottom:20px; }
    .dot { width:7px; height:7px; border-radius:50%; background:var(--green); }
    section { padding:34px 0; }
    .grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; }
    .grid3 { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }
    .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:18px; min-height:132px; }
    .card strong { display:block; color:var(--ink); margin-bottom:7px; font-size:16px; }
    .metric strong { font-size:30px; line-height:1; color:var(--green); }
    .rail { display:grid; grid-template-columns:1fr 44px 1fr 44px 1fr; gap:12px; align-items:stretch; }
    .node { background:var(--panel2); border:1px solid var(--line); border-radius:8px; padding:16px; }
    .node b { display:block; margin-bottom:8px; color:var(--ink); }
    .arrow { display:flex; align-items:center; justify-content:center; color:var(--green); font-size:24px; font-weight:900; }
    .warn { border-color:rgba(251,113,133,.38); background:rgba(251,113,133,.08); }
    .ok { border-color:rgba(45,212,191,.34); background:rgba(45,212,191,.08); }
    .code { overflow:auto; padding:16px; border-radius:8px; border:1px solid var(--line); background:#07100e; color:#d7fff7; font-size:13px; }
    table { width:100%; border-collapse:collapse; font-size:14px; }
    th, td { padding:11px; border-bottom:1px solid var(--line); text-align:left; color:var(--muted); vertical-align:top; }
    th { color:var(--ink); }
    @media (max-width:900px){ h1{font-size:35px}.grid,.grid3,.rail{grid-template-columns:1fr}.arrow{display:none}.nav a{display:none} }
  </style>
</head>
<body>
  <nav class="nav"><div class="brand"><span>LLM</span> Finetune Study</div><div><a href="#boundary">边界</a><a href="#memory">预算</a><a href="#knobs">训练旋钮</a><a href="#run">运行</a></div></nav>
  <main>
    <div class="badge"><span class="dot"></span> STUDIO · QLORA ENGINEERING</div>
    <h1><span>QLoRA</span><br>不是 Mac 上的魔法开关</h1>
    <p class="lead">Studio 用共享训练引擎做工程规划：检测硬件、选择模型策略、读取安全的 Hugging Face 元数据，并估算 LoRA/QLoRA/DeepSpeed 方案的内存边界。</p>

    <section id="boundary">
      <h2>先划清 <span>执行边界</span></h2>
      <div class="grid">
        <div class="card ok"><strong>QLoRA</strong><p>4-bit NF4 冻结 base 权重，加 LoRA adapter 训练小增量。</p></div>
        <div class="card ok"><strong>CUDA + bitsandbytes</strong><p>标准 HF QLoRA 路径依赖 CUDA 方向的 4-bit kernel。</p></div>
        <div class="card warn"><strong>Mac / MPS</strong><p>可以学习 LoRA 和做规划，但不要声称 bitsandbytes QLoRA 原生可跑。</p></div>
        <div class="card"><strong>DeepSpeed</strong><p>多 GPU 状态切分边界，不是 CUDA kernel 的替代品。</p></div>
      </div>
    </section>

    <section id="memory">
      <h2>预算拆成 <span>四块</span></h2>
      <div class="rail">
        <div class="node"><b>Base weights</b><p>fp16 LoRA 约 2 bytes/param；NF4 double quant 规划可按约 0.56 bytes/param 初筛。</p></div>
        <div class="arrow">→</div>
        <div class="node"><b>Adapter train state</b><p>LoRA A/B 权重、梯度、AdamW 两个 fp32 moments。通常远小于 base。</p></div>
        <div class="arrow">→</div>
        <div class="node"><b>Activations</b><p>由 microbatch、seq length、hidden、layers 决定；checkpointing 用重算换显存。</p></div>
      </div>
      <pre class="code">peak ~= base_weights + lora_weights + lora_grads + optimizer_states + activations + safety_margin</pre>
    </section>

    <section id="knobs">
      <h2>训练工程 <span>旋钮</span></h2>
      <div class="grid3">
        <div class="card metric"><strong>microbatch</strong><p>直接影响激活峰值。OOM 时先降它。</p></div>
        <div class="card metric"><strong>accum</strong><p>提高有效 batch，不等价于提高单步峰值。</p></div>
        <div class="card metric"><strong>checkpoint</strong><p>降低激活保存量，但 backward 更慢。</p></div>
      </div>
    </section>

    <section>
      <h2>DeepSpeed 什么时候进场</h2>
      <table>
        <tr><th>场景</th><th>优先选择</th><th>注意</th></tr>
        <tr><td>7B 单卡 QLoRA</td><td>CUDA + bitsandbytes + PEFT</td><td>先 one-step smoke test。</td></tr>
        <tr><td>13B/长上下文紧张</td><td>降 seq/batch/rank，开 checkpointing</td><td>先别把分布式复杂度拉进来。</td></tr>
        <tr><td>多卡全参或大 LoRA</td><td>DeepSpeed ZeRO / FSDP</td><td>确认量化、PEFT、保存格式兼容。</td></tr>
      </table>
    </section>

    <section id="run">
      <h2>运行</h2>
      <pre class="code">.venv/bin/python visualizer/studio/run.py --method qlora-plan ...</pre>
      <p>输出写入本次 Studio run 目录，包括 <code>plans.json</code>、<code>memory_budget.csv</code>、<code>hf_metadata.json</code> 和 <code>trace.json</code>。</p>
    </section>
  </main>
</body>
</html>
"""
    index_path.write_text(html, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/engines/data/qlora-planning-config.json")
    parser.add_argument("--model-name", default="auto")
    parser.add_argument("--report", default="visualizer/runtime/studio-manual/qlora-report.md")
    parser.add_argument("--index", default="visualizer/runtime/studio-manual/qlora-index.html")
    parser.add_argument("--output-dir", default="visualizer/runtime/studio-manual/qlora-outputs")
    parser.add_argument("--trace", default="visualizer/runtime/studio-manual/qlora-trace.json")
    parser.add_argument("--trace-delay", type=float, default=0.0)
    parser.add_argument("--seq-length", type=int, default=None)
    parser.add_argument("--per-device-batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--target-module-count", type=int, default=None)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--skip-hf-metadata", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Skip remote metadata and use the compact candidate set.")
    parser.add_argument("--extra-args-json", default="")
    args = parser.parse_args()

    config = read_json(args.config)
    defaults = dict(config["defaults"])
    extra_args = known_extra_args(
        args.extra_args_json,
        {
            "seq_length": int,
            "per_device_batch_size": int,
            "gradient_accumulation_steps": int,
            "rank": int,
            "lora_rank": int,
            "target_module_count": int,
            "activation_multiplier": float,
            "checkpoint_activation_multiplier": float,
            "safety_margin_gb": float,
        },
        context="Studio QLoRA planner",
    )
    for key in [
        "seq_length",
        "per_device_batch_size",
        "gradient_accumulation_steps",
        "target_module_count",
        "activation_multiplier",
        "checkpoint_activation_multiplier",
        "safety_margin_gb",
    ]:
        if key in extra_args:
            defaults[key] = extra_args[key]
    if "rank" in extra_args:
        defaults["lora_rank"] = extra_args["rank"]
    if "lora_rank" in extra_args:
        defaults["lora_rank"] = extra_args["lora_rank"]
    seq_length = args.seq_length or (512 if args.quick else int(defaults["seq_length"]))
    per_device_batch_size = args.per_device_batch_size or int(defaults["per_device_batch_size"])
    grad_accum = args.gradient_accumulation_steps or (4 if args.quick else int(defaults["gradient_accumulation_steps"]))
    rank = args.rank or (16 if args.quick else int(defaults["lora_rank"]))
    gradient_checkpointing = not args.no_gradient_checkpointing
    output_dir = resolve_project_path(args.output_dir)
    os.environ["HF_HOME"] = str(output_dir / "hf-cache")
    os.environ["HF_DATASETS_CACHE"] = str(output_dir / "hf-cache" / "datasets")
    os.environ["HF_XET_CACHE"] = str(output_dir / "hf-cache" / "xet")
    report_path = resolve_project_path(args.report)
    index_path = resolve_project_path(args.index)
    trace_path = resolve_project_path(args.trace)
    for path in [output_dir, report_path.parent, index_path.parent, trace_path.parent]:
        path.mkdir(parents=True, exist_ok=True)

    trace = VisualTrace(LESSON_ID, LESSON_TITLE, trace_path, args.trace_delay)
    trace.event(
        "load planning config",
        "setup",
        "读取 Studio QLoRA 训练工程配置。配置只包含规划参数，不包含 CUDA 训练入口。",
        inputs={"config": args.config},
        outputs={"profiles": [profile["name"] for profile in config["profiles"]], "quick": args.quick, "extra_args": extra_args},
    )

    torch = try_import_torch()
    machine = detect_local_config(torch)
    selected_model = resolve_model_name(args.model_name, machine)
    selected_targets = infer_lora_target_modules(selected_model)
    target_module_count = args.target_module_count or len(selected_targets) or int(defaults["target_module_count"])
    trace.event(
        "detect local hardware and model policy",
        "setup",
        "用共享 hf_model_policy helper 检测本机并解析 --model-name auto。",
        inputs={"requested_model": args.model_name},
        outputs={"selected_model": selected_model, "target_modules": selected_targets},
        metrics={"memory_gb": machine.get("memory_gb", 0), "mps_available": machine.get("mps_available", False)},
    )

    metadata: dict[str, dict[str, Any]] = {}
    should_skip_metadata = args.skip_hf_metadata or args.quick
    metadata_models = [selected_model] if should_skip_metadata else sorted(
        {selected_model if item["id"] == "auto" else item["id"] for item in config["candidate_models"]}
    )
    for model_id in metadata_models:
        if should_skip_metadata:
            metadata[model_id] = {
                "model_id": model_id,
                "available": False,
                "skipped": True,
                "error": "skipped by --quick or --skip-hf-metadata",
            }
        else:
            metadata[model_id] = try_load_hf_metadata(model_id, args.local_files_only)
    metadata_path = output_dir / "hf_metadata.json"
    write_json(metadata_path, metadata)
    trace.event(
        "optional HF metadata probe",
        "metadata",
        "尝试读取 Hugging Face config/tokenizer 元数据；这一步不下载模型权重，失败也不影响本次执行。",
        inputs={"models": metadata_models, "local_files_only": args.local_files_only, "skipped": should_skip_metadata},
        outputs={"metadata_file": metadata_path, "available_count": sum(1 for item in metadata.values() if item.get("available"))},
    )

    specs = dedupe_specs_by_model_id(resolve_candidate_specs(config, selected_model, metadata))
    if args.quick:
        keep_ids = {selected_model, "Qwen/Qwen2.5-0.5B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"}
        specs = [spec for spec in specs if spec.model_id in keep_ids]

    local_memory_gb = float(machine.get("memory_gb", 0) or 0)
    plans = [
        plan_memory(
            spec=spec,
            profile=profile,
            defaults=defaults,
            seq_length=seq_length,
            per_device_batch_size=per_device_batch_size,
            grad_accum=grad_accum,
            rank=rank,
            target_module_count=args.target_module_count or len(infer_lora_target_modules(spec.model_id)) or target_module_count,
            gradient_checkpointing=gradient_checkpointing,
            local_memory_gb=local_memory_gb,
        )
        for spec in specs
        for profile in config["profiles"]
    ]
    plan_json_path = output_dir / "plans.json"
    plan_csv_path = output_dir / "memory_budget.csv"
    write_json(plan_json_path, plans)
    write_csv(plan_csv_path, plans)
    trace.event(
        "estimate memory budget",
        "planning",
        "估算 base 权重、LoRA 训练状态、activation 和 safety margin，比较 Mac/MPS、CUDA QLoRA、DeepSpeed profiles。",
        inputs={
            "seq_length": seq_length,
            "per_device_batch_size": per_device_batch_size,
            "gradient_accumulation_steps": grad_accum,
            "rank": rank,
            "gradient_checkpointing": gradient_checkpointing,
        },
        outputs={"plan_json": plan_json_path, "plan_csv": plan_csv_path, "rows": len(plans)},
        metrics={
            "min_peak_gb": min(row["estimated_peak_gb"] for row in plans),
            "max_peak_gb": max(row["estimated_peak_gb"] for row in plans),
        },
    )

    boundaries = summarize_boundaries(machine)
    trace.event(
        "explain execution boundaries",
        "planning",
        "输出 QLoRA、量化、CUDA/bitsandbytes、Mac/MPS、梯度检查点、梯度累积和 DeepSpeed 的边界说明。",
        outputs=boundaries,
    )

    paths = {
        "plan_json": plan_json_path.relative_to(PROJECT_ROOT),
        "plan_csv": plan_csv_path.relative_to(PROJECT_ROOT),
        "metadata_json": metadata_path.relative_to(PROJECT_ROOT),
        "trace_json": trace_path.relative_to(PROJECT_ROOT),
    }
    write_report(report_path, machine, selected_model, specs, plans, metadata, boundaries, paths)
    write_index(index_path)
    trace.event(
        "write studio artifacts",
        "artifact",
        "生成 Studio 报告和静态页面，所有运行产物保存在本次 run 目录。",
        outputs={"report": report_path, "index": index_path, **paths},
    )
    trace.finish(
        "Studio QLoRA planning completed without requiring bitsandbytes, CUDA, or model-weight downloads.",
        metrics={"plan_rows": len(plans), "metadata_models": len(metadata)},
    )

    print(f"selected model: {selected_model}")
    print(f"wrote: {plan_json_path.relative_to(PROJECT_ROOT)}")
    print(f"wrote: {plan_csv_path.relative_to(PROJECT_ROOT)}")
    print(f"wrote: {metadata_path.relative_to(PROJECT_ROOT)}")
    print(f"wrote: {report_path.relative_to(PROJECT_ROOT)}")
    print(f"wrote: {index_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
