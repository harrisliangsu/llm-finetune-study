# Lesson 10: QLoRA / Training Engineering

## 本课定位

本课是本地可执行的训练工程规划器，不是 CUDA 训练脚本。它会用真实的课程模型策略 helper 选择模型，尽量读取 Hugging Face config/tokenizer 元数据，然后估算不同训练路径的显存/内存预算。

关键边界：Mac/MPS 可以做小模型 LoRA/SFT 学习，但本课不会把 bitsandbytes 4-bit QLoRA 伪装成 Mac 原生能力。标准 HF QLoRA 路径应该放到 CUDA 环境验证。

## 本机检测

- system: Darwin
- machine: arm64
- processor: arm
- memory: 32 GB
- MPS built: True
- MPS available: True
- selected model by policy: `Qwen/Qwen2.5-0.5B-Instruct`
- selected metadata available: False
- selected metadata error: `skipped by --quick or --skip-hf-metadata`

## 必须理解的概念

- QLoRA: QLoRA = frozen base model loaded in 4-bit NF4 + small trainable LoRA adapters + paged optimizers.
- Quantization: Quantization reduces storage/compute dtype for frozen weights; it does not make optimizer or activations free.
- CUDA/bitsandbytes: The common HF QLoRA path depends on CUDA-oriented bitsandbytes 4-bit kernels.
- Mac/MPS: MPS is available, but this lesson does not present bitsandbytes 4-bit QLoRA as native on Mac.
- Gradient checkpointing: Checkpointing lowers activation memory by recomputing forward segments during backward.
- Gradient accumulation: Accumulation raises effective batch size across microbatches but does not multiply peak activation memory.
- DeepSpeed: DeepSpeed is a distributed training boundary for sharding states; it is not a Mac substitute for CUDA QLoRA kernels.

## 当前策略模型预算

- Mac/MPS local profile estimated peak: 2.972 GB
- CUDA QLoRA 24GB profile estimated peak: 2.31 GB
- LoRA rank: 16
- sequence length: 512
- per-device micro batch: 1
- gradient accumulation steps: 4
- effective batch per GPU: 4
- gradient checkpointing: True

## 候选模型元数据

- `Qwen/Qwen2.5-0.5B-Instruct`: 0.494B params, hidden=896, layers=24, source=planning_config_fallback:matched_Qwen/Qwen2.5-0.5B-Instruct
- `mistralai/Mistral-7B-Instruct-v0.2`: 7.24B params, hidden=4096, layers=32, source=planning_config_fallback

## 预算表

| model | profile | estimated peak GB | fit limit GB | fits | recommendation |
|---|---:|---:|---:|---:|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | `mac_mps_lora_planning` | 2.972 | 23.04 | True | local planning only: use fp16/bf16/fp32 LoRA on MPS/CPU, or move QLoRA to CUDA. |
| `Qwen/Qwen2.5-0.5B-Instruct` | `single_cuda_qlora_24gb` | 2.31 | 24.0 | True | likely fits, then validate with a one-step CUDA smoke run. |
| `Qwen/Qwen2.5-0.5B-Instruct` | `single_cuda_lora_24gb` | 2.972 | 24.0 | True | likely fits, then validate with a one-step CUDA smoke run. |
| `Qwen/Qwen2.5-0.5B-Instruct` | `multi_cuda_deepspeed_zero3` | 2.154 | 80.0 | True | likely fits, then validate with a one-step CUDA smoke run. |
| `mistralai/Mistral-7B-Instruct-v0.2` | `mac_mps_lora_planning` | 15.804 | 23.04 | True | local planning only: use fp16/bf16/fp32 LoRA on MPS/CPU, or move QLoRA to CUDA. |
| `mistralai/Mistral-7B-Instruct-v0.2` | `single_cuda_qlora_24gb` | 6.095 | 24.0 | True | likely fits, then validate with a one-step CUDA smoke run. |
| `mistralai/Mistral-7B-Instruct-v0.2` | `single_cuda_lora_24gb` | 15.804 | 24.0 | True | likely fits, then validate with a one-step CUDA smoke run. |
| `mistralai/Mistral-7B-Instruct-v0.2` | `multi_cuda_deepspeed_zero3` | 3.922 | 80.0 | True | likely fits, then validate with a one-step CUDA smoke run. |

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

DeepSpeed ZeRO-2/ZeRO-3 解决的是多 GPU 上 optimizer/gradient/parameter state 的切分。本课的 `multi_cuda_deepspeed_zero3` 只是按配置的 `world_size` 做 per-GPU 初筛估算，不替代真实 profiler。它不能让 Mac MPS 运行 bitsandbytes CUDA kernel，也不能自动消除 QLoRA 与 ZeRO/FSDP 的兼容性限制。工程上应该先用单 GPU one-step smoke test 验证模型加载、loss、保存，再扩展到 DeepSpeed。

## 产物

- `lessons/10-qlora-engineering/outputs/plans.json`
- `lessons/10-qlora-engineering/outputs/memory_budget.csv`
- `lessons/10-qlora-engineering/outputs/hf_metadata.json`
- `lessons/10-qlora-engineering/outputs/trace.json`
