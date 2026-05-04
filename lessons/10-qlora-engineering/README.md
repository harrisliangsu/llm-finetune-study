# Lesson 10: QLoRA / Training Engineering

本课是一个本地可执行的训练工程规划器，不是 CUDA 训练脚本。

它解决的问题是：在真正启动 QLoRA 之前，先搞清楚模型大小、量化方式、activation、gradient checkpointing、gradient accumulation、DeepSpeed 边界，以及当前机器到底能不能跑。

## 运行

快速冒烟测试，不联网读取 HF metadata：

```bash
.venv/bin/python lessons/10-qlora-engineering/run.py --quick
```

带可视化事件停顿：

```bash
.venv/bin/python lessons/10-qlora-engineering/run.py --quick --trace-delay 0.2
```

尝试读取 Hugging Face config/tokenizer 元数据，不下载模型权重：

```bash
.venv/bin/python lessons/10-qlora-engineering/run.py
```

只读本地 HF cache：

```bash
.venv/bin/python lessons/10-qlora-engineering/run.py --local-files-only
```

## 输入

- `data/planning_config.json`: 候选模型、训练 profile、默认 sequence/batch/rank 参数
- 课程公共 helper:
  - `lessons.common.hf_model_policy.detect_local_config`
  - `lessons.common.hf_model_policy.resolve_model_name`
  - `lessons.common.hf_model_policy.infer_lora_target_modules`

## 输出

所有运行产物写入本课目录：

- `outputs/plans.json`
- `outputs/memory_budget.csv`
- `outputs/hf_metadata.json`
- `outputs/trace.json`
- `../../visualizer/traces/10-qlora-engineering.json`: 给可视化页面课程下拉框读取的归档 trace

学习材料：

- [report.md](report.md)
- [index.html](index.html)

## 为什么本课不直接跑 QLoRA

标准 Hugging Face QLoRA 通常是：

```text
transformers + peft + bitsandbytes 4-bit NF4 + CUDA
```

Apple Silicon/MPS 可以用于小模型 LoRA/SFT 学习，但 bitsandbytes 4-bit QLoRA 不是本课要假装支持的 Mac 原生路径。Mac 上更合理的工作流是：

1. 本地用本课规划模型、上下文长度、batch、rank 和显存预算。
2. 小模型继续用 MPS/CPU 跑 LoRA/SFT 学习闭环。
3. 真正 QLoRA 放到 CUDA 机器做 one-step smoke test。
4. 大到单卡放不下时，再评估 DeepSpeed/FSDP。

## 你必须理解的点

- QLoRA 省的是 frozen base weight 的显存，不是所有训练内存。
- 量化权重之外，LoRA adapter、梯度、optimizer state、activation 仍然占内存。
- Gradient checkpointing 用重算换显存，主要减少 activation 保存。
- Gradient accumulation 增大有效 batch，但峰值仍由 microbatch 决定。
- DeepSpeed 是多 GPU sharding 边界，不是 Mac MPS 跑 CUDA bitsandbytes 的替代品；本课只按配置的 `world_size` 做 per-GPU 初筛估算，不能替代真实 profiler。

## 常用参数

```bash
.venv/bin/python lessons/10-qlora-engineering/run.py \
  --model-name auto \
  --seq-length 2048 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 16 \
  --rank 64
```

OOM 时优先调整：

1. 降 `--per-device-batch-size`
2. 降 `--seq-length`
3. 开 gradient checkpointing，本课默认已开启
4. 降 `--rank` 或减少 target modules
5. 从 fp16 LoRA 换到 CUDA QLoRA
6. 再考虑 DeepSpeed/FSDP
