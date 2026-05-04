# Lessons

课程材料按课程序号拆分。每一课目录内都包含：

- `README.md`: 本课学习说明、执行命令、输入输出、验收点
- `run.py`: 可执行脚本
- `report.md`: 本地执行后生成的结果报告
- `index.html`: 可直接打开的可视化学习页
- `outputs/`: 本课运行时生成的缓存、checkpoint、adapter 等本地输出

每课的 `README.md` 和 `index.html` 都会补充本课相关术语解释。学习时先看术语和例子，再看执行步骤，最后回到 `run.py` 对照代码。

## 课程顺序

| 课程 | 主题 | 目录 | 重点 |
|---|---|---|---|
| 01 | Datasets 管线 | [01-datasets](01-datasets) | JSONL -> Dataset -> split -> filter -> map |
| 02 | AutoTokenizer | [02-tokenizer](02-tokenizer) | text -> input_ids / attention_mask / labels |
| 03 | Batch / Collator | [03-batching](03-batching) | list[dict] -> dict[tensor] |
| 04 | Trainer 闭环 | [04-trainer](04-trainer) | model + dataset + loss + optimizer |
| 05 | LoRA Adapter | [05-lora](05-lora) | frozen base + trainable adapter |
| 06 | PEFT LoRA | [06-peft-lora](06-peft-lora) | real HF model + PEFT adapter |
| 07 | SFT Baseline | [07-sft-baseline](07-sft-baseline) | ticket text -> strict JSON |

## 运行顺序

```bash
.venv/bin/python lessons/01-datasets/run.py
.venv/bin/python lessons/02-tokenizer/run.py
.venv/bin/python lessons/03-batching/run.py
.venv/bin/python lessons/04-trainer/run.py
.venv/bin/python lessons/05-lora/run.py
.venv/bin/python lessons/06-peft-lora/run.py
.venv/bin/python lessons/07-sft-baseline/run.py
```

也可以一次执行：

```bash
.venv/bin/python lessons/run_all.py
```

快速冒烟测试：

```bash
.venv/bin/python lessons/run_all.py --quick
```

每课脚本会把 Hugging Face cache、Trainer checkpoint、adapter 等生成内容写到自己目录下的 `outputs/`。这些内容已被 `.gitignore` 忽略，不会提交到 GitHub。

## 模型选择原则

课程里需要模型时，优先从 Hugging Face 下载真实模型，而不是手写或随机生成模型：

- Lesson 04 使用 `sshleifer/tiny-gpt2`，用于快速学习 `Trainer` 闭环。
- Lesson 06 默认 `--model-name auto`，在当前 32GB + MPS 机器上选择 `Qwen/Qwen2.5-0.5B-Instruct`，用于真实 PEFT LoRA。
- Lesson 07 默认 `--model-name auto`，在当前 32GB + MPS 机器上选择 `Qwen/Qwen2.5-0.5B-Instruct`，用于客服工单严格 JSON SFT baseline。

Lesson 05 仍然保留手写 LoRA，因为它的教学目标是看清低秩矩阵 `A/B` 如何改变模型层输出，而不是训练真实大模型。

## 实时可视化

启动本地页面：

```bash
.venv/bin/python visualizer/serve.py
```

打开：

```text
http://127.0.0.1:8765/visualizer/
```

运行任意课程时，脚本会写入 `visualizer/traces/live.json`，页面会自动刷新。演示时建议加 `--trace-delay 0.5`，让每一步停顿一下：

```bash
.venv/bin/python lessons/04-trainer/run.py --trace-delay 0.5
.venv/bin/python lessons/05-lora/run.py --trace-delay 0.5
.venv/bin/python lessons/06-peft-lora/run.py --trace-delay 0.5
.venv/bin/python lessons/07-sft-baseline/run.py --trace-delay 0.5
```

如果你只想学习页面，直接打开每课目录里的 `index.html`。

## 训练方法总览

课程 01-07 已覆盖本地数据管线、tokenizer、batch、Trainer、LoRA、PEFT 和独立 SFT baseline。后续课程建议为：

| 课程 | 主题 | 重点 |
|---|---|---|
| 08 | DPO Preference Optimization | 用 `prompt/chosen/rejected` 做小样本偏好优化 |
| 09 | Reward / RLHF Concept | 理解 reward model、reference model、KL、PPO，不做大规模本地训练 |
| 10 | QLoRA / Training Engineering | 理解量化、显存、CUDA/DeepSpeed 边界，不作为本地 Mac 主线 |

PEFT adapter 的保存、加载、重新加载对比已经是 Lesson 06/07 的一部分，不再单独拆成 Adapter Evaluation 章节。以后如果要做多 adapter 路由、多个 checkpoint 管理，那应该作为 PEFT 进阶补充，而不是 DPO 前的必修课。

Lesson 06 的效果对比不明显是设计问题：5 条样本只够验证链路，通用概念解释也不是经典 SFT 展示场景。Lesson 07 使用 [07-sft-baseline/data/train.jsonl](07-sft-baseline/data/train.jsonl) 的结构化输出任务，因为训练前后可以直接看 JSON 格式、字段选择、intent 分类是否变稳定。

想从开发者视角理解什么时候选 SFT、PEFT/LoRA、QLoRA、DPO、RLHF、蒸馏，以及每种方法需要什么数据、输出什么 artifact，见 [docs/07-training-methods-guide.md](../docs/07-training-methods-guide.md)。
