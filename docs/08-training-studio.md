# 08. Training Studio

Training Studio 是本仓库的本地模型训练工作室。它不是课程 trace 的皮肤，而是一个独立的训练入口：你可以在网页里选择训练方法、模型、JSONL 数据和参数，启动一次自包含的本地 run，然后检查 trace、report、metrics、generation 和 adapter。

启动方式：

```bash
.venv/bin/python visualizer/serve.py
```

打开：

```text
http://127.0.0.1:8765/visualizer/
```

默认进入 `模型训练` 页面；`课程学习` 是第二个页面。

## 设计边界

Training Studio 和课程学习页刻意隔离：

- Studio 入口是 `visualizer/studio/run.py`。
- Studio 方法配置在 `visualizer/studio/config.py`。
- Studio 训练引擎在 `training/engines/`。
- Studio 公共逻辑在 `training/common/`。
- Studio 不调用 `lessons/` 课程脚本。
- Studio 不覆盖课程的 `report.md`、`outputs/` 或 `visualizer/traces/<lesson-id>.json`。

课程学习页负责教学顺序和 trace 回放；Training Studio 负责独立实验和产物管理。

## 支持的方法

| 方法 | 引擎 | 数据 | 产物类型 |
|---|---|---|---|
| SFT + LoRA | `training/engines/sft_lora.py` | SFT JSONL | adapter、metrics、generation、report |
| PEFT LoRA | `training/engines/peft_lora.py` | SFT JSONL | adapter、report |
| DPO Preference | `training/engines/dpo_preference.py` | DPO JSONL | adapter、preference metrics、generation、report |
| QLoRA Plan | `training/engines/qlora_plan.py` | 不需要训练数据 | 本地/CUDA/显存/量化工程规划 |

`QLoRA Plan` 是规划器，不会在 Mac/MPS 上假装启动 bitsandbytes 4-bit QLoRA。它的作用是把显存、量化、LoRA rank、batch、gradient checkpointing 和 CUDA/bitsandbytes 边界写成可检查的计划。

## 数据格式

SFT / PEFT LoRA 每行一个 JSON object，至少 4 行：

```json
{"instruction":"把用户工单路由成严格 JSON。","input":"客户说付款成功但订单显示未支付。","output":"{\"intent\":\"payment_issue\",\"priority\":\"high\",\"department\":\"billing\",\"summary\":\"客户付款成功但订单未支付\"}"}
```

字段要求：

- `instruction`: 任务说明，不能为空。
- `input`: 用户输入，可以是空字符串。
- `output`: 标准答案，不能为空。

DPO 每行一个 JSON object，至少 2 行：

```json
{"instruction":"把用户工单路由成严格 JSON。","input":"应用一直闪退。","chosen":"{\"intent\":\"app_crash\",\"priority\":\"high\",\"department\":\"technical_support\",\"summary\":\"应用持续闪退\"}","rejected":"应用有问题，请联系客服。"}
```

字段要求：

- `instruction`: 任务说明，不能为空。
- `input`: 用户输入，可以是空字符串。
- `chosen`: 偏好答案，不能为空。
- `rejected`: 反例答案，不能为空。

页面支持三类数据来源：

- 粘贴 JSONL。
- 上传 JSONL 文件。
- 使用 `visualizer/studio/data/` 下的 Studio 示例数据。

服务端只允许 Studio 数据来自 `visualizer/studio/data/` 或 `visualizer/runtime/custom-data/`，避免误读课程目录或仓库外文件。

## 运行产物

每次点击训练都会创建一个独立目录：

```text
visualizer/runtime/studio-runs/<run-id>/
```

常见内容：

```text
data/            # 本次 run 使用的数据副本
trace.json       # Studio trace
report.md        # 训练报告
index.html       # 报告页面
metrics.json     # SFT 类指标
generations/     # 训练前/训练后/重新加载 adapter 后的生成结果
adapter/         # adapter-only checkpoint
trainer/         # 训练缓存、HF cache 等
```

Studio 还会更新：

```text
visualizer/runtime/studio-profile.json
```

这个 profile 记录最近一次 Studio run 的 model、adapter、trace、report 等路径，供训练页的 `Chat 对比` 使用。

## 推荐 workflow

1. 在 `模型训练` 页面选择 `SFT + LoRA`。
2. 选择 `auto` 或明确模型，例如 `Qwen/Qwen2.5-0.5B-Instruct`。
3. 先用 Studio 示例数据跑 quick run，确认依赖、设备和目录都正常。
4. 粘贴自己的 JSONL，点击 `校验数据`。
5. 根据显存和速度调整 `max steps`、`max length`、LoRA rank、batch 和 gradient accumulation。
6. 启动训练，观察日志、trace、metrics 和 report。
7. 用 `Chat 对比` 检查 base、adapter、reloaded adapter 的输出差异。
8. 需要理解某个环节时，切到 `课程学习` 页面，对照 01-10 的课程 trace。

## Chat 对比

训练页的 `Chat 对比` 会用同一条输入比较：

- base model
- base model + 指定 adapter
- fresh base 重新加载指定 adapter

这不是为了做最终评测，而是为了检查两件事：

- adapter 是否真的改变了输出。
- 保存后的 adapter 路径是否能被重新加载。

如果还没有可用 adapter，可以选择 base only，或者先完成一次 SFT/PEFT/DPO Studio run。

## 和课程学习页的关系

课程学习页读取：

```text
visualizer/traces/live.json
visualizer/traces/<lesson-id>.json
```

Training Studio 读取和写入：

```text
visualizer/runtime/studio-runs/<run-id>/trace.json
visualizer/runtime/studio-profile.json
```

两边的 trace 不混用。这样做的原因是：

- 课程 trace 要稳定对应 01-10 的教学内容。
- Studio run 要允许自由试数据、模型和参数。
- 独立目录更容易删除、复现和对比。

## Extra Engine Args JSON

`Extra Engine Args JSON` 用来传递页面没有显式暴露的引擎参数。例如：

```json
{
  "logging-steps": 1,
  "save-steps": 20
}
```

限制：

- 必须是 JSON object。
- key 会被转换成 CLI flag，例如 `logging_steps` -> `--logging-steps`。
- 不能覆盖 Studio 已经管理的参数，例如 `model-name`、`data`、`adapter-dir`、`max-steps`、`rank`。

## 常见问题

### 数据校验失败

检查 JSONL 是否是一行一个 object。不要粘贴 JSON array；不要把多行 pretty JSON 当作 JSONL。

### adapter 不可用

确认目标目录存在：

```text
visualizer/runtime/studio-runs/<run-id>/adapter/adapter_config.json
```

如果这个文件不存在，Chat 对比不会把它当作可加载 adapter。

### Studio 结果没有出现在课程学习页

这是预期行为。Studio 结果在 `visualizer/runtime/studio-runs/`，课程页只读课程 trace。需要看 Studio 结果时，在训练页查看最近 run、report、adapter 和 Chat 对比。

### Mac/MPS 上 QLoRA 没有真的训练

这是预期行为。`QLoRA Plan` 是工程规划器。真实 bitsandbytes 4-bit QLoRA 主要面向 CUDA 环境；Mac/MPS 学习时先用 SFT/PEFT LoRA 跑通流程，再把 QLoRA 当作大模型训练工程规划来理解。

## 相关文件

- [visualizer/README.md](../visualizer/README.md)
- [visualizer/studio/run.py](../visualizer/studio/run.py)
- [visualizer/studio/config.py](../visualizer/studio/config.py)
- [training/engines/sft_lora.py](../training/engines/sft_lora.py)
- [training/engines/peft_lora.py](../training/engines/peft_lora.py)
- [training/engines/dpo_preference.py](../training/engines/dpo_preference.py)
- [training/engines/qlora_plan.py](../training/engines/qlora_plan.py)
- [training/engines/chat_compare.py](../training/engines/chat_compare.py)
