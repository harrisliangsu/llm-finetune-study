# LLM Study Studio

这个页面现在是一个本地模型训练工作室，课程学习是第二个子页面：

```text
JSONL -> Dataset -> tokenizer -> batch -> model forward -> loss -> backward -> update/save/load
```

如果你是第一次使用训练工作室，先读 [../docs/08-training-studio.md](../docs/08-training-studio.md)。本文档主要记录 `visualizer/` 页面、服务端 API 和 trace 展示约定。

## 启动页面

在仓库根目录运行：

```bash
.venv/bin/python visualizer/serve.py
```

然后打开：

```text
http://127.0.0.1:8765/visualizer/
```

## 模型训练首页

默认进入 `模型训练` 页面，可以配置独立于课程页的训练任务：

- 训练方法：SFT + LoRA、PEFT LoRA、DPO 偏好优化、QLoRA 本地训练规划。
- 模型：`auto`、`Qwen/Qwen2.5-0.5B-Instruct`、`sshleifer/tiny-gpt2` 或自定义 Hugging Face model id。
- 数据：按训练方法切换 JSONL schema；支持粘贴 JSONL、上传 JSONL 文件，或使用 `visualizer/studio/data/` 下的 Studio 示例数据；点击数据路径或 `查看数据` 会打开独立窗口查看完整内容。
- 参数：max steps、max length、max new tokens、learning rate、LoRA rank/alpha/dropout、trace delay，以及按训练方法启用的高级参数；`Extra Engine Args JSON` 用于传入 selected engine 支持的额外配置，不能覆盖页面已经管理的参数。
- Chat 对比：训练页内可以输入一条工单，并显式指定 compare model、adapter dir 和 instruction；也可以使用最新 Studio 训练产物。点击一次会固定输出 base、adapter、reloaded adapter 三个结果，不需要手动选 Target。

页面职责：

- 负责收集 method/model/data/params。
- 负责 JSONL 预览和 schema 校验。
- 负责启动 `POST /api/studio/run`。
- 负责显示最近 Studio profile、run 状态、report、adapter 路径和 Chat 对比。

不负责：

- 不直接调用课程脚本。
- 不写课程 report。
- 不更新课程学习页的 `visualizer/traces/live.json`。

SFT / PEFT 粘贴数据至少 4 行，一行一个 JSON object，至少包含：

```json
{"instruction":"...","input":"...","output":"..."}
```

DPO 粘贴数据至少 2 行，至少包含：

```json
{"instruction":"...","input":"...","chosen":"...","rejected":"..."}
```

启动训练后，服务端会调用独立的 `visualizer/studio/run.py` 入口。这个入口只调公共 `training/engines/`，共享逻辑放在 `training/common/`，不会去调用 `lessons/` 课程脚本。Studio 的数据副本、trace、report、adapter、generation、metrics 等运行产物会放在：

```text
visualizer/runtime/studio-runs/<run-id>/
```

课程目录里的 `report.md`、课程归档产物和 `visualizer/traces/live.json` 不会被 Studio 训练覆盖；课程学习页不会自动读到 Studio 的运行结果。

最近一次 Studio run 的摘要会写入：

```text
visualizer/runtime/studio-profile.json
```

训练页的 Chat 对比会优先使用这个 profile 里的 model 和 adapter 信息。

## 课程学习页面

页面顶部运行控制条支持：

- 选择并运行单个 lesson
- 一键运行 01-10
- quick run 冒烟执行
- 暂停、单步、继续、停止当前课程进程
- 查看当前命令、pid 和日志尾部

服务端 API 会启动真实课程脚本，并把 trace 写入：

```text
visualizer/traces/live.json
visualizer/traces/<lesson-id>.json
```

页面默认每 5s 自动读取一次。`live.json` 用来观察当前运行；`<lesson-id>.json` 是课程归档。`Delay` 只用于演示时放慢步骤，方便观察和单步。

## Chat Lab

课程页右侧 `Chat Lab` 默认连接 Lesson 07 SFT baseline。训练页内的 `Chat 对比` 会用同一条输入调用：

- base model
- 指定 adapter dir
- fresh base 重新加载指定 adapter

用于观察训练前后输出格式和业务路由结果的差异，同时验证保存后的 adapter 部署路径是否可重新加载。

如果某个课程显示 `need rerun trace`，说明本地有课程输出目录，但旧运行没有留下归档 trace。重新执行该课程后，课程卡片会变成可查看状态。

## 本地 API

`visualizer/serve.py` 只绑定 `127.0.0.1`，提供这些本地接口：

- `GET /api/state`
- `POST /api/run`
- `POST /api/run-all`
- `POST /api/studio/run`
- `POST /api/studio/validate-data`
- `POST /api/control`
- `POST /api/chat`

## Trace 内容

每个事件包含：

- `inputs`: 当前步骤输入
- `outputs`: 当前步骤输出
- `tensors`: 张量名称、shape、dtype 或含义
- `model`: base、adapter、更新参数、保存状态
- `metrics`: loss、learning rate、eval loss 等训练指标
- `sample`: 当前样本、prompt、labels decode 等辅助信息
