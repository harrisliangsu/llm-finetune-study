# Local Training Visualizer

这个页面用来配合 01-06 课程观察训练过程：

```text
JSONL -> Dataset -> tokenizer -> batch -> model forward -> loss -> backward -> update/save/load
```

## 启动页面

在仓库根目录运行：

```bash
.venv/bin/python visualizer/serve.py
```

然后打开：

```text
http://127.0.0.1:8765/visualizer/
```

## 运行课程并观察页面变化

另开一个终端运行任意课程：

```bash
.venv/bin/python lessons/04-trainer/run.py --trace-delay 0.5
.venv/bin/python lessons/05-lora/run.py --trace-delay 0.5
.venv/bin/python lessons/06-peft-lora/run.py --trace-delay 0.5
```

脚本会写入：

```text
visualizer/traces/live.json
```

页面每 700ms 自动读取一次。`--trace-delay` 不是训练必需参数，只是为了让演示时每一步停顿一下，方便肉眼观察。

## Trace 内容

每个事件包含：

- `inputs`: 当前步骤输入
- `outputs`: 当前步骤输出
- `tensors`: 张量名称、shape、dtype 或含义
- `model`: base、adapter、更新参数、保存状态
- `metrics`: loss、learning rate、eval loss 等训练指标
- `sample`: 当前样本、prompt、labels decode 等辅助信息
