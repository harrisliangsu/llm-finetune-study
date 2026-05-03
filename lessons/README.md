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

## 运行顺序

```bash
.venv/bin/python lessons/01-datasets/run.py
.venv/bin/python lessons/02-tokenizer/run.py
.venv/bin/python lessons/03-batching/run.py
.venv/bin/python lessons/04-trainer/run.py
.venv/bin/python lessons/05-lora/run.py
.venv/bin/python lessons/06-peft-lora/run.py
```

每课脚本会把 Hugging Face cache、Trainer checkpoint、adapter 等生成内容写到自己目录下的 `outputs/`。这些内容已被 `.gitignore` 忽略，不会提交到 GitHub。

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
```

如果你只想学习页面，直接打开每课目录里的 `index.html`。
