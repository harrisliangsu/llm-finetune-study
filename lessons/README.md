# Lessons

课程材料按课程序号拆分。每一课目录内都包含：

- `README.md`: 本课学习说明、执行命令、输入输出、验收点
- `run.py`: 可执行脚本
- `report.md`: 本地执行后生成的结果报告
- `index.html`: 可直接打开的可视化学习页

## 课程顺序

| 课程 | 主题 | 目录 | 重点 |
|---|---|---|---|
| 01 | Datasets 管线 | [01-datasets](01-datasets) | JSONL -> Dataset -> split -> filter -> map |
| 02 | AutoTokenizer | [02-tokenizer](02-tokenizer) | text -> input_ids / attention_mask / labels |
| 03 | Batch / Collator | [03-batching](03-batching) | list[dict] -> dict[tensor] |
| 04 | Trainer 闭环 | [04-trainer](04-trainer) | model + dataset + loss + optimizer |
| 05 | LoRA Adapter | [05-lora](05-lora) | frozen base + trainable adapter |

## 运行顺序

```bash
.venv/bin/python lessons/01-datasets/run.py
.venv/bin/python lessons/02-tokenizer/run.py
.venv/bin/python lessons/03-batching/run.py
.venv/bin/python lessons/04-trainer/run.py
.venv/bin/python lessons/05-lora/run.py
```

如果你只想学习页面，直接打开每课目录里的 `index.html`。
