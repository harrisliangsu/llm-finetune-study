# Contributing

谢谢你愿意花时间。这是一个面向本地学习者的微调课程仓库，一个原则：**任何改动都要在 macOS（Apple Silicon）和普通 Linux 开发机上能跑通**，并且不依赖大显存或外网模型。

英文导览见 [README.en.md](README.en.md)。行为上的说明以本文件为准。

## 用哪种方式开口

| 情况 | 用这个 |
|---|---|
| 课程脚本或 Studio 跑失败，你能给出命令和 traceback | [Issue · 环境复现](https://github.com/harrisliangsu/llm-finetune-study/issues/new/choose) |
| 文档、课程说明、README 写错了，或链接断了 | Issue · 文档纠错 |
| 新课、练习或 Studio 能力，你已经能说清放在哪 | Issue · 课程或想法；还没成稿就先去 Discussions 的 **Ideas** |
| 怎么学、卡在概念上、还没落到具体文件 | [Discussions · Q&A](https://github.com/harrisliangsu/llm-finetune-study/discussions/categories/q-a) |
| 自我介绍、学习进度、闲聊 | [Discussions · General](https://github.com/harrisliangsu/llm-finetune-study/discussions/categories/general)。欢迎帖：[欢迎：自我介绍 / 学习进度 / 纠错建议](https://github.com/harrisliangsu/llm-finetune-study/discussions/1) |
| 展示你自己的小实验或笔记 | [Discussions · Show and tell](https://github.com/harrisliangsu/llm-finetune-study/discussions/categories/show-and-tell) |

需要改某一段文字或代码时，请开 Issue 或 PR，这样能对到文件。开放式讨论留在 Discussions。**Announcements** 留给维护者发更新；投票用 **Polls**。

## 本地校验

```bash
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m compileall lessons training visualizer   # 语法 / 可加载
.venv/bin/python lessons/run_all.py --help                   # 课程入口可启动
.venv/bin/python visualizer/serve.py --help                  # studio 入口可启动
```

涉及具体某一课时，至少跑通对应目录下的 `run.py`，并确认 `report.md` 和 `outputs/` 能正常生成。只改 README、模板或 Pages 门户时，说明不涉及课程行为即可。

## 加新课 / 改课程

1. 一节课要同时维护三件套：`lessons/<n>-<topic>/README.md`、`run.py`、`index.html`。
   缺任何一件都不算完成。
2. 课程默认模型用 `auto`（小模型 / tiny-gpt2）能跑通，再可选切到 `Qwen/Qwen2.5-0.5B-Instruct` 之类的真实模型。
3. 如果新课引入了新依赖，加进 `requirements.txt` 时写明用途，并确认在 Apple Silicon 上能装。
4. 新增或改动 `lessons/` 下的目录，请同步更新 `lessons/README.md` 的课程顺序表和根 `README.md`、`README.en.md` 的在线阅读表。

## 改 Studio / Visualizer

1. 改 `visualizer/serve.py` 的 API 时同步更新 `visualizer/README.md` 和 `docs/08-training-studio.md`。
2. 改前端布局或交互时，确认课程子页 (`lessons/<n>/index.html`) 还能正常打开和滚动。
3. GitHub Pages 只托管课程 HTML。不要把 Studio 写成「打开 Pages 就能训练」。Studio 仍然是 `./bootstrap.sh studio`。

## PR checklist

- [ ] `python -m compileall lessons training visualizer` 通过（纯文档改动请在 PR 里注明）
- [ ] 涉及的课程 `run.py` 能跑出 `report.md`
- [ ] 文档（README / lessons README / docs）和代码同步更新
- [ ] 一次提交一个独立改动，subject 用祈使句

## 报 bug

用环境复现模板，附上：

- 操作系统 + 芯片（如 `macOS 15.4 / Apple M3`）
- `python --version`
- `pip list | grep -E 'torch|transformers|peft|trl|datasets'`
- 你跑的完整命令和完整输出（包含 traceback）
