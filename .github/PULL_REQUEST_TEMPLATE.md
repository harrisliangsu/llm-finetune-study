## 这个 PR 做了什么

一两句话讲清楚。

## 为什么

如果背景在某个 issue / Discussion 里，链接它。如果是踩坑发现的，描述现象。

## 怎么验证

```bash
# 让审核者可以 copy-paste 验证
```

## checklist

- [ ] `python -m compileall lessons training visualizer` 通过
- [ ] 涉及行为变化的改动有跑过对应课程 `run.py`，能生成 `report.md`
- [ ] 文档（README / lessons README / docs）和代码同步更新
- [ ] 一次 PR 一个独立改动，subject 用祈使句
- [ ] 如有新增依赖，已加到 `requirements-train.txt` 或 `requirements-core.txt` 并说明用途
