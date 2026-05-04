# Lesson 07: SFT Baseline

本课用于补齐独立 SFT 基线。目标不是先学 LoRA，而是先看清：

- 同一批固定 eval prompts 在训练前后如何变化
- prompt 区域为什么不参与 loss
- assistant answer 如何学成严格结构
- 数据量和场景选择为什么会影响可视化效果

## 自包含数据

本课训练数据放在：

```text
lessons/07-sft-baseline/data/train.jsonl
```

场景是客服工单路由到严格 JSON。输出字段固定为：

```json
{
  "intent": "refund_request",
  "priority": "high",
  "department": "billing",
  "summary": "会员重复扣费，用户要求退款"
}
```

这个场景比通用概念解释更适合观察 SFT 效果，因为训练前后可以直接检查：

- 是否输出 JSON
- 字段是否齐全
- intent / department 是否选对
- 是否输出了 JSON 以外的废话

后续实现本课脚本时，`run.py`、`report.md`、`index.html` 和 `outputs/` 都应留在本目录内，保持课程自包含。
