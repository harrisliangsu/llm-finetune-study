# Lesson 05: LoRA Adapter 微调

## 是否需要下载模型

本课不下载模型。

为了先学清楚 LoRA 机制，我们继续使用 Lesson 04 的 tiny causal LM 作为 base model，只在 `lm_head` 上插入 LoRA A/B 低秩矩阵。

真实 LLM LoRA 后续需要下载 base model，但一开始不需要。先把 adapter 的训练、保存、加载吃透更重要。

## 本课目标

理解参数高效微调的核心结构：

```text
frozen base model + trainable low-rank adapter = LoRA fine-tuning
```

本课执行链路：

```text
tiny causal LM -> freeze base -> attach LoRA -> train adapter -> save adapter -> load adapter
```

## 运行

```bash
.venv/bin/python lessons/05-lora/run.py
```

## 输入

- `examples/sample_sft.jsonl`
- Lesson 02 生成的本地 tokenizer
- Lesson 04 同结构 tiny causal LM

## 输出

- [report.md](report.md)
- [index.html](index.html)
- 本课 datasets 缓存：`lessons/05-lora/outputs/hf-cache`
- adapter-only checkpoint: `lessons/05-lora/outputs/lora/lora_adapter.pt`

## 本课关键结果

- target module: `lm_head`
- LoRA rank `r`: 4
- LoRA alpha: 8
- trainable params: 1,408
- total params: 89,280
- trainable ratio: 1.5771%
- eval loss before LoRA training: 5.5519
- train loss: 5.0742
- eval loss after LoRA training: 5.7332

## 你必须理解的点

1. LoRA 训练的是低秩增量，不是完整 base model。
2. base 参数被冻结，只有 `lora_A/lora_B` 更新。
3. `r` 越大，可训练参数越多，表达能力越强，也更容易过拟合。
4. `alpha/r` 是 LoRA 分支的缩放系数。
5. adapter checkpoint 只保存 adapter，不保存完整模型。

## 验收

你应该能解释：

- 为什么 trainable params 只有 1.5771%。
- 为什么 adapter 可以单独保存。
- 为什么重新加载 adapter 后输出应和训练后 LoRA 输出一致。
- 真实 LLM 中为什么 target modules 常见是 `q_proj/v_proj/o_proj`，而本课用 `lm_head`。
