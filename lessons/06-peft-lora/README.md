# Lesson 06: PEFT + 真实 Hugging Face 小模型 LoRA

## 本课是否需要下载模型

需要。脚本会下载并缓存：

```text
Qwen/Qwen2.5-0.5B-Instruct
```

下载位置在仓库内：

```text
lessons/06-peft-lora/outputs/hf-cache
```

首次下载后，后续运行会直接读本地缓存。Hugging Face 登录可以减少匿名请求限流，但本课已经跑完后不再强制需要登录。

## 为什么选这个模型

本机配置：

- Apple M2 Max
- 32GB 内存
- PyTorch MPS 可用

`Qwen/Qwen2.5-0.5B-Instruct` 是 0.5B 级别真实 instruct causal LM，比 `tiny-gpt2` 更接近中文 LLM 微调工程，同时足够小，适合本地短步数 LoRA 学习。

## 本课目标

把 Lesson 05 的手写 LoRA 换成真实工程接口：

```text
AutoModelForCausalLM
+ LoraConfig
+ get_peft_model
+ Trainer
+ save_pretrained(adapter)
+ PeftModel.from_pretrained(base, adapter)
```

## 运行

```bash
.venv/bin/python lessons/06-peft-lora/run.py
```

## 输入

- `examples/sample_sft.jsonl`
- Hugging Face model: `Qwen/Qwen2.5-0.5B-Instruct`
- LoRA target modules: `q_proj`, `v_proj`

## 输出

- [report.md](report.md)
- [index.html](index.html)
- 本课 model/datasets 缓存：`lessons/06-peft-lora/outputs/hf-cache`
- adapter-only checkpoint: `lessons/06-peft-lora/outputs/adapter`

## 本课关键结果

- tokenizer vocab size: 151,665
- LoRA rank `r`: 8
- LoRA alpha: 16
- trainable params: 540,672
- total params: 494,573,440
- trainable ratio: 0.1093%
- eval loss before training: 4.9219
- train loss: 3.5209
- eval loss after training: 4.7615

## 你必须理解的点

1. `AutoModelForCausalLM.from_pretrained` 加载的是完整 base model。
2. `get_peft_model` 不复制完整模型，而是在目标线性层挂 adapter。
3. 真实 LLM LoRA 常插在 attention projection 层，例如 `q_proj`、`v_proj`。
4. `save_pretrained(adapter_dir)` 保存的是 adapter，不是完整 base model。
5. `PeftModel.from_pretrained(base_model, adapter_dir)` 是推理时的典型加载方式。

## 验收

你应该能解释：

- 为什么 trainable ratio 只有 0.1093%。
- 为什么 Lesson 06 比 Lesson 05 更接近真实项目。
- 为什么重新加载 adapter 后输出应和训练后输出一致。
- 登录 Hugging Face 对下载速度和限流有什么影响。
