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

## 专有词解释 + 简短例子

### PEFT

作用：PEFT 是 Parameter-Efficient Fine-Tuning，参数高效微调工具链。它提供统一接口来做 LoRA、Prefix Tuning 等方法；LoRA 是 PEFT 支持的一种具体方法，不等于 PEFT 本身。

例子：本课用 PEFT 的 `LoraConfig` 和 `get_peft_model` 把 Qwen base model 改造成只训练 LoRA adapter 的模型。

### PEFT 支持的常见方法

PEFT 不是一种算法，而是一组“少训练参数”的方法集合。当前最适合作为本课程主线的是 LoRA，因为它最常见、工程资料最多，也最容易观察 adapter 的保存和加载。

| 方法 | 核心想法 | 简短例子 | 学习优先级 |
|---|---|---|---|
| LoRA | 在目标线性层旁边加低秩 A/B 矩阵，只训练增量 | 本课把 LoRA 加到 `q_proj`、`v_proj` | 最高 |
| AdaLoRA | 训练时动态分配不同层的 rank，把参数预算给更重要的层 | 不是所有层都固定 `r=8`，重要层 rank 更高 | 中 |
| IA3 | 不加低秩矩阵，而是学习少量缩放向量来调节激活 | 给 attention/FFN 的中间表示乘一个可训练 gate | 中 |
| Prompt Tuning | 冻结模型，只训练一段连续向量形式的 soft prompt | 在真实 prompt 前面拼一串可训练“虚拟 token” | 中 |
| Prefix Tuning | 给 Transformer 每层 attention 加可训练 prefix key/value | 每层都多一段可学习上下文，影响注意力 | 中 |
| P-Tuning | 用可训练 prompt 表示或 prompt encoder 引导模型 | 比手写 prompt 更可训练，但仍不改大部分模型权重 | 中 |
| LoHa / LoKr | LoRA 的低秩分解变体，用 Hadamard/Kronecker 结构提高表达 | 常见于 LyCORIS 系列 adapter | 低 |
| OFT / BOFT | 用正交变换方式调整权重，关注保持原模型表示结构 | 适合了解，不建议作为第一条本地实践线 | 低 |
| X-LoRA | 用门控/混合方式组合多个 LoRA adapter | 不同任务 adapter 由路由权重决定贡献 | 低 |
| LayerNorm Tuning | 只训练 LayerNorm 等极少数参数 | 参数量极小，但能力也受限 | 低 |

本课程当前只执行 LoRA。你先掌握 `base model + adapter`、`target_modules`、`save_pretrained`、`PeftModel.from_pretrained`，再扩展到 AdaLoRA、IA3、Prompt/Prefix Tuning。

### Hugging Face model id

作用：model id 是 Hugging Face Hub 上定位模型的名字，通常是 `组织或用户/模型名`。代码用它来下载 tokenizer、config 和权重。

例子：`Qwen/Qwen2.5-0.5B-Instruct` 就是本课的 model id，`from_pretrained` 会根据它找到对应模型文件。

### AutoModelForCausalLM

作用：`AutoModelForCausalLM` 会根据模型配置自动选择合适的 causal language model 类。它适合加载“输入文本，预测下一个 token”的生成式语言模型。

例子：本课用 `AutoModelForCausalLM.from_pretrained(model_id)` 加载完整 Qwen base model，然后再交给 PEFT 注入 LoRA。

### AutoTokenizer

作用：`AutoTokenizer` 会根据 model id 加载对应 tokenizer，把中文、英文和符号切成模型能理解的 token id。

例子：输入“介绍一下 LoRA”，tokenizer 会把它编码成一串整数 id，训练脚本再把这些 id 喂给 Qwen。

### target_modules

作用：`target_modules` 告诉 PEFT 要把 LoRA adapter 挂到哪些模块上。选得越多，可训练参数越多；选得太少，模型可调整空间可能不足。

例子：本课设置 `target_modules=["q_proj", "v_proj"]`，所以 LoRA 只加在 attention 的 query/value 投影层上。

### q_proj / v_proj

作用：`q_proj` 和 `v_proj` 是 Transformer attention 里的线性投影层。修改它们能影响模型“看哪里”和“取出什么信息”，所以是真实 LLM LoRA 的常见目标。

例子：用户问一个问题时，`q_proj` 影响 query 表示，`v_proj` 影响 value 表示；LoRA 在这两处加 delta，相当于轻量调整注意力行为。

### save_pretrained

作用：`save_pretrained` 是 Hugging Face/PEFT 常用保存接口。对 PEFT model 调用它时，通常保存 adapter 配置和 adapter 权重，而不是完整 base model。

例子：本课 `model.save_pretrained(adapter_dir)` 会在 `lessons/06-peft-lora/outputs/adapter` 下保存 `adapter_config.json` 和 adapter 权重文件。

### PeftModel.from_pretrained

作用：`PeftModel.from_pretrained` 用来把已经保存的 PEFT adapter 加载回一个 base model。推理部署时常用“先加载 base，再加载 adapter”的方式。

例子：先用 `AutoModelForCausalLM.from_pretrained(model_id)` 得到 fresh Qwen，再用 `PeftModel.from_pretrained(base_model, adapter_dir)` 挂回本课训练出的 LoRA。

### trainable ratio

作用：trainable ratio 是可训练参数占总参数的比例，用来衡量参数高效微调到底省了多少训练量。

例子：本课 trainable params 是 540,672，总参数约 494.6M，trainable ratio 只有 0.1093%，说明训练时只更新很小一部分参数。

## 验收

你应该能解释：

- 为什么 trainable ratio 只有 0.1093%。
- 为什么 Lesson 06 比 Lesson 05 更接近真实项目。
- 为什么重新加载 adapter 后输出应和训练后输出一致。
- 登录 Hugging Face 对下载速度和限流有什么影响。
