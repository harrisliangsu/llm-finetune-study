# 07. 开发者训练方法指南

这篇文档回答一个工程问题：当你想让模型在某个场景里表现更好时，应该选择哪种训练方法、准备什么数据、看什么指标、产出什么 artifact。

先记住一个判断：

```text
训练方法 = 数据形态 + 优化目标 + 更新哪些参数 + 评估方式
```

很多名词不在同一维度：

- SFT 是训练目标和数据形态：用标准答案教模型怎么回答。
- PEFT 是参数更新方式：只更新少量参数，比如 LoRA adapter。
- RLHF / DPO 是偏好优化：用 chosen/rejected 或 reward 让模型更符合偏好。
- QLoRA 是资源优化组合：量化加载 base model，再训练 LoRA adapter。

## 0. 先判断是否需要训练

不要把所有模型问题都归因于“需要微调”。开发者应该先按成本从低到高排查：

| 问题 | 优先方案 | 何时才训练 |
|---|---|---|
| 回答格式不稳定 | prompt、system message、schema 约束 | 大量稳定格式都失败 |
| 缺少业务知识 | RAG、工具调用、上下文注入 | 知识是风格或长期语料分布，不适合每次检索 |
| 语气/流程不符合产品 | few-shot、模板、SFT | 有稳定示例集，希望模型内化表达方式 |
| 某类回答总是不被用户喜欢 | preference 数据、DPO/RLHF | 已有 chosen/rejected 或打分数据 |
| 本地资源不够 | PEFT/LoRA/QLoRA | 完整微调成本过高 |

训练的收益来自可重复的数据分布。如果你没有稳定数据、没有验证集、没有对比指标，先不要扩大训练规模。

## 1. 训练方法全景

| 方法 | 解决什么 | 数据格式 | 更新参数 | 本地可行性 | 主要产物 |
|---|---|---|---|---|---|
| Feature Extraction | 用模型做特征，外接小模型 | 文本 + label | 分类头或外部模型 | 高 | 分类器或小头 |
| Full Fine-tuning | 大幅改变模型能力或领域行为 | 任务数据或领域数据 | 全部参数 | 低，除非小模型 | 完整模型 checkpoint |
| Continued Pretraining | 注入领域语言和知识分布 | 大量无标注文本 | 全部或部分参数 | 中低 | 领域 base model |
| SFT | 教模型按指令生成标准答案 | prompt + completion | 全参或 PEFT 参数 | 高 | SFT model 或 adapter |
| PEFT / LoRA | 降低微调成本 | 任意微调数据 | 少量 adapter 参数 | 高 | adapter checkpoint |
| QLoRA | 显存不足时训练 LoRA | SFT 或偏好数据 | LoRA adapter | 中，CUDA 更成熟 | quantized base + adapter |
| Reward Modeling | 学一个“回答好不好”的打分模型 | prompt + chosen/rejected 或评分 | reward model 参数 | 中 | reward model |
| RLHF / PPO | 用 reward 在线优化回答偏好 | prompt + reward model | policy + value 等 | 低，本地不建议先做 | aligned policy |
| DPO | 不训练 reward model，直接用偏好对齐 | prompt + chosen + rejected | policy 参数或 adapter | 中 | DPO model 或 adapter |
| KTO / ORPO / CPO 等 | DPO 的偏好优化变体 | 偏好或正负样本 | policy 参数或 adapter | 中 | preference-tuned model |
| GRPO | 强化学习优化，常用于推理/数学类奖励 | prompt + reward function | policy 参数 | 中低 | RL-tuned policy |
| Distillation | 用强模型教小模型 | teacher 输出或 logits | student 参数 | 中 | 小模型 |

本课程当前实践到：

```text
SFT 数据 + PEFT/LoRA 参数更新 = 本地低成本指令微调
```

Lesson 07 已经把 SFT 独立成可执行 baseline；后续再进入 DPO/RLHF。

## 2. SFT: Supervised Fine-Tuning

SFT 的目标是让模型在给定输入下模仿标准答案。它最适合教模型：

- 固定回答格式
- 产品语气
- 工具调用前后的表达
- 某类任务的标准操作流程
- 小范围领域问答风格

典型数据：

```json
{
  "instruction": "解释什么是梯度累积",
  "input": "",
  "output": "梯度累积是在显存不足时，把多个 mini-batch 的梯度累加后再执行一次参数更新的方法。"
}
```

聊天格式：

```json
{
  "messages": [
    {"role": "system", "content": "你是一个模型训练导师。"},
    {"role": "user", "content": "解释什么是 LoRA"},
    {"role": "assistant", "content": "LoRA 是一种参数高效微调方法..."}
  ]
}
```

训练时最关键的是 loss mask：

```text
prompt token: labels = -100，不计算 loss
assistant answer token: labels = token_id，计算 loss
```

开发者检查点：

- 是否用目标模型自己的 chat template。
- 是否只对 assistant response 计算 loss。
- 是否保留 validation split。
- 是否固定 5 到 20 个 eval prompts 做训练前后对比。
- 是否记录 train loss、eval loss、样例输出，而不是只看 loss。

常见失败：

- prompt 和 response 都参与 loss，模型学会复读用户问题。
- 训练模板和推理模板不一致，训练效果看不出来。
- 数据太少但 steps 太多，训练 loss 降，eval loss 升。
- 标准答案质量不稳定，模型学到错误格式或错误事实。

## 3. PEFT 和 LoRA

PEFT 是 Parameter-Efficient Fine-Tuning，参数高效微调。它不是单独的训练目标，而是“怎么少更新参数”的工程策略。

LoRA 是最常用的 PEFT 方法。原始线性层是：

```text
y = W x
```

LoRA 后变成：

```text
y = W x + scale * B(Ax)
```

其中：

- `W` 是 base model 原始权重，冻结。
- `A/B` 是 LoRA adapter 新增的小矩阵，训练。
- `scale = lora_alpha / r`。

开发者应该理解：

- LoRA 可以配合 SFT，也可以配合 DPO/RLHF。
- adapter 可以单独保存，加载时必须配合同一个 base model 或兼容结构。
- `target_modules` 决定 LoRA 加在哪里，比如 `q_proj`、`v_proj`、`o_proj`。
- trainable ratio 是可训练参数占总参数比例，不是效果分数。

本课程 06 的实际例子：

```text
base model: Qwen/Qwen2.5-0.5B-Instruct
target_modules: q_proj, v_proj
trainable params: 540,672
total params: 494,573,440
trainable ratio: 0.1093%
```

## 4. QLoRA

QLoRA 通常表示：

```text
低比特量化加载 base model
+ 冻结 base model
+ 训练 LoRA adapter
```

它解决的是显存问题，不是数据问题。QLoRA 适合 GPU 显存不足但又想在较大模型上训练 LoRA 的场景。

开发者注意：

- 量化会引入 bitsandbytes、CUDA、硬件兼容问题。
- Mac 本地学习阶段不建议把 QLoRA 当第一实践目标。
- 先把普通 LoRA 的数据、mask、保存、加载跑明白，再学 QLoRA。

## 5. Continued Pretraining

继续预训练也叫 domain adaptive pretraining。目标仍然是语言建模：

```text
给定前文，预测下一个 token
```

它适合让模型熟悉领域文本分布，例如：

- 法律条款表达
- 医疗病历表达
- 金融研报风格
- 公司内部文档术语
- 代码仓库风格

它不直接教模型“听指令”。如果你的目标是问答格式、客服流程、工具调用格式，通常先做 SFT，而不是继续预训练。

开发者检查点：

- 领域文本是否足够干净。
- 是否会污染模型通用能力。
- 是否有独立的领域 eval set。
- 是否需要在继续预训练后再做 SFT。

## 6. Reward Modeling

Reward Model 学的是“哪个回答更好”。典型数据：

```json
{
  "prompt": "解释什么是梯度累积",
  "chosen": "梯度累积是在显存不足时...",
  "rejected": "梯度累积是一种数据库缓存技术..."
}
```

Reward Model 的输出通常是一个标量分数：

```text
score(prompt, answer)
```

它常用于 RLHF / PPO，也可用于离线评估候选回答。

开发者注意：

- reward model 质量决定后续 RLHF 的方向。
- 偏好数据必须有清晰标注标准。
- 不能只奖励长度、格式或关键词，否则模型会钻奖励漏洞。

## 7. RLHF / PPO

RLHF 是 Reinforcement Learning from Human Feedback。经典 InstructGPT 流程是：

```text
1. SFT: 用人工示范答案训练初始策略
2. Reward Model: 用人类偏好对训练打分模型
3. PPO: 用 reward model 优化策略，同时用 KL 约束别偏离参考模型太远
```

RLHF 适合：

- 有大量偏好数据
- 有 reward model
- 有在线采样和评估基础设施
- 需要优化“人类更喜欢哪个回答”

本地学习阶段不建议先做完整 RLHF，因为它比 SFT/LoRA 多了：

- reward model
- reference model
- policy model
- value model
- KL penalty
- 在线生成和打分循环

开发者可以先理解日志指标：

- `objective/kl`: 当前 policy 偏离 reference 的程度。
- `objective/scores`: reward model 给出的原始分。
- `objective/rlhf_reward`: reward 扣除 KL penalty 后的结果。
- `loss/policy_avg`: policy 更新损失。
- `loss/value_avg`: value model 学 reward 的误差。

## 8. DPO

DPO 是 Direct Preference Optimization。它用偏好数据直接训练 policy，不需要显式训练 reward model，也不需要 PPO 的在线采样循环。

典型数据：

```json
{
  "prompt": "解释什么是梯度累积",
  "chosen": "梯度累积是在显存不足时，把多个 mini-batch 的梯度累加后再更新参数的方法。",
  "rejected": "梯度累积是把数据库查询结果累加起来。"
}
```

核心目标：

```text
同一个 prompt 下，提高 chosen 相对 rejected 的偏好边界
```

开发者检查点：

- chosen 和 rejected 必须针对同一个 prompt。
- rejected 不能太离谱，否则训练信号太容易，泛化弱。
- DPO 通常从 SFT 后的模型开始，而不是从 base model 裸跑。
- 可以用 LoRA adapter 做 DPO，降低训练成本。

什么时候选 DPO：

- 你已经有一批偏好对。
- 你不想训练 reward model。
- 你想比 SFT 更直接地优化“用户更喜欢哪个回答”。

## 9. GRPO、KTO、ORPO、CPO 等偏好优化

这些方法都属于 post-training 或 preference optimization 的扩展路线。

| 方法 | 粗略理解 | 什么时候看 |
|---|---|---|
| GRPO | 用组内相对奖励做强化学习优化，常见于推理任务 | 有 reward function 或可自动判分任务 |
| KTO | 不一定需要成对 chosen/rejected，可用 desirable/undesirable 信号 | 偏好数据不是严格成对时 |
| ORPO | 把 SFT 和偏好优化合在一个目标中 | 想简化 SFT + DPO 两阶段流程时 |
| CPO | 直接做偏好优化，常作为 DPO 类方法对比 | 阅读 preference optimization 代码时 |

学习建议：不要在没掌握 SFT、LoRA、DPO 前直接跳这些方法。它们解决的是更后面的优化问题。

## 10. Distillation

蒸馏是用强模型教弱模型。数据可以来自：

- teacher model 生成的答案
- teacher 的解释过程
- teacher 的偏好判断
- teacher logits，视框架和权限而定

适合：

- 把大模型能力迁移到小模型。
- 为本地部署准备更小模型。
- 生成 SFT 数据草稿，再人工筛选。

风险：

- teacher 错误会被 student 学进去。
- 只蒸馏风格可能不提升真实能力。
- 合成数据必须去重、过滤、抽样人工检查。

## 11. 开发者选型指南

按目标选：

| 目标 | 首选方法 | 备选 |
|---|---|---|
| 模型按固定格式回答 | SFT + LoRA | prompt engineering |
| 模型学习业务语气 | SFT + LoRA | few-shot |
| 模型掌握领域术语表达 | Continued Pretraining -> SFT | RAG |
| 模型减少坏回答、偏好好回答 | DPO + LoRA | RLHF |
| 有人类打分系统和在线采样 | RLHF/PPO | DPO |
| 显存不足 | LoRA / QLoRA | 更小模型 |
| 想保留多个任务版本 | LoRA adapter | full fine-tune |
| 想压缩部署成本 | Distillation | 量化 |

按本地学习顺序选：

```text
1. 数据和 tokenizer
2. Causal LM 最小训练闭环
3. SFT
4. LoRA / PEFT
5. adapter 保存、加载、对比生成
6. DPO 小样本实验
7. reward model 和 RLHF 概念实验
8. QLoRA / 分布式 / 加速框架
```

## 12. 一个标准训练项目应该包含什么

目录建议：

```text
project/
  data/
    raw.jsonl
    train.jsonl
    validation.jsonl
    eval_prompts.jsonl
  scripts/
    prepare_data.py
    train_sft_lora.py
    eval_generate.py
    compare_outputs.py
  outputs/
    adapter/
    checkpoints/
    generations/
    metrics.json
  README.md
```

每次实验至少记录：

- base model 名称和版本。
- tokenizer/chat template。
- 数据版本、样本数、过滤规则。
- max length、batch size、gradient accumulation。
- 学习率、scheduler、warmup、epochs 或 max_steps。
- PEFT 配置：`r`、`alpha`、`dropout`、`target_modules`。
- train loss、eval loss、偏好指标或 reward 指标。
- 固定 eval prompts 的训练前后输出。
- checkpoint / adapter 路径。

## 13. 训练脚本骨架

真实项目可以变化很多，但一个 SFT + LoRA 脚本通常应该按这个顺序写：

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

raw = load_dataset("json", data_files={"train": "data/train.jsonl", "validation": "data/validation.jsonl"})

def preprocess(row):
    user_text = row["instruction"] + ("\n" + row["input"] if row.get("input") else "")
    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user_text},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    answer_text = row["output"] + (tokenizer.eos_token or "")

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:512]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:512]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }

tokenized = raw.map(preprocess, remove_columns=raw["train"].column_names)

base = AutoModelForCausalLM.from_pretrained(model_id)
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)
model = get_peft_model(base, peft_config)
model.print_trainable_parameters()
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, label_pad_token_id=-100)

args = TrainingArguments(
    output_dir="outputs/checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_steps=100,
    eval_strategy="steps",
    eval_steps=20,
    save_steps=20,
    logging_steps=5,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    data_collator=data_collator,
)
trainer.train()
trainer.save_model("outputs/adapter")
```

这个骨架里每一步的输入输出是：

| 步骤 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `load_dataset` | JSONL/CSV/Parquet | `DatasetDict` | 把文件变成可切分、可 map 的数据对象 |
| `preprocess` | 一行原始样本 | `input_ids/attention_mask/labels` | 把业务字段变成模型训练字段，并 mask prompt loss |
| `from_pretrained` | base model id/path | base model | 加载预训练权重和结构 |
| `LoraConfig` | rank、alpha、target modules | PEFT 配置 | 描述 adapter 插入哪里、训练多少参数 |
| `get_peft_model` | base model + PEFT 配置 | PEFT model | 冻结 base，挂上可训练 adapter |
| `DataCollatorForSeq2Seq` | 多条变长 tokenized 样本 | padded batch | 对齐 batch 长度，并用 `-100` padding labels |
| `TrainingArguments` | 超参和输出路径 | 训练配置 | 控制 batch、lr、eval、save、logging |
| `Trainer.train()` | model + dataset + args | checkpoint / metrics | 执行 forward、loss、backward、optimizer step |
| `save_model` | PEFT model | adapter artifact | 保存后续加载和推理需要的 adapter |

严肃 SFT 训练一定要检查 `labels`。如果 `labels` 里 prompt 区域不是 `-100`，模型会被训练去复读用户问题和模板文本。

如果你的 `transformers` 版本不接受 `eval_strategy`，改成旧参数名 `evaluation_strategy`。本仓库课程脚本里做了版本兼容处理。

## 14. 训练前后必须做的检查

训练前：

- 随机打印 5 条样本。
- 解码 `input_ids` 看 prompt 模板是否正确。
- 解码 `labels != -100` 看 loss 是否只打在目标回答上。
- 统计 train/validation 数量和长度分布。
- 确认可训练参数不是 0。

训练中：

- 看 loss 是否下降。
- 看 eval loss 是否异常上升。
- 看 grad_norm 是否爆炸。
- 看学习率是否符合预期。
- 定期生成固定 prompt。

训练后：

- 比较 base / trained / loaded adapter 输出。
- 重新加载 checkpoint 验证结果一致。
- 用不在训练集里的 prompt 做人工评估。
- 检查格式、事实性、安全性、拒答边界。
- 保存实验配置和报告。

## 15. 和当前课程的对应关系

| 能力 | 当前课程 |
|---|---|
| 数据加载、切分、seed | Lesson 01 |
| tokenizer、labels、loss mask | Lesson 02 |
| batch、collator、effective batch | Lesson 03 |
| Trainer 最小闭环 | Lesson 04 |
| 手写 LoRA 机制 | Lesson 05 |
| PEFT + 真实 Qwen LoRA、adapter 保存/加载/输出对比 | Lesson 06 |
| 独立 SFT baseline 和训练前后输出对比 | Lesson 07 |
| DPO / preference optimization | 后续 Lesson 08 |
| RLHF / reward model / PPO | 后续课程 |

下一步建议不是马上做 RLHF，而是在已经完成 SFT baseline 后，进入小样本 DPO：

```text
Lesson 07: 已完成，SFT baseline，用客服工单 -> 严格 JSON 路由任务，固定 eval prompts，比较训练前后输出
Lesson 08: DPO 小样本偏好优化
Lesson 09: Reward model / RLHF 概念实验
Lesson 10: QLoRA / Training Engineering，理解量化和大模型训练工程边界
```

Lesson 07 的场景选择要满足三个条件：

- 训练前后能肉眼判断：是否输出严格 JSON，字段是否齐全，intent/department 是否正确。
- 数据分布足够集中：40-80 条同一格式样本，比 5 条杂糅问答更适合观察 SFT 行为变化。起始数据放在 `lessons/07-sft-baseline/data/train.jsonl`，保持课程自包含。
- 场景足够经典：客服工单路由、信息抽取、工具调用参数生成，都比“解释一个通用概念”更适合展示 SFT。

## 参考资料

- Hugging Face TRL: SFT、DPO、Reward、PPO、GRPO 等 trainer 方法总览: https://huggingface.co/docs/trl/index
- Hugging Face TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
- Hugging Face TRL DPOTrainer: https://huggingface.co/docs/trl/dpo_trainer
- Hugging Face TRL PPOTrainer: https://huggingface.co/docs/trl/ppo_trainer
- Hugging Face PEFT: https://huggingface.co/docs/peft/index
- Hugging Face PEFT Adapters: https://huggingface.co/docs/peft/en/conceptual_guides/adapter
- LoRA paper: https://arxiv.org/abs/2106.09685
- DPO paper: https://arxiv.org/abs/2305.18290
- InstructGPT / RLHF paper: https://arxiv.org/abs/2203.02155
