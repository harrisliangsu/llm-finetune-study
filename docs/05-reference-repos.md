# 05. 参考仓库阅读顺序

这些仓库适合从不同角度学习微调。阅读时不要按 star 数排序，而要按学习阶段排序。

## 1. huggingface/datasets

仓库：

- https://github.com/huggingface/datasets

重点目录：

- `src/`
- `docs/`
- `notebooks/`
- `templates/`

学习内容：

- 数据集加载
- map/filter/shuffle/select
- train/test split
- streaming
- 和 tokenizer 的配合

为什么先学：

微调失败很多时候不是模型问题，而是数据字段、切分、清洗和 labels 问题。

## 2. huggingface/transformers

仓库：

- https://github.com/huggingface/transformers

重点目录：

- `examples/pytorch/text-classification`
- `examples/pytorch/language-modeling`
- `examples/pytorch/summarization`
- `examples/pytorch/translation`
- `examples/pytorch/token-classification`
- `src/transformers`

学习内容：

- `Trainer`
- `TrainingArguments`
- model/tokenizer 加载
- checkpoint
- eval
- generation

本地建议：

先跑 text classification，再跑 language modeling。

## 3. Lordog/dive-into-llms

仓库：

- https://github.com/Lordog/dive-into-llms

重点目录：

- `documents/chapter1`
- `documents/chapter4`
- `documents/chapter11`

学习内容：

- 微调与部署
- 数学推理 SFT
- RLHF 安全对齐

价值：

中文教程友好，适合建立大模型微调全局理解。

## 4. ymcui/Chinese-LLaMA-Alpaca

仓库：

- https://github.com/ymcui/Chinese-LLaMA-Alpaca

重点目录：

- `data/`
- `examples/`
- `notebooks/`
- `scripts/training/`

重点文件：

- `run_clm_pt_with_peft.py`
- `run_clm_sft_with_peft.py`
- `run_pt.sh`
- `run_sft.sh`

学习内容：

- 中文 LLaMA
- PEFT
- LoRA
- 继续预训练
- SFT

本地建议：

先读脚本结构，不要直接追求大模型训练。可以把思想迁移到更小模型上练。

## 5. LianjiaTech/BELLE

仓库：

- https://github.com/LianjiaTech/BELLE

重点目录：

- `train/`
- `data/`
- `eval/`
- `models/`
- `docs/`

学习内容：

- 中文指令数据
- finetune
- LoRA
- DPO/PPO/RLHF
- 评估集
- Deepspeed-Chat 集成

阅读重点：

- 数据如何组织
- 训练脚本如何划分
- 评估集如何设计
- 不同训练数据质量如何影响结果

## 6. LlamaChinese/Llama-Chinese

仓库：

- https://github.com/LlamaChinese/Llama-Chinese

重点目录：

- `train/pretrain`
- `train/sft`
- `train/merge_peft_model`
- `data/`
- `docs/`

学习内容：

- 中文 LLaMA 生态
- pretrain/SFT/PEFT 合并
- 中文资料索引

## 7. LAION-AI/Open-Assistant

仓库：

- https://github.com/LAION-AI/Open-Assistant

重点目录：

- `model/model_training/`
- `oasst-data/`

重点文件：

- `trainer_sft.py`
- `trainer_rm.py`
- `trainer_rl.py`

学习内容：

- SFT
- reward model
- RLHF
- 数据收集和偏好数据

本地建议：

进阶阅读为主。本地只做小样本理解，不要直接复制完整训练规模。

## 8. deepspeedai/DeepSpeed

仓库：

- https://github.com/deepspeedai/DeepSpeed

重点目录：

- `deepspeed/`
- `examples/`
- `docs/`
- `benchmarks/`

学习内容：

- ZeRO
- 数据并行
- 模型并行
- pipeline parallel
- 显存优化

本地建议：

先理解配置和原理。真正体验通常需要 NVIDIA GPU 或多卡环境。

## 9. huggingface/diffusers

仓库：

- https://github.com/huggingface/diffusers

重点目录：

- `examples/text_to_image`
- `examples/dreambooth`
- `examples/textual_inversion`
- `examples/controlnet`

学习内容：

- diffusion 模型微调
- DreamBooth
- textual inversion
- ControlNet

本地建议：

如果目标是 LLM 微调，可以放后面。它更适合图像生成方向。

