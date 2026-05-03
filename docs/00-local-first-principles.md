# 00. 本地优先学习原则

微调学习的第一条原则：先控制规模，再追求效果。

本地电脑最适合学习的是训练流程、数据格式、loss 构造、评估和 checkpoint 管理。不要一开始就把目标设成 full fine-tune 7B/13B。

## 本地机器能学什么

### CPU 或普通 Mac

可以学：

- 小型文本分类微调
- 小型 causal LM 语言模型微调
- tokenizer、padding、truncation、labels 构造
- 小样本 SFT 数据格式
- Trainer 参数、checkpoint、eval loop

建议模型：

- `distilbert-base-uncased`
- `bert-base-chinese`
- `sshleifer/tiny-gpt2`
- 小型中文 GPT 或 0.5B 级别模型

不建议：

- 7B full fine-tune
- 大规模 RLHF
- 大规模 diffusion DreamBooth
- 多机分布式训练

### Apple Silicon

可以利用 MPS 做一些 PyTorch 小实验，但要注意：

- 不是所有算子都完全等价于 CUDA
- 量化训练支持通常不如 NVIDIA GPU 稳定
- QLoRA/bitsandbytes 相关流程经常依赖 CUDA
- 先用小模型跑通逻辑，再考虑云 GPU

### 单张 NVIDIA GPU

可以尝试：

- LoRA / PEFT
- 小模型 SFT
- 小规模 DreamBooth 或 textual inversion
- gradient accumulation
- mixed precision

仍然要避免：

- 没有必要的 full fine-tune
- 数据质量没看就扩大训练
- loss 降了就认为模型变好

## 本地学习的最小闭环

每个训练实验都应该包含这些环节：

1. 明确任务：分类、生成、SFT、偏好优化
2. 准备数据：字段、样本数量、训练/验证切分
3. 构造输入：prompt、tokenizer、attention mask
4. 构造 labels：哪些 token 参与 loss
5. 启动训练：learning rate、batch size、epoch
6. 观察训练：loss、eval loss、样例输出
7. 保存结果：checkpoint 或 adapter
8. 加载推理：验证保存结果真的可用
9. 复盘问题：数据、参数、过拟合、格式错误

## 一条硬规则

如果你不能在 20 条样本上让模型明显过拟合，就不要扩大到 2 万条样本。

小样本过拟合测试能快速暴露：

- 数据字段错了
- labels 全是 `-100`
- prompt/response 拼接错了
- tokenizer pad/eos 设置错了
- learning rate 太小或模型参数没被训练

