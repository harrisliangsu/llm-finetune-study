# 本地微调检查清单

## 训练前

- [ ] 明确任务类型：分类、生成、SFT、LoRA、DPO
- [ ] 明确训练方法维度：数据形态、优化目标、更新哪些参数、评估方式
- [ ] 确认是否真的需要训练，prompt/RAG/schema/few-shot 是否已经尝试过
- [ ] 选定产物形态：完整 checkpoint、LoRA adapter、reward model、DPO adapter 或蒸馏后小模型
- [ ] 数据字段已检查
- [ ] train/validation 已切分
- [ ] 抽样打印原始数据
- [ ] 抽样打印拼接后的 prompt
- [ ] 抽样打印 tokenized decode
- [ ] 抽样打印 labels 中参与 loss 的部分
- [ ] max length 不会截断关键回答
- [ ] pad/eos token 设置合理
- [ ] 保存实验配置

## 启动训练

- [ ] 先跑 20 条样本过拟合
- [ ] 确认 loss 能下降
- [ ] LoRA 场景确认 trainable parameters 不为 0
- [ ] 记录 learning rate
- [ ] 记录 batch size
- [ ] 记录 gradient accumulation
- [ ] 记录 seed
- [ ] 设置 eval strategy
- [ ] 设置 save strategy

## 训练后

- [ ] 记录 train loss
- [ ] 记录 eval loss
- [ ] 固定 prompts 生成样例
- [ ] 比较 base model 和 fine-tuned model
- [ ] 对 adapter 场景比较 base、trained adapter、reloaded adapter 三组输出
- [ ] 保存 checkpoint 或 adapter
- [ ] 从磁盘重新加载验证
- [ ] 记录失败样例
- [ ] 只选择一个变量做下一轮改动

## 进入更大模型前

- [ ] 小模型训练流程完全理解
- [ ] 能解释 labels mask
- [ ] 能解释 LoRA target modules
- [ ] 能解释 checkpoint 和 adapter 差异
- [ ] 能定位常见数据问题
- [ ] 有固定评估集
- [ ] 有实验记录
