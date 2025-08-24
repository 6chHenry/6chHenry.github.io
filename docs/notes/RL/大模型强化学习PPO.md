# 大模型强化学习（PPO）代码实现

## 一、前置知识与训练阶段概述
- 大模型训练三阶段：预训练 → 监督微调（SFT）→ 强化学习（PPO），视频聚焦第三阶段
- 强化学习前提：需完成监督微调，并训练一个**奖励模型（Reward Model）** 提供奖励信号
- 建议先观看作者关于PPO算法原理的前置视频

## 二、奖励模型（Reward Model）训练
### 1. 数据要求
- 核心数据：**偏好数据（Preference Data）**，包含同一问题的两个回答（`chosen`优质回答、`rejected`较差回答）
- 优势：相比直接打分，偏好数据更易生成（"不怕不识货，就怕货比货"）

### 2. 模型结构
- 基础模型：采用与待优化大模型能力接近或更强的模型（评价比生成更容易）
- 输出层改造：在大模型基础上增加`score head`（输入为token的hidden size，输出维度为1）
- 评分逻辑：仅对序列最后一个token调用`score head`（可观察完整序列）

### 3. 损失函数（Loss）
- 公式：对`chosen`与`rejected`的得分差值应用`log sigmoid`函数
  ```
  loss = log_sigmoid(score_chosen - score_rejected)
  ```
- 特性：当`score_chosen < score_rejected`时，损失呈指数级增长；反之则快速下降

### 4. 训练代码（基于Hugging Face TRL库）
```python
# 1. 定义分词器和模型（序列分类模型，输出连续值）
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

# 2. 配置LoRA参数（量化训练）
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLASSIFICATION,
    ...  # 其他参数
)
model = get_peft_model(model, lora_config)

# 3. 数据处理（拼接问题与回答，生成输入对）
def process_data(examples):
    chosen_inputs = tokenizer(examples["question"] + examples["chosen"], ...)
    rejected_inputs = tokenizer(examples["question"] + examples["rejected"], ...)
    return {
        "input_ids_chosen": chosen_inputs["input_ids"],
        "attention_mask_chosen": chosen_inputs["attention_mask"],
        "input_ids_rejected": rejected_inputs["input_ids"],
        "attention_mask_rejected": rejected_inputs["attention_mask"],
    }

# 4. 定义训练配置与训练器
reward_config = RewardConfig(output_dir="./reward_model")
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=reward_config,
    train_dataset=processed_dataset,
)
trainer.train()  # 启动训练
```


## 三、PPO模型训练
### 1. 涉及的四个核心模型
| 模型类型         | 作用                                  | 输出层                  | 优化目标                     |
|------------------|---------------------------------------|-------------------------|------------------------------|
| 基准模型         | 限制新模型与原始模型的分布差异        | LM Head（字典维度）     | 无（固定参数）               |
| 训练模型         | 生成文本并优化                        | LM Head                 | 最大化奖励信号               |
| 奖励模型         | 对完整问答序列打分                    | Score Head（维度1）     | 无（已预训练）               |
| 状态价值模型     | 预测每个token序列的期望回报          | Value Head（维度1）     | 最小化价值预测误差           |

- 优化方案：通过**LoRA适配器**共享底层大模型权重，仅加载1个大模型+2个LoRA参数，减少显存占用


### 2. 奖励信号计算
- 每个token的奖励 = KL散度惩罚 + 最终得分
  ```
  reward = (-0.2 * kl_divergence) + score
  ```
  （其中`kl_divergence`为训练模型与基准模型的分布差异，系数-0.2为调整参数）


### 3. 优势函数（GAE）
- 迭代表达式：
  ```
  advantage_t = delta_t + gamma * lambda * advantage_{t+1}
  ```
  （从后向前计算，平衡偏差与方差）


### 4. 损失函数（Loss）
#### （1）状态价值网络损失
- 标签计算：采用广义优势法（`return = advantage + value`）
- 损失公式：
  ```
  vf_loss = 0.5 * mean(max(vf_loss1, vf_loss2))
  其中：
  vf_loss1 = (value_pred - return)^2
  vf_loss2 = (clipped_value_pred - return)^2  # 截断预测值以稳定训练
  ```

#### （2）PPO损失
- 公式：
  ```
  ppo_loss = mean(max(r * (-advantage), clipped_r * (-advantage)))
  其中：
  r = 训练模型概率 / 重要性采样模型概率
  clipped_r = clip(r, 1-epsilon, 1+epsilon)  # 限制概率比值范围
  ```
- 注意：KL散度已融入奖励信号，故损失函数中无需额外包含


### 5. 训练流程（伪代码）
```python
for each batch in prompt_dataset:
    # 生成回答并计算奖励
    responses = importance_sampling_model.generate(batch["queries"])
    scores = reward_model.score(batch["queries"] + responses)
    
    # 计算概率分布、价值与优势
    all_probs, values = importance_sampling_model(batch["queries"] + responses)
    kl_div = calculate_kl(all_probs, base_model_probs)
    rewards = (-0.2 * kl_div) + scores
    advantages = compute_gae(rewards, values)
    
    # 多轮更新（内循环）
    for _ in range(epochs):
        # 计算损失并反向传播
        new_probs, new_values = training_model(batch["queries"] + responses)
        vf_loss = compute_vf_loss(new_values, advantages + values)
        ppo_loss = compute_ppo_loss(new_probs, all_probs, advantages)
        total_loss = ppo_loss + 0.5 * vf_loss
        total_loss.backward()
        optimizer.step()
```


### 6. 训练代码（基于Hugging Face TRL库）
```python
# 1. 配置模型（含价值头的因果语言模型）
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    base_model_name,
    peft_config=lora_config,  # 共享基准模型权重
    reward_adapter=reward_model_path,
)

# 2. 加载数据（仅需问题，回答由模型生成）
dataset = load_dataset("json", data_files="queries.json")
tokenized_dataset = dataset.map(lambda x: tokenizer(x["query"], ...))

# 3. 定义PPO配置
ppo_config = PPOConfig(
    kl_div="forward",  # 标准KL散度计算
    per_device_train_batch_size=4,
    num_train_epochs=3,
    ...
)

# 4. 定义训练器并启动训练
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=base_model,  # 基准模型
    tokenizer=tokenizer,
    dataset=tokenized_dataset,
)

# 5. 训练循环
for batch in ppo_trainer.dataloader:
    queries = batch["input_ids"]
    responses = ppo_trainer.generate(queries, max_new_tokens=100)  # 生成回答
    scores = reward_model.score(queries, responses)  # 评分
    ppo_trainer.step(queries, responses, scores)  # 更新模型
```


## 四、注意事项
1. **数据层面**：偏好数据需保证`chosen`与`rejected`的质量差异明确，否则奖励模型学习效果差
2. **模型层面**：
   - 奖励模型能力需与待优化模型匹配（不可过弱）
   - 通过LoRA技术减少显存占用，避免同时加载多个大模型
3. **训练参数**：
   - 生成回答时需关闭`top_k`，`temperature=1.0`，保证采样覆盖全概率空间
   - KL散度系数（如-0.2）需根据任务调整，平衡探索与稳定
4. **损失计算**：
   - 价值网络需截断预测值，防止训练不稳定
   - PPO损失需限制概率比值范围（通常`epsilon=0.2`）
5. **迭代逻辑**：采用外循环（batch数据）+内循环（多轮更新）结构，提升训练效率


## 五、核心结论
- 强化学习（PPO）是大模型能力逼近预训练极限的关键步骤
- 奖励模型与价值模型的设计是训练核心，需平衡偏差与方差
- 基于Hugging Face TRL库可简化实现，但需理解底层公式与参数含义
- 代码地址：[https://github.com/RethinkFun/trian_ppo/tree/main/train_ppo](https://github.com/RethinkFun/trian_ppo/tree/main/train_ppo)