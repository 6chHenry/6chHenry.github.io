# Loss Function

## PositiveBCELoss

PositiveBCELoss 是一种 二元交叉熵损失（Binary Cross-Entropy Loss, BCE Loss）的变体，专门用于 仅计算正样本（Positive Samples）的损失。
它的核心思想是：**只关注正类（target=1）的预测概率，忽略负类（target=0）的贡献。**

适用场景:

- 当数据中 **负样本远多于正样本**（类别不平衡）时，传统 BCE Loss 可能被负样本主导，导致模型对正样本学习不足。

- 希望 **更聚焦正样本的优化**（如医学检测中的罕见病例、异常检测等）。

```python

import torch.nn as nn
import torch.nn.functional as F

class PositiveBCELoss(nn.Module):
    def __init__(self, class_weight=None):
        super().__init__()
        self.register_buffer("class_weight", class_weight)
        
    def forward(self, logit, target):
        x = F.logsigmoid(logit) * target  # 关键步骤：仅正样本参与计算
        if self.class_weight is not None:
            x = x * self.class_weight   # 可选：正样本的类别权重
        loss = -(x.mean())              # 取平均并取负（最大化对数概率 = 最小化损失）
        return loss

```

代码讲解：

- F.logsigmoid(logit)

对模型的原始输出 logit 计算对数 Sigmoid（即 log(σ(logit))），等价于 log(1 / (1 + exp(-logit)))。

这表示 预测正类的对数概率（因为 σ(logit) 是预测正类的概率）。

> [!INFO] 在二元分类问题中，σ(logit)（即 Sigmoid 函数）的输出被解释为 预测正类的概率，这是由逻辑回归（Logistic Regression）的数学原理决定的。下面详细解释为什么：
 
>  Sigmoid 函数的作用:Sigmoid 函数定义为：
$\sigma(x) = \frac{1}{1+e^{-x}}$
它将任意实数x（即 logit）映射到 (0,1) 区间，输出可以直观理解为概率。

> 从 Logit 到概率的推导
> 在二元分类中：
模型的原始输出是 logit（也称为“对数几率”），范围是 $(-\infty,+\infty)$。
通过 Sigmoid 将 logit 转换为概率：
$P(y=1∣x)=σ(logit)$
​如果 logit 很大（如 +∞），P(y=1)→1。
如果 logit 很小（如 -∞）, P(y=1)→0。

> 为什么是“正类”的概率？
对数几率的解释：
logit 的本质是 正类概率的对数几率（log-odds）：
$logit=log(\frac{P(y=1)}{P(y=0)})$
通过 Sigmoid 的反向推导：
$\sigma(logit)=\frac{P(y=1)}{P(y=0)+P(y=1)}=P(y=1)$
因此，σ(logit) 直接表示正类的概率。


- \* target

通过 target（0 或 1 的掩码）过滤，仅保留正样本的损失（target=1 的位置保留值，target=0 的位置归零）。

- class_weight（可选）

如果提供 class_weight，会对正样本的损失加权（常用于进一步平衡类别）。

- -(x.mean())

对剩余的正样本损失取平均，并取负号（因为 log(p) 是负数，取负后得到正损失值）。

## L1 Loss (Mean Absolute Error - MAE)

### 数学表达式
$$
\mathcal{L}(x, y) = \frac{1}{n}\sum_{i=1}^{n} |x_i - y_i|
$$
其中：
- \( x \) 是预测值（模型输出）
- \( y \) 是目标值（真实标签）
- \( n \) 是样本数量

### 适用场景
1. **回归任务**：当需要预测连续值且对异常值不敏感时
2. **鲁棒性要求高**：相比L2 Loss更不易受异常值影响
3. **稀疏梯度需求**：在需要梯度大小恒定的场景

### 参数说明
在PyTorch中通过`torch.nn.L1Loss`实现，主要参数：

```python
torch.nn.L1Loss(
    size_average=None,  # 已废弃（默认取mean）
    reduce=None,        # 已废弃
    reduction='mean'    # 可选：'none'|'mean'|'sum'
)
"""
reduction:

'none'：返回逐元素损失 The same shape as Input

'mean'（默认）：返回损失均值  (1,)

'sum'：返回损失总和  (1,)
"""
```
[L1 Loss官方文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.L1Loss.html)

## Negative Log Likelihood Loss (NLL Loss)

### 数学表达式
$$
\mathcal{L}(x, y) = -\frac{1}{n}\sum_{i=1}^{n} x_{i,y_i}
$$
或更一般的形式：
$$
\mathcal{L}(x, y) = -\sum_{i=1}^{n} \log(p_{y_i})
$$
其中：
- \( x \) 是预测的对数概率（log-probabilities），形状为 (N, C)
- \( y \) 是目标类别索引，形状为 (N)
- \( p_{y_i} \) 是对应真实类别的预测概率
- \( n \) 是 batch size
- \( C \) 是类别数

### 适用场景
1. **多分类问题**：常与LogSoftmax配合使用
2. **神经网络输出概率分布**：要求输入已经是对数概率
3. **语言模型/文本分类**：在NLP任务中广泛使用

### 参数说明
PyTorch实现：`torch.nn.NLLLoss`

```python
torch.nn.NLLLoss(
    weight=None,        # 各类别的权重（1D Tensor）
    size_average=None,  # 已废弃
    ignore_index=-100,  # 忽略的目标类别索引
    reduce=None,        # 已废弃
    reduction='mean'    # 'none'|'mean'|'sum'
)
"""
ignore_index：指定忽略的类别（不贡献梯度）

weight：类别不平衡时可通过此参数调整权重
"""
```

注意：

- 输入需要先经过LogSoftmax
- **与CrossEntropyLoss的关系：CrossEntropy = LogSoftmax + NLLLoss**
- 最小化负对数似然等价于最大化似然函数

示例：

```python

m = nn.LogSoftmax(dim=1)
loss = nn.NLLLoss()
input = m(torch.randn(3, 5))
target = torch.tensor([1, 0, 4])
output = loss(input, target)

```

[NLLLoss官方文档](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html)

## Poisson Negative Log Likelihood Loss (PoissonNLLLoss)

### 数学表达式
$$
\mathcal{L}(x, y) = \frac{1}{n}\sum_{i=1}^{n} (x_i - y_i \log(x_i))
$$
当设置 `log_input=False`（默认）时计算上式。  
当 `log_input=True` 时（输入已取对数）：
$$
\mathcal{L}(x, y) = \frac{1}{n}\sum_{i=1}^{n} (\exp(x_i) - y_i x_i)
$$

### 适用场景
1. **计数数据建模**：事件发生次数的预测（泊松分布）
2. **非负连续值预测**：如流量预测、疾病发病率
3. **输入输出呈泊松分布关系**：方差等于均值的情况

### 参数说明
PyTorch实现：`torch.nn.PoissonNLLLoss`

```python
torch.nn.PoissonNLLLoss(
    log_input=True,      # 输入是否为对数形式（默认True）
    full=False,          # 是否计算完整损失（带斯特林近似项）
    size_average=None,   # 已废弃
    eps=1e-8,            # 防止log(0)的小数值
    reduce=None,         # 已废弃
    reduction='mean'     # 'none'|'mean'|'sum'
)
"""
log_input：

True：输入需先取对数（更数值稳定）

False：直接使用原始输入

full：

True：增加斯特林近似项 y*log(y) - y + 0.5*log(2πy)

False（默认）：仅计算基本项

eps：防止数值不稳定（当 log_input=False 且输入接近0时）
"""
```

[PoissonNLLLoss官方文档](https://pytorch.org/docs/stable/generated/torch.nn.PoissonNLLLoss.html)

## Gaussian Negative Log Likelihood Loss (GaussianNLLLoss)

### 数学表达式
$$
\mathcal{L}(x, y, \sigma) = \frac{1}{n}\sum_{i=1}^{n} \left( \frac{(x_i - y_i)^2}{2\sigma_i^2} + \log(\sigma_i) \right)
$$
其中：
- \( x \) 是预测值（模型输出）
- \( y \) 是目标值（真实标签）
- \( \sigma \) 是预测的标准差（不确定性）
- \( n \) 是样本数量

### 适用场景
1. **不确定性建模**：需要同时预测值和不确定性
2. **贝叶斯神经网络**：输出概率分布而非单点预测
3. **异方差回归**：方差随输入变化的回归任务
4. **安全关键应用**：如自动驾驶、医疗诊断等需要知道预测置信度的场景

### 参数说明
PyTorch实现：`torch.nn.GaussianNLLLoss`

```python
torch.nn.GaussianNLLLoss(
    full=False,          # 是否计算完整损失（带常数项）
    eps=1e-6,            # 防止方差为零的小数值
    reduction='mean'     # 'none'|'mean'|'sum'
)
"""
full：
True：增加常数项 0.5*log(2π)
False（默认）：不包含常数项

eps：防止数值不稳定（当方差接近0时）
"""
```

[GaussianNLLLoss官方文档](https://pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html)

## Kullback-Leibler Divergence Loss (KLDivLoss)

### 数学表达式
$$
\mathcal{L}(x, y) = \frac{1}{n}\sum_{i=1}^{n} \sum_{j=1}^{C} y_{i,j} \left( \log(y_{i,j}) - x_{i,j} \right)
$$
其中：
- \( x \) 是预测的对数概率（log-probabilities），形状为 (N, C)
- \( y \) 是目标概率分布，形状为 (N, C)
- \( n \) 是 batch size
- \( C \) 是类别数

### 适用场景
1. **分布匹配**：衡量两个概率分布的差异
2. **变分自编码器（VAE）**：作为正则化项
3. **知识蒸馏**：让学生模型学习教师模型的概率分布
4. **生成模型**：训练模型生成接近目标分布的数据

### 参数说明
PyTorch实现：`torch.nn.KLDivLoss`

```python
torch.nn.KLDivLoss(
    size_average=None,  # 已废弃
    reduce=None,        # 已废弃
    reduction='mean',   # 'none'|'batchmean'|'mean'|'sum'
    log_target=False    # 目标是否为对数概率
)
"""
reduction：
'none'：返回逐元素损失
'batchmean'：返回batch的损失和（推荐用于KL散度）
'mean'：返回损失均值
'sum'：返回损失总和

log_target：
True：目标已经是对数概率
False（默认）：目标是普通概率
"""
```

注意：
- 输入x需要是对数概率（log-probabilities）
- 目标y需要是概率分布（sum to 1）
- KL散度是非对称的：KL(P||Q) ≠ KL(Q||P)

示例：

```python
# 输入需要是对数概率
input = F.log_softmax(torch.randn(3, 5), dim=1)
# 目标需要是概率分布
target = F.softmax(torch.randn(3, 5), dim=1)
loss = nn.KLDivLoss(reduction='batchmean')
output = loss(input, target)
```

[KLDivLoss官方文档](https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html)

## Mean Squared Error Loss (MSELoss)

### 数学表达式
$$
\mathcal{L}(x, y) = \frac{1}{n}\sum_{i=1}^{n} (x_i - y_i)^2
$$
其中：
- \( x \) 是预测值（模型输出）
- \( y \) 是目标值（真实标签）
- \( n \) 是样本数量

### 适用场景
1. **回归任务**：预测连续值的标准损失函数
2. **图像重建**：如自编码器、图像去噪等
3. **平滑数据**：当数据噪声较小且需要精确预测时
4. **梯度下降优化**：提供平滑的梯度信号

### 参数说明
PyTorch实现：`torch.nn.MSELoss`

```python
torch.nn.MSELoss(
    size_average=None,  # 已废弃
    reduce=None,        # 已废弃
    reduction='mean'    # 'none'|'mean'|'sum'
)
"""
reduction：
'none'：返回逐元素损失 The same shape as Input
'mean'（默认）：返回损失均值 (1,)
'sum'：返回损失总和 (1,)
"""
```

注意：
- MSE对异常值敏感，因为误差被平方
- 相比L1 Loss，MSE在接近最优点时梯度更小，有助于精细调整
- 最小化MSE等价于最大化高斯分布下的似然函数

[MSELoss官方文档](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)

## Binary Cross Entropy Loss (BCELoss)

### 数学表达式
$$
\mathcal{L}(x, y) = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log(x_i) + (1 - y_i) \log(1 - x_i) \right]
$$
其中：
- \( x \) 是预测概率，范围在 [0, 1]
- \( y \) 是目标值（0 或 1）
- \( n \) 是样本数量

### 适用场景
1. **二元分类任务**：判断样本属于两个类别中的哪一个
2. **概率预测**：输出样本属于正类的概率
3. **多标签分类**：每个样本可以属于多个类别
4. **医学诊断**：如疾病检测、风险预测等

### 参数说明
PyTorch实现：`torch.nn.BCELoss`

```python
torch.nn.BCELoss(
    weight=None,        # 各样本的权重（1D Tensor）
    size_average=None,  # 已废弃
    reduce=None,        # 已废弃
    reduction='mean'    # 'none'|'mean'|'sum'
)
"""
weight：可用于样本不平衡时的加权

reduction：
'none'：返回逐元素损失
'mean'（默认）：返回损失均值
'sum'：返回损失总和
"""
```

注意：
- 输入需要经过Sigmoid函数，确保在[0,1]范围内
- 如果输入接近0或1，可能导致数值不稳定
- 对于数值稳定性更好的实现，推荐使用BCEWithLogitsLoss

示例：

```python
m = nn.Sigmoid()
loss = nn.BCELoss()
input = m(torch.randn(3))
target = torch.empty(3).random_(2)
output = loss(input, target)
```

[BCELoss官方文档](https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html)

## Binary Cross Entropy with Logits Loss (BCEWithLogitsLoss)

### 数学表达式
$$
\mathcal{L}(x, y) = -\frac{1}{n}\sum_{i=1}^{n} \left[ y_i \log(\sigma(x_i)) + (1 - y_i) \log(1 - \sigma(x_i)) \right]
$$
其中：
- \( x \) 是预测的原始输出（logits）
- \( y \) 是目标值（0 或 1）
- \( \sigma \) 是Sigmoid函数：\(\sigma(x) = \frac{1}{1+e^{-x}}\)
- \( n \) 是样本数量

### 适用场景
1. **二元分类任务**：与BCELoss相同的应用场景
2. **数值稳定性要求高**：相比BCELoss + Sigmoid更稳定
3. **深度神经网络**：避免梯度消失/爆炸问题
4. **大规模训练**：更适合大规模数据集训练

### 参数说明
PyTorch实现：`torch.nn.BCEWithLogitsLoss`

```python
torch.nn.BCEWithLogitsLoss(
    weight=None,        # 各样本的权重（1D Tensor）
    size_average=None,  # 已废弃
    reduce=None,        # 已废弃
    reduction='mean',   # 'none'|'mean'|'sum'
    pos_weight=None     # 正样本的权重（用于类别不平衡）
)
"""
weight：样本级别的权重

pos_weight：正样本的权重，可用于处理类别不平衡
如果设置，正样本的损失将乘以该值

reduction：
'none'：返回逐元素损失
'mean'（默认）：返回损失均值
'sum'：返回损失总和
"""
```

注意：
- 结合了Sigmoid和BCELoss，数值更稳定
- 使用log-sum-exp技巧避免数值溢出
- 推荐使用BCEWithLogitsLoss而非单独的Sigmoid + BCELoss
- pos_weight参数对于正负样本不平衡的情况很有用

[BCEWithLogitsLoss官方文档](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)

## Hinge Embedding Loss

### 数学表达式
$$
\mathcal{L}(x, y) = \frac{1}{n}\sum_{i=1}^{n} \begin{cases}
x_i, & \text{if } y_i = 1 \\
\max(0, \text{margin} - x_i), & \text{if } y_i = -1
\end{cases}
$$
其中：
- \( x \) 是预测值（通常是相似度得分）
- \( y \) 是目标值（1 或 -1）
- \( n \) 是样本数量
- margin 是边界参数（默认为1.0）

### 适用场景
1. **度量学习**：学习嵌入空间中的相似性
2. **人脸识别/验证**：判断两张人脸是否为同一个人
3. **相似性学习**：学习样本间的相似度度量
4. **半监督学习**：利用相似性信息进行学习

### 参数说明
PyTorch实现：`torch.nn.HingeEmbeddingLoss`

```python
torch.nn.HingeEmbeddingLoss(
    margin=1.0,         # 边界参数
    size_average=None,  # 已废弃
    reduce=None,        # 已废弃
    reduction='mean'    # 'none'|'mean'|'sum'
)
"""
margin：
当y=-1时，如果x < margin，则计算margin - x
否则损失为0

reduction：
'none'：返回逐元素损失
'mean'（默认）：返回损失均值
'sum'：返回损失总和
"""
```

注意：
- 目标标签应为1（相似）或-1（不相似）
- 对于相似样本（y=1），直接使用预测值作为损失
- 对于不相似样本（y=-1），只有当预测值小于margin时才有损失

[HingeEmbeddingLoss官方文档](https://pytorch.org/docs/stable/generated/torch.nn.HingeEmbeddingLoss.html)

## Multi Label Margin Loss

### 数学表达式
$$
\mathcal{L}(x, y) = \frac{1}{n}\sum_{i=1}^{n} \sum_{j} \max(0, \text{margin} - (x_{i,y_i} - x_{i,j}))
$$
其中：
- \( x \) 是预测值（通常是相似度得分）
- \( y \) 是目标值（包含正样本类别的索引）
- \( n \) 是样本数量
- \( j \) 遍历所有不属于 \( y_i \) 的类别
- margin 是边界参数（默认为1.0）

### 适用场景
1. **多标签分类**：每个样本可以属于多个类别
2. **图像标注**：一张图像可以有多个标签
3. **文本分类**：一篇文章可以属于多个主题
4. **推荐系统**：一个用户可能喜欢多个类别

### 参数说明
PyTorch实现：`torch.nn.MultiLabelMarginLoss`

```python
torch.nn.MultiLabelMarginLoss(
    size_average=None,  # 已废弃
    reduce=None,        # 已废弃
    reduction='mean'    # 'none'|'mean'|'sum'
)
"""
reduction：
'none'：返回逐样本损失
'mean'（默认）：返回损失均值
'sum'：返回损失总和
"""
```

注意：
- 目标标签应包含所有正样本类别的索引
- 对于每个样本，损失计算正类别与所有负类别之间的margin
- 目标标签的维度应为 (N, C)，其中C是类别数
- 目标中，正样本位置的值应为1，负样本位置的值应为0

示例：

```python
loss = nn.MultiLabelMarginLoss()
input = torch.randn(3, 5)
# 目标：第一个样本属于类别0和2，第二个样本属于类别1和3，第三个样本属于类别0和4
target = torch.tensor([[1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 1]])
output = loss(input, target)
```

[MultiLabelMarginLoss官方文档](https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelMarginLoss.html)

## Smooth L1 Loss

### 数学表达式
$$
\mathcal{L}(x, y) = \frac{1}{n}\sum_{i=1}^{n} \begin{cases}
0.5 (x_i - y_i)^2 / \beta, & \text{if } |x_i - y_i| < \beta \\
|x_i - y_i| - 0.5 \beta, & \text{otherwise}
\end{cases}
$$
其中：
- \( x \) 是预测值（模型输出）
- \( y \) 是目标值（真实标签）
- \( n \) 是样本数量
- \( \beta \) 是平滑参数（默认为1.0）

### 适用场景
1. **目标检测**：如Faster R-CNN中的边界框回归
2. **鲁棒回归**：对异常值不敏感的回归任务
3. **深度学习训练**：作为L1和L2损失的折中方案
4. **计算机视觉**：需要既平滑又鲁棒的损失函数

### 参数说明
PyTorch实现：`torch.nn.SmoothL1Loss`

```python
torch.nn.SmoothL1Loss(
    size_average=None,  # 已废弃
    reduce=None,        # 已废弃
    reduction='mean',   # 'none'|'mean'|'sum'
    beta=1.0            # 平滑参数
)
"""
beta：
指定从二次损失转为线性损失的阈值点
默认值为1.0

reduction：
'none'：返回逐元素损失
'mean'（默认）：返回损失均值
'sum'：返回损失总和
"""
```

注意：
- 当误差小于beta时，使用二次损失（类似MSE）
- 当误差大于beta时，使用线性损失（类似MAE）
- 这种设计使得损失函数对异常值不敏感，同时在小误差处保持平滑
- 在目标检测中常用于边界框回归，因为对异常值不敏感

[SmoothL1Loss官方文档](https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html)

## Huber Loss

### 数学表达式
$$
\mathcal{L}(x, y) = \frac{1}{n}\sum_{i=1}^{n} \begin{cases}
0.5 (x_i - y_i)^2, & \text{if } |x_i - y_i| < \delta \\
\delta (|x_i - y_i| - 0.5 \delta), & \text{otherwise}
\end{cases}
$$
其中：
- \( x \) 是预测值（模型输出）
- \( y \) 是目标值（真实标签）
- \( n \) 是样本数量
- \( \delta \) 是阈值参数（默认为1.0）

### 适用场景
1. **鲁棒回归**：对异常值不敏感的回归任务
2. **强化学习**：作为Q-learning的损失函数
3. **控制理论**：系统控制中的误差度量
4. **金融预测**：对极端值不敏感的预测任务

### 参数说明
PyTorch实现：`torch.nn.HuberLoss`

```python
torch.nn.HuberLoss(
    reduction='mean',   # 'none'|'mean'|'sum'
    delta=1.0           # 阈值参数
)
"""
delta：
指定从二次损失转为线性损失的阈值点
默认值为1.0

reduction：
'none'：返回逐元素损失
'mean'（默认）：返回损失均值
'sum'：返回损失总和
"""
```

注意：
- 当误差小于delta时，使用二次损失（类似MSE）
- 当误差大于delta时，使用线性损失（类似MAE）
- Huber Loss是SmoothL1Loss的泛化形式，当delta=1.0时两者等价
- 对异常值不敏感，同时在小误差处保持平滑的梯度

[HuberLoss官方文档](https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html)

## Soft Margin Loss

### 数学表达式
$$
\mathcal{L}(x, y) = \frac{1}{n}\sum_{i=1}^{n} \log(1 + \exp(-y_i x_i))
$$
其中：
- \( x \) 是预测值（模型输出）
- \( y \) 是目标值（1 或 -1）
- \( n \) 是样本数量

### 适用场景
1. **二元分类**：与Hinge Loss类似的应用场景
2. **概率输出需求**：需要概率解释的分类任务
3. **SVM替代**：作为SVM的平滑替代方案
4. **神经网络训练**：提供平滑的梯度信号

### 参数说明
PyTorch实现：`torch.nn.SoftMarginLoss`

```python
torch.nn.SoftMarginLoss(
    size_average=None,  # 已废弃
    reduce=None,        # 已废弃
    reduction='mean'    # 'none'|'mean'|'sum'
)
"""
reduction：
'none'：返回逐元素损失
'mean'（默认）：返回损失均值
'sum'：返回损失总和
"""
```

注意：
- 目标标签应为1（正类）或-1（负类）
- Soft Margin Loss是Hinge Loss的平滑版本
- 相比Hinge Loss，Soft Margin Loss处处可导，更适合梯度下降
- 损失值范围在[0, +∞)，当预测正确且置信度高时接近0

示例：

```python
loss = nn.SoftMarginLoss()
input = torch.randn(3, requires_grad=True)
target = torch.tensor([-1, 1, 1], dtype=torch.float)
output = loss(input, target)
```

[SoftMarginLoss官方文档](https://pytorch.org/docs/stable/generated/torch.nn.SoftMarginLoss.html)

## Cross Entropy Loss

### 数学表达式
$$
\mathcal{L}(x, y) = -\frac{1}{n}\sum_{i=1}^{n} \log\left(\frac{\exp(x_{i,y_i})}{\sum_{j=1}^{C} \exp(x_{i,j})}\right)
$$
或等价形式：
$$
\mathcal{L}(x, y) = -\frac{1}{n}\sum_{i=1}^{n} \left[ x_{i,y_i} - \log\left(\sum_{j=1}^{C} \exp(x_{i,j})\right) \right]
$$
其中：
- \( x \) 是预测的原始输出（logits），形状为 (N, C)
- \( y \) 是目标类别索引，形状为 (N)
- \( n \) 是 batch size
- \( C \) 是类别数

### 适用场景
1. **多分类任务**：标准的多类别分类问题
2. **图像分类**：如MNIST、CIFAR、ImageNet等
3. **自然语言处理**：文本分类、情感分析等
4. **语音识别**：音频分类任务

### 参数说明
PyTorch实现：`torch.nn.CrossEntropyLoss`

```python
torch.nn.CrossEntropyLoss(
    weight=None,        # 各类别的权重（1D Tensor）
    size_average=None,  # 已废弃
    ignore_index=-100,  # 忽略的目标类别索引
    reduce=None,        # 已废弃
    reduction='mean',   # 'none'|'mean'|'sum'
    label_smoothing=0.0 # 标签平滑参数
)
"""
weight：类别不平衡时可通过此参数调整权重

ignore_index：指定忽略的类别（不贡献梯度）

reduction：
'none'：返回逐样本损失
'mean'（默认）：返回损失均值
'sum'：返回损失总和

label_smoothing：
标签平滑系数，范围[0,1]
0表示不进行平滑，1表示完全平滑
"""
```

注意：
- CrossEntropyLoss = LogSoftmax + NLLLoss
- 输入不需要经过Softmax，内部会自动处理
- 对类别不平衡问题，可以通过weight参数调整
- label_smoothing有助于提高模型的泛化能力和鲁棒性

示例：

```python
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
```

[CrossEntropyLoss官方文档](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)

## Multi Label Soft Margin Loss

### 数学表达式
$$
\mathcal{L}(x, y) = -\frac{1}{n}\sum_{i=1}^{n} \frac{1}{C} \sum_{j=1}^{C} \left[ y_{i,j} \log(\sigma(x_{i,j})) + (1 - y_{i,j}) \log(1 - \sigma(x_{i,j})) \right]
$$
其中：
- \( x \) 是预测的原始输出（logits），形状为 (N, C)
- \( y \) 是目标值（0 或 1），形状为 (N, C)
- \( \sigma \) 是Sigmoid函数：\(\sigma(x) = \frac{1}{1+e^{-x}}\)
- \( n \) 是 batch size
- \( C \) 是类别数

### 适用场景
1. **多标签分类**：每个样本可以属于多个类别
2. **图像标注**：一张图像可以有多个标签
3. **文本分类**：一篇文章可以属于多个主题
4. **推荐系统**：一个用户可能喜欢多个类别

### 参数说明
PyTorch实现：`torch.nn.MultiLabelSoftMarginLoss`

```python
torch.nn.MultiLabelSoftMarginLoss(
    weight=None,        # 各类别的权重（1D Tensor）
    size_average=None,  # 已废弃
    reduce=None,        # 已废弃
    reduction='mean'    # 'none'|'mean'|'sum'
)
"""
weight：类别不平衡时可通过此参数调整权重

reduction：
'none'：返回逐样本损失
'mean'（默认）：返回损失均值
'sum'：返回损失总和
"""
```

注意：
- 目标标签应为0或1，表示样本是否属于该类别
- 内部使用Sigmoid函数将logits转换为概率
- 相比使用多个BCEWithLogitsLoss，这个损失函数对多标签任务进行了优化
- 适用于标签之间可能存在相关性的多标签任务

示例：

```python
loss = nn.MultiLabelSoftMarginLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, 5).random_(2)
output = loss(input, target)
```

[MultiLabelSoftMarginLoss官方文档](https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelSoftMarginLoss.html)

## Cosine Embedding Loss

### 数学表达式
$$
\mathcal{L}(x, y) = \frac{1}{n}\sum_{i=1}^{n} \begin{cases}
1 - \cos(x_{i1}, x_{i2}), & \text{if } y_i = 1 \\
\max(0, \cos(x_{i1}, x_{i2}) - \text{margin}), & \text{if } y_i = -1
\end{cases}
$$
其中：
- \( x \) 是输入张量对，形状为 (N, *)
- \( y \) 是目标值（1 或 -1），形状为 (N)
- \( \cos(x_{i1}, x_{i2}) \) 是余弦相似度
- \( n \) 是样本数量
- margin 是边界参数（默认为0.0）

### 适用场景
1. **度量学习**：学习嵌入空间中的相似性
2. **人脸识别/验证**：判断两张人脸是否为同一个人
3. **相似性学习**：学习样本间的相似度度量
4. **孪生网络**：训练孪生神经网络

### 参数说明
PyTorch实现：`torch.nn.CosineEmbeddingLoss`

```python
torch.nn.CosineEmbeddingLoss(
    margin=0.0,         # 边界参数
    size_average=None,  # 已废弃
    reduce=None,        # 已废弃
    reduction='mean'    # 'none'|'mean'|'sum'
)
"""
margin：
当y=-1时，如果余弦相似度大于margin，则计算余弦相似度-margin
否则损失为0
默认值为0.0

reduction：
'none'：返回逐样本损失
'mean'（默认）：返回损失均值
'sum'：返回损失总和
"""
```

注意：
- 输入应为两个形状相同的张量（input1, input2）
- 目标标签应为1（相似）或-1（不相似）
- 余弦相似度计算：\(\cos(a, b) = \frac{a \cdot b}{||a|| \cdot ||b||}\)
- 对于相似样本（y=1），最大化余弦相似度
- 对于不相似样本（y=-1），最小化余弦相似度

示例：

```python
loss = nn.CosineEmbeddingLoss()
input1 = torch.randn(3, 5, requires_grad=True)
input2 = torch.randn(3, 5, requires_grad=True)
target = torch.tensor([1, -1, 1], dtype=torch.float)
output = loss(input1, input2, target)
```

[CosineEmbeddingLoss官方文档](https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html)

## Margin Ranking Loss

### 数学表达式
$$
\mathcal{L}(x_1, x_2, y) = \frac{1}{n}\sum_{i=1}^{n} \max(0, -y_i (x_{1i} - x_{2i}) + \text{margin})
$$
其中：
- \( x_1 \) 是第一个输入张量，形状为 (N, *)
- \( x_2 \) 是第二个输入张量，形状为 (N, *)
- \( y \) 是目标值（1 或 -1），形状为 (N)
- \( n \) 是样本数量
- margin 是边界参数（默认为0.0）

### 适用场景
1. **学习排序**：训练模型学习正确的排序关系
2. **推荐系统**：学习用户偏好排序
3. **信息检索**：文档相关性排序
4. **度量学习**：学习样本间的相对距离

### 参数说明
PyTorch实现：`torch.nn.MarginRankingLoss`

```python
torch.nn.MarginRankingLoss(
    margin=0.0,         # 边界参数
    size_average=None,  # 已废弃
    reduce=None,        # 已废弃
    reduction='mean'    # 'none'|'mean'|'sum'
)
"""
margin：
指定期望的边界大小
默认值为0.0

reduction：
'none'：返回逐元素损失
'mean'（默认）：返回损失均值
'sum'：返回损失总和
"""
```

注意：
- 目标标签应为1（x1应排名高于x2）或-1（x2应排名高于x1）
- 当y=1时，希望x1 > x2 + margin
- 当y=-1时，希望x2 > x1 + margin
- 如果条件满足，损失为0；否则损失为差距的绝对值

示例：

```python
loss = nn.MarginRankingLoss()
input1 = torch.randn(3, requires_grad=True)
input2 = torch.randn(3, requires_grad=True)
target = torch.tensor([1, -1, 1], dtype=torch.float)
output = loss(input1, input2, target)
```

[MarginRankingLoss官方文档](https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html)

## Multi Margin Loss

### 数学表达式
$$
\mathcal{L}(x, y) = \frac{1}{n}\sum_{i=1}^{n} \frac{1}{C} \sum_{j \neq y_i} \max(0, \text{margin} - (x_{i,y_i} - x_{i,j}) + p_j)
$$
其中：
- \( x \) 是预测值（模型输出），形状为 (N, C)
- \( y \) 是目标类别索引，形状为 (N)
- \( n \) 是 batch size
- \( C \) 是类别数
- margin 是边界参数（默认为1.0）
- \( p_j \) 是第j类的尺度因子（由weight参数决定）

### 适用场景
1. **多分类任务**：特别是需要区分相似类别的情况
2. **图像分类**：如细粒度分类任务
3. **文本分类**：区分语义相近的类别
4. **语音识别**：区分发音相似的词

### 参数说明
PyTorch实现：`torch.nn.MultiMarginLoss`

```python
torch.nn.MultiMarginLoss(
    p=1,                # 范数参数（仅支持1或2）
    margin=1.0,         # 边界参数
    weight=None,        # 各类别的权重（1D Tensor）
    size_average=None,  # 已废弃
    reduce=None,        # 已废弃
    reduction='mean'    # 'none'|'mean'|'sum'
)
"""
p：
范数参数，仅支持1或2
默认值为1（L1范数）

margin：
指定期望的边界大小
默认值为1.0

weight：类别不平衡时可通过此参数调整权重

reduction：
'none'：返回逐样本损失
'mean'（默认）：返回损失均值
'sum'：返回损失总和
"""
```

注意：
- 对于每个样本，计算正确类别与其他所有类别之间的边界
- 当p=1时，使用L1范数；当p=2时，使用L2范数
- 如果正确类别的分数与其他类别的分数差距大于margin，则损失为0
- 适用于需要强制模型区分相似类别的任务

示例：

```python
loss = nn.MultiMarginLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.tensor([1, 0, 4])
output = loss(input, target)
```

[MultiMarginLoss官方文档](https://pytorch.org/docs/stable/generated/torch.nn.MultiMarginLoss.html)

## Triplet Margin Loss

### 数学表达式
$$
\mathcal{L}(a, p, n) = \frac{1}{n}\sum_{i=1}^{n} \max(0, d(a_i, p_i) - d(a_i, n_i) + \text{margin})
$$
其中：
- \( a \) 是锚点样本（anchor），形状为 (N, *)
- \( p \) 是正样本（positive），形状为 (N, *)
- \( n \) 是负样本（negative），形状为 (N, *)
- \( d(x, y) \) 是距离函数（默认为欧氏距离）
- \( n \) 是样本数量
- margin 是边界参数（默认为1.0）

### 适用场景
1. **度量学习**：学习嵌入空间中的距离度量
2. **人脸识别**：学习人脸特征嵌入
3. **图像检索**：学习图像特征表示
4. **推荐系统**：学习用户和物品的嵌入表示

### 参数说明
PyTorch实现：`torch.nn.TripletMarginLoss`

```python
torch.nn.TripletMarginLoss(
    margin=1.0,         # 边界参数
    p=2.0,              # 范数参数（距离函数的阶数）
    eps=1e-6,           # 数值稳定性小量
    swap=False,         # 是否使用swap
    size_average=None,  # 已废弃
    reduce=None,        # 已废弃
    reduction='mean'    # 'none'|'mean'|'sum'
)
"""
margin：
指定期望的边界大小
默认值为1.0

p：
范数参数，用于计算距离
默认值为2.0（欧氏距离）

eps：
防止除以零的小数值
默认值为1e-6

swap：
如果为True，当d(a,n) < d(a,p)时，计算d(p,n) - d(a,p) + margin
默认值为False

reduction：
'none'：返回逐样本损失
'mean'（默认）：返回损失均值
'sum'：返回损失总和
"""
```

注意：
- 目标是使锚点与正样本的距离小于锚点与负样本的距离，且差距至少为margin
- 如果条件满足，损失为0；否则损失为差距的绝对值
- 常用于训练孪生网络或三元组网络
- swap参数有助于处理困难负样本的情况

示例：

```python
loss = nn.TripletMarginLoss()
anchor = torch.randn(100, 128, requires_grad=True)
positive = torch.randn(100, 128, requires_grad=True)
negative = torch.randn(100, 128, requires_grad=True)
output = loss(anchor, positive, negative)
```

[TripletMarginLoss官方文档](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html)

## Triplet Margin with Distance Loss

### 数学表达式
$$
\mathcal{L}(a, p, n) = \frac{1}{n}\sum_{i=1}^{n} \max(0, \text{distance\_function}(a_i, p_i) - \text{distance\_function}(a_i, n_i) + \text{margin})
$$
其中：
- \( a \) 是锚点样本（anchor），形状为 (N, *)
- \( p \) 是正样本（positive），形状为 (N, *)
- \( n \) 是负样本（negative），形状为 (N, *)
- \( \text{distance\_function} \) 是自定义的距离函数
- \( n \) 是样本数量
- margin 是边界参数（默认为1.0）

### 适用场景
1. **度量学习**：需要自定义距离函数的度量学习
2. **人脸识别**：使用特定距离度量的人脸特征嵌入
3. **图像检索**：使用特定相似度度量的图像特征表示
4. **推荐系统**：使用特定距离度量的用户和物品嵌入

### 参数说明
PyTorch实现：`torch.nn.TripletMarginWithDistanceLoss`

```python
torch.nn.TripletMarginWithDistanceLoss(
    distance_function=None,  # 自定义距离函数
    margin=1.0,             # 边界参数
    reduction='mean'         # 'none'|'mean'|'sum'
)
"""
distance_function：
自定义距离函数，签名为(Tensor, Tensor) -> Tensor
如果为None，默认使用p=2的范数距离
默认值为None

margin：
指定期望的边界大小
默认值为1.0

reduction：
'none'：返回逐样本损失
'mean'（默认）：返回损失均值
'sum'：返回损失总和
"""
```

注意：
- 这是TripletMarginLoss的泛化版本，允许使用自定义距离函数
- 距离函数应接受两个相同形状的张量并返回一个标量张量
- 目标是使锚点与正样本的距离小于锚点与负样本的距离，且差距至少为margin
- 常用于需要特定距离度量的场景，如余弦距离、马氏距离等

示例：

```python
# 使用欧氏距离
loss = nn.TripletMarginWithDistanceLoss()
anchor = torch.randn(100, 128, requires_grad=True)
positive = torch.randn(100, 128, requires_grad=True)
negative = torch.randn(100, 128, requires_grad=True)
output = loss(anchor, positive, negative)

# 使用自定义距离函数（如余弦距离）
def cosine_distance(x1, x2):
    return 1 - F.cosine_similarity(x1, x2, dim=1)
    
loss = nn.TripletMarginWithDistanceLoss(distance_function=cosine_distance)
output = loss(anchor, positive, negative)
```

[TripletMarginWithDistanceLoss官方文档](https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginWithDistanceLoss.html)

## Connectionist Temporal Classification Loss (CTCLoss)

### 数学表达式
$$
\mathcal{L}(x, y, l) = -\log(p(l|x))
$$
其中：
- \( x \) 是输入序列（通常是RNN的输出），形状为 (T, N, C)
- \( y \) 是目标序列，形状为 (N, S) 或目标和长度的元组
- \( l \) 是输入长度和目标长度
- \( p(l|x) \) 是给定输入序列x时，目标序列l的条件概率

### 适用场景
1. **序列标注**：输入和输出长度不同的序列任务
2. **语音识别**：将音频信号转换为文本
3. **手写识别**：将手写图像转换为文本
4. **视频分析**：视频序列到标签序列的转换

### 参数说明
PyTorch实现：`torch.nn.CTCLoss`

```python
torch.nn.CTCLoss(
    blank=0,            # 空白标签索引
    reduction='mean',   # 'none'|'mean'|'sum'
    zero_infinity=False # 是否将无限损失设为零
)
"""
blank：
空白标签的索引
默认值为0

reduction：
'none'：返回逐样本损失
'mean'（默认）：返回损失均值
'sum'：返回损失总和

zero_infinity：
如果为True，将无限损失设为零（用于处理异常情况）
默认值为False
"""
```

注意：
- CTC允许输入和输出序列长度不同
- 输入通常需要经过log_softmax处理
- 需要提供输入长度和目标长度
- 空白标签用于对齐不同长度的序列
- 适用于不需要先验对齐信息的序列到序列任务

示例：

```python
T = 50      # 输入序列长度
C = 20      # 类别数（包括空白标签）
N = 16      # batch size
S = 30      # 目标序列长度
S_min = 10  # 最小目标长度

# 随机生成数据
input = torch.randn(T, N, C).log_softmax(2)
target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)

# 生成随机长度
input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)

loss = nn.CTCLoss()
output = loss(input, target, input_lengths, target_lengths)
```

[CTCLoss官方文档](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html)