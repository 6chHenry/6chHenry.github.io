---
title: "🧭 TensorBoard 全面学习路线"
updatedAt: "2025-08-06T08:05:49+08:00"
tags:
  - "Python-Tutorial"
draft: false
legacyPath: "/notes/Python-Tutorial/Tensorboard/"
---
# 🧭 TensorBoard 全面学习路线

我们会分模块进行讲解，每个模块包括：

* 📘 **功能介绍**
* 🧪 **代码示例**
* 🔍 **TensorBoard 页面展示**
* 🧠 **典型应用场景**

---

## ✅ 第1课：基本使用（标量指标的记录）

### 📘 功能介绍：

最基础的功能是记录标量，比如训练过程中的 `loss`、`accuracy`，方便你查看训练是否收敛。

### 🧪 代码示例：

以 PyTorch 为例：

```python
from torch.utils.tensorboard import SummaryWriter
import torch
import random

writer = SummaryWriter("runs/basic_scalars")

for epoch in range(100):
    loss = random.random()
    accuracy = 1 - loss
    writer.add_scalar("Loss/train", loss, epoch)
    writer.add_scalar("Accuracy/train", accuracy, epoch)

writer.close()
```

#### 🔍 TensorBoard 页面展示：

* 打开 Scalar 面板，你会看到两个图线：

  * Loss/train
  * Accuracy/train

#### 🧠 应用场景：

* 跟踪模型训练效果（Loss/Accuracy/学习率）
* 比较不同实验（多个 runs）

---

## ✅ 第2课：记录计算图（模型结构可视化）

### 📘 功能介绍：

你可以将模型的前向传播图可视化，便于理解结构。

### 🧪 代码示例：

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
    def forward(self, x):
        return self.net(x)

model = MyModel()
dummy_input = torch.randn(1, 10)

writer = SummaryWriter("runs/graph_example")
writer.add_graph(model, dummy_input)
writer.close()
```

### 🔍 TensorBoard 页面展示：

* 打开 Graph 面板，会看到完整的模型图。

### 🧠 应用场景：

* 快速验证模型结构
* 给团队成员展示模型构成

---

## ✅ 第3课：记录模型参数（权重分布直方图）

### 📘 功能介绍：

你可以记录每一层的参数分布，查看训练过程中参数是否合理。

### 🧪 代码示例：

```python
model = nn.Linear(10, 2)
writer = SummaryWriter("runs/param_hist")

for epoch in range(100):
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)

writer.close()
```

### 🔍 TensorBoard 页面展示：

* 打开 **Histograms** 面板，可以看到随训练过程模型参数分布的演变。

### 🧠 应用场景：

* 监控训练是否出现梯度爆炸/消失
* 观察模型收敛趋势

---

## ✅ 第4课：记录图像（训练样本、可视化结果）

### 📘 功能介绍：

将图像输入、预测结果可视化，可以帮助你 debug 或展示效果。

### 🧪 代码示例：

```python
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

images, labels = next(iter(dataloader))
img_grid = torchvision.utils.make_grid(images)

writer = SummaryWriter("runs/images")
writer.add_image("mnist_images", img_grid)
writer.close()
```

### 🔍 TensorBoard 页面展示：

* 打开 **Images** 面板，显示采样图像（如 MNIST）

### 🧠 应用场景：

* 验证数据预处理结果
* 对比预测结果与真实标签（语义分割、生成图像）

---

## ✅ 第5课：高阶功能（投影嵌入 Embeddings）

### 📘 功能介绍：

你可以用 TensorBoard 的 Projector 查看高维嵌入的投影，比如 Word Embedding、Image Embedding。

### 🧪 代码示例：

```python
import numpy as np
from torch.utils.tensorboard import SummaryWriter

features = torch.randn(100, 64)  # 100个样本，64维特征
labels = [str(i % 10) for i in range(100)]  # 假设是0-9的标签

writer = SummaryWriter("runs/embeddings")
writer.add_embedding(features, metadata=labels)
writer.close()
```

### 🔍 TensorBoard 页面展示：

* 打开 **Projector** 面板，可以查看2D/3D嵌入投影图。

### 🧠 应用场景：

* 可视化分类器输出、词向量分布
* 检查样本聚类、模型分离能力

---

## ✅ 第6课：多实验对比（命名不同 runs）

### 📘 功能介绍：

你可以记录多个实验到不同子目录，然后同时在 TensorBoard 中比较它们。

### 🧪 代码示例：

```python
writer1 = SummaryWriter("runs/exp1")
writer2 = SummaryWriter("runs/exp2")

for epoch in range(100):
    writer1.add_scalar("Loss/train", 1 / (epoch+1), epoch)
    writer2.add_scalar("Loss/train", 0.8 / (epoch+1), epoch)

writer1.close()
writer2.close()
```

然后：

```python
from tensorboard import program

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir=runs'])
tb.launch()
```

### 🔍 TensorBoard 页面展示：

* 你会看到 Loss/train 有两条线，一条来自 `exp1`，一条来自 `exp2`

### 🧠 应用场景：

* 对比不同超参数
* 对比不同模型结构
* 汇总实验结果

---

## 🚀 实用技巧 & 建议

| 技巧                                               | 说明                               |
| ------------------------------------------------ | -------------------------------- |
| 使用 `runs/实验名` 做目录管理                              | 方便后期对比多个实验                       |
| 写日志的时候尽量用语义清晰的 tag，如 `train/loss`、`val/accuracy` | 方便在 TensorBoard 中分组              |
| 每次训练开始前清理旧日志（`shutil.rmtree('runs')`）            | 避免旧日志混淆                          |
| TensorBoard 日志记录不应太频繁                            | 每几个 step 或每个 epoch 记录一次即可，避免文件膨胀 |

---
