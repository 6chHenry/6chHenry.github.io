# DAG-GNN: Directed Acyclic Graph - Graph Neural Network

## 概述

DAG-GNN是一种基于图神经网络的因果发现算法，由Zheng et al.在2018年提出。该方法将因果发现问题转化为连续优化问题，通过学习变量间的结构化关系来估计有向无环图(DAG)。

## 数学原理

### 基础概念

**图表示**: 有向无环图$G = (V, E)$，其中$V$是节点集合，$E$是有向边集合。邻接矩阵$W \in \mathbb{R}^{d \times d}$表示图结构，$W_{ij} \neq 0$表示存在从节点$i$到节点$j$的有向边。

**DAG约束**: 图必须是有向无环图，这等价于邻接矩阵$W$的指数矩阵$e^{W \circ W}$的对角线元素为零：
$$\text{tr}(e^{W \circ W}) = d$$

### 图神经网络架构

DAG-GNN使用图神经网络进行编码-解码：

**编码器(Encoder)**:
$$H^{(l+1)} = \sigma(W^{(l)}H^{(l)} + \sum_{j \in N(i)} \frac{1}{\sqrt{|N(i)||N(j)|}} H^{(l)}_j)$$

**解码器(Decoder)**:
$$\hat{X}_i = f(H_i; \theta)$$

### 目标函数

DAG-GNN的目标函数包含三个部分：

1. **重构损失**:
$$\mathcal{L}_{recon} = \frac{1}{n}\sum_{i=1}^n \|X_i - \hat{X}_i\|^2$$

2. **DAG约束**:
$$\mathcal{L}_{dag} = \text{tr}(e^{W \circ W}) - d$$

3. **稀疏性约束**:
$$\mathcal{L}_{sparse} = \|W\|_1$$

**总目标函数**:
$$\mathcal{L} = \mathcal{L}_{recon} + \lambda_1 \mathcal{L}_{dag} + \lambda_2 \mathcal{L}_{sparse}$$

### 优化策略

使用对数行列式展开(log-exp-sum)技巧来处理DAG约束：
$$\text{tr}(e^{W \circ W}) = \sum_{k=0}^{\infty} \frac{(W \circ W)^k}{k!}$$

实际计算中，使用截断近似：
$$\text{tr}(e^{W \circ W}) \approx \sum_{k=0}^{K} \frac{(W \circ W)^k}{k!}$$

## 算法流程

### 输入
- 观测数据矩阵 $X \in \mathbb{R}^{n \times d}$
- 图神经网络层数 $L$
- 正则化参数 $\lambda_1, \lambda_2$
- 学习率 $\eta$

### 初始化
- 随机初始化GNN参数 $\theta$
- 随机初始化邻接矩阵 $W$

### 迭代优化
```python
for epoch in range(max_epochs):
    # 前向传播
    H = gnn_encoder(X, W, theta)  # 编码
    X_hat = gnn_decoder(H, theta)  # 解码
    
    # 计算损失
    loss_recon = reconstruction_loss(X, X_hat)
    loss_dag = dag_constraint(W)
    loss_sparse = l1_norm(W)
    
    total_loss = loss_recon + lambda1 * loss_dag + lambda2 * loss_sparse
    
    # 反向传播
    total_loss.backward()
    optimizer.step()
    
    # 输出当前结果
    if epoch % log_interval == 0:
        print(f"Epoch {epoch}, Loss: {total_loss.item()}")
```

### 输出
- 学习到的邻接矩阵 $W^*$
- 估计的DAG结构 $G^* = (V, E^*)$

## 使用方法

### PyTorch实现示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DAG_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(DAG_GNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GNN层
        self.gnn_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) if i == 0 
            else nn.Linear(hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # 解码器
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
        # 邻接矩阵（可学习参数）
        self.W = nn.Parameter(torch.randn(input_dim, input_dim))
        
    def forward(self, X):
        batch_size, num_nodes, feature_dim = X.shape
        
        # GNN编码
        H = X
        for layer in self.gnn_layers:
            # 图卷积操作
            H_new = torch.zeros_like(H)
            for i in range(num_nodes):
                # 聚合邻居信息
                neighbor_sum = torch.zeros(batch_size, self.hidden_dim)
                for j in range(num_nodes):
                    if i != j:
                        weight = torch.sigmoid(self.W[i, j])
                        neighbor_sum += weight * H[:, j, :]
                
                # 自身信息 + 邻居信息
                H_new[:, i, :] = layer(H[:, i, :] + neighbor_sum)
            
            H = torch.relu(H_new)
        
        # 解码
        X_hat = self.decoder(H)
        
        return X_hat, self.W
    
    def dag_constraint(self):
        """计算DAG约束"""
        W = self.W
        # 使用截断指数函数近似
        exp_W = torch.matrix_exp(W * W)
        return torch.trace(exp_W) - self.input_dim

def train_dag_gnn(data, hidden_dim=64, num_layers=3, 
                 lambda1=0.1, lambda2=0.01, lr=0.001, 
                 epochs=1000, batch_size=32):
    """
    训练DAG-GNN模型
    """
    input_dim = data.shape[1]
    model = DAG_GNN(input_dim, hidden_dim, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 数据准备
    dataset = prepare_dataset(data, batch_size)
    
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_X in dataset:
            optimizer.zero_grad()
            
            # 前向传播
            X_hat, W = model(batch_X)
            
            # 计算损失
            recon_loss = nn.MSELoss()(X_hat, batch_X)
            dag_loss = model.dag_constraint()
            sparse_loss = torch.norm(W, p=1)
            
            total_loss_batch = recon_loss + lambda1 * dag_loss + lambda2 * sparse_loss
            
            # 反向传播
            total_loss_batch.backward()
            optimizer.step()
            
            total_loss += total_loss_batch.item()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss/len(dataset):.4f}")
    
    return model

def prepare_dataset(data, batch_size):
    """准备数据集"""
    # 将数据重塑为 (batch_size, num_nodes, feature_dim)
    # 这里简化处理，实际需要根据具体数据结构调整
    n_samples = data.shape[0]
    n_vars = data.shape[1]
    
    # 为每个样本创建图结构
    dataset = []
    for i in range(0, n_samples - batch_size + 1, batch_size):
        batch_data = data[i:i+batch_size]
        # 重塑为 (batch_size, num_vars, 1)
        batch_reshaped = batch_data.unsqueeze(-1)
        dataset.append(batch_reshaped)
    
    return dataset
```

### 参数选择

1. **网络结构参数**:
   - `hidden_dim`: 隐藏层维度，通常64-256
   - `num_layers`: GNN层数，通常2-4层

2. **正则化参数**:
   - `lambda1`: DAG约束强度，通常0.01-0.5
   - `lambda2`: 稀疏性约束强度，通常0.001-0.1

3. **训练参数**:
   - `lr`: 学习率，通常0.0001-0.01
   - `batch_size`: 批大小，根据数据量调整
   - `epochs`: 训练轮数，通常1000-10000

## 应用示例

### 线性结构学习

**真实结构**: $X_1 \rightarrow X_2 \rightarrow X_3$

**数据生成**:
```python
# 生成线性因果数据
n_samples = 1000
X1 = np.random.normal(0, 1, n_samples)
X2 = 0.8 * X1 + np.random.normal(0, 0.5, n_samples)
X3 = 0.6 * X2 + np.random.normal(0, 0.3, n_samples)

data = np.column_stack([X1, X2, X3])
```

**训练过程**:
```python
# 训练DAG-GNN
model = train_dag_gnn(data, hidden_dim=32, num_layers=2, 
                     lambda1=0.1, lambda2=0.01, lr=0.001)

# 获取学习到的邻接矩阵
W_learned = model.W.detach().numpy()
print("Learned adjacency matrix:")
print(W_learned)
```

### 非线性结构学习

**真实结构**: $X_1 \rightarrow X_2 \rightarrow X_3$，其中关系为非线性

**数据生成**:
```python
# 生成非线性因果数据
X1 = np.random.normal(0, 1, n_samples)
X2 = np.sin(X1) + np.random.normal(0, 0.3, n_samples)
X3 = X2**2 + np.random.normal(0, 0.2, n_samples)

data = np.column_stack([X1, X2, X3])
```

## 实际应用场景

1. **基因调控网络**: 发现基因间的非线性调控关系
2. **金融风险传导**: 识别金融风险在市场中的传导路径
3. **医疗诊断**: 分析疾病症状间的因果关系
4. **社交网络**: 研究社交影响和信息传播机制

## 算法优缺点

### 优点
- 能够处理非线性关系
- 端到端学习，不需要预先指定函数形式
- 可扩展到大规模问题
- 能够处理高维数据

### 缺点
- 计算复杂度较高，训练时间较长
- 需要调整多个超参数
- 可能陷入局部最优解
- 对初始值敏感

## 改进变体

1. **NOTEARS**: 使用连续优化的方法
2. **GraN-DAG**: 结合图神经网络和梯度下降
3. **DAGs with NO TEARS**: 改进的DAG约束处理方法
4. **CAM pruning**: 结合剪枝技术提高效率

## 参考资源

- Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous Optimization for Structure Learning. NeurIPS.
- Yu, Y., Chen, J., Gao, T., & Mo, S. (2019). DAG-GNN: DAG Structure Learning with Graph Neural Networks. ICLR.
- Bello, K., Aragam, B., & Ravikumar, P. (2022). GraN-DAG: A Continuous Optimization Approach for Learning DAGs from Data. AISTATS.