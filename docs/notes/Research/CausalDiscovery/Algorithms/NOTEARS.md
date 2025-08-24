# NOTEARS: Nonlinear Structural Equations with Alternative Regularization and Sparsity

## 概述

NOTEARS是一种基于连续优化的因果发现算法，由Zheng et al.在2018年提出。该算法将因果发现问题转化为连续优化问题，通过引入DAG约束来学习有向无环图结构。

## 数学原理

### 基础概念

**结构方程模型(SEM)**: 假设每个变量可以表示为其父节点的函数加上噪声：
$$X_j = f_j(X_{pa(j)}) + \epsilon_j$$

其中$\epsilon_j$是独立噪声项，$pa(j)$是节点$j$的父节点集合。

**邻接矩阵**: $W \in \mathbb{R}^{d \times d}$表示图结构，$W_{ij} \neq 0$表示存在从节点$i$到节点$j$的有向边。

### DAG约束

NOTEARS的核心创新在于将DAG约束转化为连续可微的约束：

**矩阵指数约束**: 矩阵$W$表示有向无环图当且仅当：
$$\text{tr}(e^{W \circ W}) = d$$

其中$\circ$表示Hadamard积（逐元素乘法），$e^A$是矩阵指数函数。

**矩阵指数展开**:
$$e^A = \sum_{k=0}^{\infty} \frac{A^k}{k!}$$

实际计算中使用截断近似：
$$e^A \approx \sum_{k=0}^{K} \frac{A^k}{k!}$$

### 目标函数

NOTEARS的目标函数包含三个部分：

1. **重构损失**:
$$\mathcal{L}_{recon} = \frac{1}{2n}\sum_{i=1}^n \|X_i - X_i W\|_2^2$$

2. **DAG约束**:
$$h(W) = \text{tr}(e^{W \circ W}) - d$$

3. **稀疏性约束**:
$$\mathcal{L}_{sparse} = \lambda \|W\|_1$$

**总目标函数**:
$$\min_W \mathcal{L}_{recon}(W) + \lambda \|W\|_1$$
$$\text{s.t. } h(W) = 0$$

### 非线性扩展

NOTEARS可以扩展到非线性情况：

**MLP结构方程**:
$$X_j = f_j(X_{pa(j)}; \theta_j) + \epsilon_j$$

其中$f_j$是多层感知机，$\theta_j$是其参数。

**等价表示**: 非线性结构方程可以表示为：
$$X = f(X; \theta) + \epsilon$$

其中$f$是向量值函数。

## 算法流程

### 输入
- 观测数据矩阵 $X \in \mathbb{R}^{n \times d}$
- 正则化参数 $\lambda$
- 学习率 $\eta$
- 最大迭代次数 $T$

### 初始化
- 随机初始化邻接矩阵 $W^{(0)}$
- 设置 $t = 0$

### 连续优化
```python
for t in range(max_iterations):
    # 计算梯度
    grad_recon = compute_reconstruction_gradient(X, W)
    grad_dag = compute_dag_gradient(W)
    grad_sparse = lambda * np.sign(W)
    
    # 总梯度
    total_grad = grad_recon + grad_sparse
    
    # 投影梯度下降
    W_new = W - learning_rate * total_grad
    
    # 投影到DAG空间
    W_new = project_to_dag(W_new)
    
    # 检查收敛
    if converged(W, W_new):
        break
    
    W = W_new
    t += 1
```

### 投影算法

NOTEARS使用增广拉格朗日方法处理约束：

**增广拉格朗日函数**:
$$\mathcal{L}_\rho(W, \alpha) = \mathcal{L}_{recon}(W) + \lambda \|W\|_1 + \alpha h(W) + \frac{\rho}{2} h(W)^2$$

**更新规则**:
1. **W更新**: 
   $$W^{k+1} = \arg\min_W \mathcal{L}_\rho(W, \alpha^k)$$

2. **拉格朗日乘子更新**:
   $$\alpha^{k+1} = \alpha^k + \rho h(W^{k+1})$$

3. **惩罚参数更新**:
   $$\rho^{k+1} = \min(\rho_{max}, \gamma \rho^k)$$

### 非线性NOTEARS流程

```python
def nonlinear_notears(X, hidden_dims, lambda1=0.01, rho=1.0, alpha=0.0):
    """
    非线性NOTEARS算法
    """
    n, d = X.shape
    
    # 初始化MLP参数
    models = [MLP(d, hidden_dims, 1) for _ in range(d)]
    
    # 初始化邻接矩阵
    W = np.zeros((d, d))
    
    for iteration in range(max_iterations):
        # 前向传播
        X_pred = forward_pass(X, models, W)
        
        # 计算损失
        loss_recon = reconstruction_loss(X, X_pred)
        loss_dag = dag_constraint(W)
        loss_sparse = lambda1 * np.sum(np.abs(W))
        
        total_loss = loss_recon + alpha * loss_dag + 0.5 * rho * loss_dag**2 + loss_sparse
        
        # 反向传播
        total_loss.backward()
        
        # 更新参数
        update_parameters(models, W, learning_rate)
        
        # 更新拉格朗日乘子
        alpha += rho * loss_dag.item()
        
        # 更新惩罚参数
        if iteration % update_freq == 0:
            rho = min(rho * 1.1, rho_max)
    
    return models, W
```

## 使用方法

### PyTorch实现示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.linalg import expm

class NOTEARS(nn.Module):
    def __init__(self, input_dim, lambda1=0.01):
        super(NOTEARS, self).__init__()
        self.input_dim = input_dim
        self.lambda1 = lambda1
        
        # 可学习的邻接矩阵
        self.W = nn.Parameter(torch.randn(input_dim, input_dim))
        
    def forward(self, X):
        # 线性结构方程: X = XW + noise
        return X @ self.W
    
    def dag_constraint(self):
        """计算DAG约束"""
        W = self.W
        # 使用矩阵指数
        exp_W = torch.matrix_exp(W * W)
        return torch.trace(exp_W) - self.input_dim
    
    def loss(self, X, X_pred):
        """计算总损失"""
        # 重构损失
        recon_loss = 0.5 * torch.mean((X - X_pred)**2)
        
        # DAG约束
        dag_loss = self.dag_constraint()
        
        # 稀疏性约束
        sparse_loss = self.lambda1 * torch.norm(self.W, p=1)
        
        return recon_loss + dag_loss + sparse_loss

class NonlinearNOTEARS(nn.Module):
    def __init__(self, input_dim, hidden_dims=[16, 8], lambda1=0.01):
        super(NonlinearNOTEARS, self).__init__()
        self.input_dim = input_dim
        self.lambda1 = lambda1
        
        # 为每个变量创建MLP
        self.models = nn.ModuleList([
            self._create_mlp(input_dim, hidden_dims) 
            for _ in range(input_dim)
        ])
        
        # 邻接矩阵（用于稀疏性约束）
        self.W = nn.Parameter(torch.randn(input_dim, input_dim))
        
    def _create_mlp(self, input_dim, hidden_dims):
        """创建MLP模型"""
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        return nn.Sequential(*layers)
    
    def forward(self, X):
        """非线性前向传播"""
        batch_size, n_vars = X.shape
        X_pred = torch.zeros_like(X)
        
        for j in range(n_vars):
            # 使用第j个MLP预测第j个变量
            # 输入是所有变量，但通过邻接矩阵进行稀疏化
            input_j = X * torch.sigmoid(self.W[:, j])  # 稀疏化输入
            X_pred[:, j] = self.models[j](input_j).squeeze()
        
        return X_pred
    
    def dag_constraint(self):
        """计算DAG约束"""
        W = self.W
        exp_W = torch.matrix_exp(W * W)
        return torch.trace(exp_W) - self.input_dim
    
    def loss(self, X, X_pred):
        """计算总损失"""
        # 重构损失
        recon_loss = 0.5 * torch.mean((X - X_pred)**2)
        
        # DAG约束
        dag_loss = self.dag_constraint()
        
        # 稀疏性约束
        sparse_loss = self.lambda1 * torch.norm(self.W, p=1)
        
        return recon_loss + dag_loss + sparse_loss

def train_notears(data, model_type='linear', lambda1=0.01, 
                  lr=0.01, epochs=1000):
    """
    训练NOTEARS模型
    """
    input_dim = data.shape[1]
    
    # 创建模型
    if model_type == 'linear':
        model = NOTEARS(input_dim, lambda1)
    else:
        model = NonlinearNOTEARS(input_dim, lambda1=lambda1)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 转换为PyTorch张量
    X_tensor = torch.FloatTensor(data)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 前向传播
        X_pred = model(X_tensor)
        
        # 计算损失
        loss = model.loss(X_tensor, X_pred)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 打印进度
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return model

# 使用示例
def example_linear_notears():
    """线性NOTEARS示例"""
    # 生成线性因果数据
    np.random.seed(42)
    n_samples = 1000
    n_vars = 5
    
    # 真实邻接矩阵
    true_W = np.array([
        [0.0, 0.8, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.6, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.7, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.5],
        [0.0, 0.0, 0.0, 0.0, 0.0]
    ])
    
    # 生成数据
    X = np.random.randn(n_samples, n_vars)
    for i in range(n_samples):
        X[i] = X[i] @ true_W + 0.1 * np.random.randn(n_vars)
    
    # 训练模型
    model = train_notears(X, model_type='linear', lambda1=0.01, 
                        lr=0.01, epochs=1000)
    
    # 获取结果
    learned_W = model.W.detach().numpy()
    
    print("True adjacency matrix:")
    print(true_W)
    print("Learned adjacency matrix:")
    print(learned_W)
    
    return model

def example_nonlinear_notears():
    """非线性NOTEARS示例"""
    # 生成非线性因果数据
    np.random.seed(42)
    n_samples = 1000
    n_vars = 4
    
    # 生成数据
    X = np.random.randn(n_samples, n_vars)
    
    # 非线性关系
    X[:, 1] = np.sin(X[:, 0]) + 0.1 * np.random.randn(n_samples)
    X[:, 2] = X[:, 1]**2 + 0.1 * np.random.randn(n_samples)
    X[:, 3] = np.tanh(X[:, 2]) + 0.1 * np.random.randn(n_samples)
    
    # 训练模型
    model = train_notears(X, model_type='nonlinear', lambda1=0.01, 
                        lr=0.001, epochs=2000)
    
    # 获取结果
    learned_W = model.W.detach().numpy()
    
    print("Learned adjacency matrix (nonlinear):")
    print(learned_W)
    
    return model

if __name__ == "__main__":
    print("Linear NOTEARS Example:")
    linear_model = example_linear_notears()
    
    print("\nNonlinear NOTEARS Example:")
    nonlinear_model = example_nonlinear_notears()
```

### 参数选择

1. **正则化参数**:
   - `lambda1`: 稀疏性约束强度，通常0.001-0.1
   - `rho`: 增广拉格朗日惩罚参数，通常1.0-10.0
   - `alpha`: 拉格朗日乘子，初始化为0

2. **优化参数**:
   - `learning_rate`: 学习率，通常0.001-0.01
   - `epochs`: 训练轮数，通常1000-10000
   - `rho_max`: 最大惩罚参数，通常1e16

3. **网络结构参数**:
   - `hidden_dims`: 隐藏层维度，通常[16, 8]或[32, 16]

## 应用示例

### 线性因果发现

**场景**: 发现线性因果结构

```python
# 生成线性数据
n_samples = 2000
n_vars = 6

# 创建稀疏邻接矩阵
W_true = np.zeros((n_vars, n_vars))
W_true[0, 1] = 0.8
W_true[1, 2] = 0.6
W_true[2, 3] = 0.7
W_true[3, 4] = 0.5
W_true[4, 5] = 0.4

# 生成数据
X = np.random.randn(n_samples, n_vars)
for i in range(n_samples):
    X[i] = X[i] @ W_true + 0.1 * np.random.randn(n_vars)

# 训练NOTEARS
model = train_notears(X, model_type='linear', lambda1=0.01, 
                    lr=0.01, epochs=2000)

# 评估结果
W_learned = model.W.detach().numpy()
print("MSE between true and learned W:", 
      np.mean((W_true - W_learned)**2))
```

### 非线性因果发现

**场景**: 发现非线性因果结构

```python
# 生成非线性数据
n_samples = 2000
n_vars = 5

# 生成基础变量
X = np.random.randn(n_samples, n_vars)

# 非线性关系
X[:, 1] = 0.7 * np.sin(X[:, 0]) + 0.1 * np.random.randn(n_samples)
X[:, 2] = 0.5 * X[:, 1]**2 + 0.1 * np.random.randn(n_samples)
X[:, 3] = 0.6 * np.tanh(X[:, 2]) + 0.1 * np.random.randn(n_samples)
X[:, 4] = 0.4 * np.exp(-X[:, 3]**2) + 0.1 * np.random.randn(n_samples)

# 训练非线性NOTEARS
model = train_notears(X, model_type='nonlinear', lambda1=0.01, 
                    lr=0.001, epochs=3000)

# 分析结果
W_learned = model.W.detach().numpy()
print("Learned adjacency matrix (nonlinear):")
print(W_learned)
```

## 实际应用场景

1. **基因调控网络**: 发现基因间的线性/非线性调控关系
2. **经济系统**: 识别经济变量间的因果关系
3. **生物医学**: 分析生物标志物间的因果结构
4. **工程系统**: 发现系统组件间的因果关系

## 算法优缺点

### 优点
- 理论基础扎实，基于连续优化
- 可扩展到大规模问题
- 支持线性和非线性关系
- 计算效率高

### 缺点
- 仅适用于无隐变量的情况
- 对噪声敏感
- 可能陷入局部最优解
- 需要调整多个超参数

## 改进变体

1. **DAGs with NO TEARS**: 原始NOTEARS算法
2. **NOTEARS-MLP**: 使用MLP的非线性扩展
3. **NOTEARS-SL**: 稀疏线性NOTEARS
4. **NOTEARS-V**: 处理混合变量的NOTEARS

## 参考资源

- Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous Optimization for Structure Learning. NeurIPS.
- Zheng, X., et al. (2020). Learning DAGs with Continuous Optimization. AISTATS.
- Bello, K., Aragam, B., & Ravikumar, P. (2022). Learning Sparse Nonlinear DAGs. ICML.