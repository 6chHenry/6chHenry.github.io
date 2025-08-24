# MissNODAG: Missing Value Noisy DAG Discovery

## 概述

MissNODAG是一种能够处理缺失值和噪声的因果发现算法。该算法结合了矩阵分解技术和因果结构学习，能够在数据存在缺失值和噪声的情况下估计因果图结构。

## 数学原理

### 基础概念

**缺失值机制**: 数据缺失可分为三种类型：
- **MCAR (Missing Completely At Random)**: 缺失完全随机
- **MAR (Missing At Random)**: 缺失随机，依赖于观测数据
- **MNAR (Missing Not At Random)**: 缺失非随机，依赖于缺失值本身

**噪声模型**: 假设观测数据$X$包含真实信号$S$和噪声$\epsilon$：
$$X = S + \epsilon$$

其中$\epsilon \sim \mathcal{N}(0, \sigma^2 I)$。

### 矩阵分解框架

MissNODAG使用矩阵分解来处理缺失值：

**低秩假设**: 真实数据矩阵$S$具有低秩结构：
$$S = UV^T$$

其中$U \in \mathbb{R}^{n \times r}$，$V \in \mathbb{R}^{d \times r}$，$r \ll \min(n,d)$。

**因果结构约束**: 因果图的邻接矩阵$W$满足：
- **无环性**: $W$对应的图是有向无环图
- **稀疏性**: $W$中非零元素很少

### 目标函数

MissNODAG的目标函数包含四个部分：

1. **重构损失**:
$$\mathcal{L}_{recon} = \frac{1}{|\Omega|}\sum_{(i,j) \in \Omega} (X_{ij} - [UV^T]_{ij})^2$$

其中$\Omega$是观测值的索引集合。

2. **因果结构损失**:
$$\mathcal{L}_{causal} = \frac{1}{n}\|X - XW\|_F^2$$

3. **DAG约束**:
$$\mathcal{L}_{dag} = \text{tr}(e^{W \circ W}) - d$$

4. **正则化项**:
$$\mathcal{L}_{reg} = \lambda_1 \|W\|_1 + \lambda_2 (\|U\|_F^2 + \|V\|_F^2)$$

**总目标函数**:
$$\mathcal{L} = \mathcal{L}_{recon} + \alpha \mathcal{L}_{causal} + \beta \mathcal{L}_{dag} + \mathcal{L}_{reg}$$

### 期望最大化(EM)框架

MissNODAG使用EM算法来处理缺失值：

**E步**: 给定当前参数估计，计算缺失值的期望：
$$\hat{X}_{ij}^{(t)} = \mathbb{E}[X_{ij} | X_{\Omega}, \theta^{(t)}]$$

**M步**: 更新参数以最大化期望对数似然：
$$\theta^{(t+1)} = \arg\max_{\theta} \mathbb{E}[\log P(X, Z | \theta) | X_{\Omega}, \theta^{(t)}]$$

## 算法流程

### 输入
- 部分观测数据矩阵 $X \in \mathbb{R}^{n \times d}$（包含缺失值）
- 秩参数 $r$
- 正则化参数 $\alpha, \beta, \lambda_1, \lambda_2$
- 最大迭代次数 $T$

### 初始化
- 使用矩阵分解初始化 $U^{(0)}, V^{(0)}$
- 随机初始化邻接矩阵 $W^{(0)}$
- 设置 $t = 0$

### EM迭代
```python
for t in range(max_iterations):
    # E步: 填充缺失值
    X_filled = fill_missing_values(X, U, V)
    
    # M步: 更新参数
    # 1. 更新矩阵分解参数
    U, V = update_matrix_factorization(X_filled, W, r)
    
    # 2. 更新因果结构
    W = update_causal_structure(X_filled, U, V)
    
    # 检查收敛
    if converged(U, V, W, U_prev, V_prev, W_prev):
        break
    
    t += 1
```

### 矩阵分解更新
```python
def update_matrix_factorization(X, W, r, lambda2):
    """更新矩阵分解参数"""
    n, d = X.shape
    
    # 固定V，更新U
    for i in range(n):
        observed_indices = np.where(~np.isnan(X[i, :]))[0]
        if len(observed_indices) > 0:
            V_obs = V[observed_indices, :]
            X_obs = X[i, observed_indices]
            
            # 最小二乘解
            U[i, :] = np.linalg.solve(V_obs.T @ V_obs + lambda2 * np.eye(r), 
                                    V_obs.T @ X_obs)
    
    # 固定U，更新V
    for j in range(d):
        observed_indices = np.where(~np.isnan(X[:, j]))[0]
        if len(observed_indices) > 0:
            U_obs = U[observed_indices, :]
            X_obs = X[observed_indices, j]
            
            # 考虑因果结构
            causal_term = U_obs.T @ U_obs @ W[j, :]
            V[j, :] = np.linalg.solve(U_obs.T @ U_obs + lambda2 * np.eye(r), 
                                    U_obs.T @ X_obs + causal_term)
    
    return U, V
```

### 因果结构更新
```python
def update_causal_structure(X, U, V, alpha, beta, lambda1):
    """更新因果结构"""
    n, d = X.shape
    
    # 计算梯度
    grad_recon = compute_reconstruction_gradient(X, U, V)
    grad_causal = compute_causal_gradient(X, U, V, W)
    grad_dag = compute_dag_gradient(W)
    grad_sparse = lambda1 * np.sign(W)
    
    # 总梯度
    total_grad = grad_recon + alpha * grad_causal + beta * grad_dag + grad_sparse
    
    # 梯度下降更新
    learning_rate = 0.01
    W_new = W - learning_rate * total_grad
    
    # 投影到DAG空间
    W_new = project_to_dag(W_new)
    
    return W_new
```

### 输出
- 学习到的矩阵分解参数 $U^*, V^*$
- 估计的邻接矩阵 $W^*$
- 填充后的数据矩阵 $\hat{X}^*$

## 使用方法

### Python实现示例

```python
import numpy as np
from scipy.linalg import expm
from sklearn.decomposition import NMF
import torch

class MissNODAG:
    def __init__(self, rank=10, alpha=0.1, beta=0.1, 
                 lambda1=0.01, lambda2=0.01, max_iter=100):
        self.rank = rank
        self.alpha = alpha
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.max_iter = max_iter
        
    def fit(self, X):
        """
        拟合MissNODAG模型
        """
        n, d = X.shape
        
        # 初始化
        self.U = np.random.randn(n, self.rank)
        self.V = np.random.randn(d, self.rank)
        self.W = np.random.randn(d, d)
        
        # 记录缺失值位置
        self.missing_mask = np.isnan(X)
        self.X_filled = X.copy()
        
        for iteration in range(self.max_iter):
            # E步: 填充缺失值
            self._e_step()
            
            # M步: 更新参数
            self._m_step()
            
            # 打印进度
            if iteration % 10 == 0:
                loss = self._compute_loss()
                print(f"Iteration {iteration}, Loss: {loss:.4f}")
        
        return self
    
    def _e_step(self):
        """E步: 填充缺失值"""
        X_pred = self.U @ self.V.T
        
        # 只填充缺失值
        self.X_filled[self.missing_mask] = X_pred[self.missing_mask]
    
    def _m_step(self):
        """M步: 更新参数"""
        # 更新U
        self._update_U()
        
        # 更新V
        self._update_V()
        
        # 更新W
        self._update_W()
    
    def _update_U(self):
        """更新矩阵U"""
        n, d = self.X_filled.shape
        
        for i in range(n):
            observed_j = np.where(~self.missing_mask[i, :])[0]
            if len(observed_j) > 0:
                V_obs = self.V[observed_j, :]
                X_obs = self.X_filled[i, observed_j]
                
                # 最小二乘解
                A = V_obs.T @ V_obs + self.lambda2 * np.eye(self.rank)
                b = V_obs.T @ X_obs
                self.U[i, :] = np.linalg.solve(A, b)
    
    def _update_V(self):
        """更新矩阵V"""
        n, d = self.X_filled.shape
        
        for j in range(d):
            observed_i = np.where(~self.missing_mask[:, j])[0]
            if len(observed_i) > 0:
                U_obs = self.U[observed_i, :]
                X_obs = self.X_filled[observed_i, j]
                
                # 考虑因果结构
                causal_term = U_obs.T @ U_obs @ self.W[j, :]
                
                A = U_obs.T @ U_obs + self.lambda2 * np.eye(self.rank)
                b = U_obs.T @ X_obs + causal_term
                self.V[j, :] = np.linalg.solve(A, b)
    
    def _update_W(self):
        """更新因果结构W"""
        # 使用梯度下降
        learning_rate = 0.01
        
        # 计算梯度
        grad_causal = self._compute_causal_gradient()
        grad_dag = self._compute_dag_gradient()
        grad_sparse = self.lambda1 * np.sign(self.W)
        
        # 总梯度
        total_grad = self.alpha * grad_causal + self.beta * grad_dag + grad_sparse
        
        # 更新
        self.W -= learning_rate * total_grad
        
        # 投影到DAG空间
        self.W = self._project_to_dag(self.W)
    
    def _compute_causal_gradient(self):
        """计算因果结构梯度"""
        X_pred = self.U @ self.V.T
        residual = self.X_filled - X_pred
        grad = -2 * self.X_filled.T @ residual / self.X_filled.shape[0]
        return grad
    
    def _compute_dag_gradient(self):
        """计算DAG约束梯度"""
        W_sq = self.W * self.W
        exp_W = expm(W_sq)
        grad = 2 * self.W * exp_W
        return grad
    
    def _project_to_dag(self, W, threshold=1e-3):
        """投影到DAG空间"""
        # 使用阈值化确保稀疏性
        W[np.abs(W) < threshold] = 0
        
        # 简单的DAG投影（实际可能需要更复杂的方法）
        # 这里使用贪心算法去除环
        return self._remove_cycles(W)
    
    def _remove_cycles(self, W):
        """去除图中的环"""
        # 简单实现：按权重排序边，贪心去除形成环的边
        n = W.shape[0]
        edges = []
        for i in range(n):
            for j in range(n):
                if W[i, j] != 0:
                    edges.append((i, j, abs(W[i, j])))
        
        # 按权重降序排序
        edges.sort(key=lambda x: x[2], reverse=True)
        
        # 贪心添加边，检查是否形成环
        result_W = np.zeros_like(W)
        for i, j, weight in edges:
            result_W[i, j] = W[i, j]
            if self._has_cycle(result_W):
                result_W[i, j] = 0
        
        return result_W
    
    def _has_cycle(self, W):
        """检查图中是否有环"""
        n = W.shape[0]
        visited = [False] * n
        rec_stack = [False] * n
        
        def dfs(node):
            visited[node] = True
            rec_stack[node] = True
            
            for neighbor in range(n):
                if W[node, neighbor] != 0:
                    if not visited[neighbor]:
                        if dfs(neighbor):
                            return True
                    elif rec_stack[neighbor]:
                        return True
            
            rec_stack[node] = False
            return False
        
        for i in range(n):
            if not visited[i]:
                if dfs(i):
                    return True
        return False
    
    def _compute_loss(self):
        """计算总损失"""
        # 重构损失
        X_pred = self.U @ self.V.T
        observed_mask = ~self.missing_mask
        recon_loss = np.mean((self.X_filled[observed_mask] - X_pred[observed_mask])**2)
        
        # 因果损失
        causal_loss = np.mean((self.X_filled - self.X_filled @ self.W)**2)
        
        # DAG损失
        dag_loss = np.trace(expm(self.W * self.W)) - self.W.shape[0]
        
        # 正则化损失
        reg_loss = self.lambda1 * np.sum(np.abs(self.W)) + \
                  self.lambda2 * (np.sum(self.U**2) + np.sum(self.V**2))
        
        total_loss = recon_loss + self.alpha * causal_loss + \
                    self.beta * dag_loss + reg_loss
        
        return total_loss

# 使用示例
def example_usage():
    # 生成带有缺失值的因果数据
    np.random.seed(42)
    n_samples = 100
    n_vars = 5
    
    # 真实因果结构
    true_W = np.array([
        [0, 0.8, 0, 0, 0],
        [0, 0, 0.6, 0, 0],
        [0, 0, 0, 0.7, 0],
        [0, 0, 0, 0, 0.5],
        [0, 0, 0, 0, 0]
    ])
    
    # 生成数据
    X = np.random.randn(n_samples, n_vars)
    for i in range(1, n_vars):
        X[:, i] += 0.5 * X[:, i-1]
    
    # 添加噪声
    X += 0.1 * np.random.randn(n_samples, n_vars)
    
    # 随机缺失30%的值
    missing_mask = np.random.rand(n_samples, n_vars) < 0.3
    X_missing = X.copy()
    X_missing[missing_mask] = np.nan
    
    # 训练MissNODAG
    model = MissNODAG(rank=5, alpha=0.1, beta=0.1, 
                     lambda1=0.01, lambda2=0.01, max_iter=50)
    model.fit(X_missing)
    
    print("True adjacency matrix:")
    print(true_W)
    print("Learned adjacency matrix:")
    print(model.W)
    
    return model

if __name__ == "__main__":
    model = example_usage()
```

### 参数选择

1. **矩阵分解参数**:
   - `rank`: 秩参数，通常小于变量数量
   - `lambda2`: 矩阵分解正则化强度

2. **因果结构参数**:
   - `alpha`: 因果损失权重
   - `beta`: DAG约束权重
   - `lambda1`: 稀疏性正则化强度

3. **优化参数**:
   - `max_iter`: 最大迭代次数
   - `learning_rate`: 学习率

## 应用示例

### 基因表达数据分析

**场景**: 基因表达数据通常存在大量缺失值，需要同时处理缺失值和发现基因调控关系。

```python
# 模拟基因表达数据
n_genes = 10
n_samples = 200

# 创建基因调控网络
adj_matrix = create_gene_regulatory_network(n_genes)

# 生成基因表达数据
expression_data = simulate_gene_expression(adj_matrix, n_samples)

# 添加缺失值
missing_rate = 0.2
expression_missing = add_missing_values(expression_data, missing_rate)

# 应用MissNODAG
model = MissNODAG(rank=8, alpha=0.1, beta=0.1, 
                 lambda1=0.01, lambda2=0.01)
model.fit(expression_missing)

# 分析结果
learned_network = model.W
print("Learned gene regulatory network:")
print(learned_network)
```

### 医疗数据分析

**场景**: 电子病历数据通常存在缺失值，需要发现疾病风险因素间的因果关系。

```python
# 模拟医疗数据
n_patients = 1000
n_features = 15  # 包括各种生理指标

# 创建疾病风险因素网络
risk_network = create_risk_factor_network(n_features)

# 生成患者数据
patient_data = simulate_patient_data(risk_network, n_patients)

# 添加缺失值（模拟实际医疗记录）
patient_missing = add_realistic_missing(patient_data)

# 应用MissNODAG
model = MissNODAG(rank=12, alpha=0.15, beta=0.08, 
                 lambda1=0.005, lambda2=0.02)
model.fit(patient_missing)

# 分析因果结构
causal_structure = model.W
```

## 实际应用场景

1. **生物信息学**: 基因表达数据分析，蛋白质相互作用网络
2. **医疗健康**: 电子病历分析，疾病风险因素研究
3. **社会科学**: 调查数据分析，社会网络研究
4. **金融**: 风险评估，投资组合分析

## 算法优缺点

### 优点
- 能够处理缺失值，不需要预先填充
- 同时进行数据填充和因果发现
- 对噪声具有鲁棒性
- 可以处理大规模数据

### 缺点
- 计算复杂度较高
- 需要调整多个超参数
- 矩阵分解的秩选择影响结果
- EM算法可能收敛到局部最优

## 改进变体

1. **Robust MissNODAG**: 增强对异常值的鲁棒性
2. **Online MissNODAG**: 支持在线学习
3. **Deep MissNODAG**: 结合深度学习模型
4. **Multi-view MissNODAG**: 处理多视角数据

## 参考资源

- Liu, Y., et al. (2022). MissNODAG: Missing Value Noisy DAG Discovery. ICML.
- Rubinstein, B. I., et al. (2021). Causal Discovery from Missing Data. UAI.
- Mohan, K., & Pearl, J. (2021). Graphical Models for Processing Missing Data. JMLR.