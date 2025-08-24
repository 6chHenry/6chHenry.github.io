# Peter-Clark (PC) Algorithm

## 概述

PC算法是一种基于约束的因果发现算法，通过条件独立性检验来学习因果结构。该算法由Spirtes和Glymour提出，能够从观测数据中恢复因果图的骨架结构，并识别部分方向。

## 数学原理

### 基础概念

**因果Markov条件**: 给定因果图G中节点X的父节点PA(X)，X与其所有非后代节点条件独立：
$$X \perp\!\!\!\perp ND(X) | PA(X)$$

**忠诚性假设**: 数据中所有的条件独立性关系都由因果图的d-分离模式决定。

**d-分离**: 对于有向无环图G中的三个节点集X, Y, Z，如果所有从X到Y的路径都被Z中的节点阻断，则称X和Y被Z d-分离。

### 条件独立性检验

PC算法的核心是条件独立性检验，常用方法包括：

1. **偏相关检验**: 
   $$\rho_{XY|Z} = \frac{\rho_{XY} - \rho_{XZ}\rho_{YZ}}{\sqrt{(1-\rho_{XZ}^2)(1-\rho_{YZ}^2)}}$$

2. **卡方检验**: 对于离散变量
   $$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

3. **G-检验**: 基于信息论的独立性检验

### 算法复杂度

PC算法的时间复杂度为$O(p^2 \cdot 2^{p-2})$，其中p是变量数量。在实际应用中，对于稀疏图，复杂度会显著降低。

## 算法流程

### 阶段 0: 数据准备

**输入**:
- 观测数据矩阵 $X \in \mathbb{R}^{n \times p}$
- 变量集 $V = \{X_1, X_2, ..., X_p\}$
- 显著性水平 $\alpha$ (通常为0.05)

**输出**:
- 条件独立性表 $I$，其中 $I(i,j|S) = 1$ 表示 $X_i \perp\!\!\!\perp X_j | S$

### 阶段 1: 骨架学习

#### 1.1 初始化
- 创建完全无向图 $G_0$，所有节点两两相连

#### 1.2 迭代删边
```python
for k = 0, 1, 2, ..., p-2:
    for 每条边 (X_i, X_j) 仍在图中:
        for 每个大小为k的子集 S ⊆ adj(G, X_i)\{X_j}:
            if 检验 X_i ⊥ X_j | S 成立:
                删除边 X_i–X_j
                记录 SepSet(i,j) = S
                break  # 找到分离集即可停止
```

### 阶段 2: v-结构识别

遍历所有无向三元组 $X_i - X_k - X_j$ 且 $X_i$ 与 $X_j$ 不相邻：
```python
for 所有这样的三元组:
    if X_k ∉ SepSet(i,j):
        定向为 X_i → X_k ← X_j  # 标记为对撞节点
```

### 阶段 3: 方向传播

应用Meek规则进行方向传播：

**R1**: 若存在 $X \rightarrow Y - Z$ 且 $X$ 与 $Z$ 不相邻 $\Rightarrow Y \rightarrow Z$

**R2**: 若存在 $X \rightarrow Y \rightarrow Z$ 且 $X - Z$ $\Rightarrow X \rightarrow Z$

**R3**: 若存在 $X - Y - Z$ 且存在 $X \rightarrow W \leftarrow Z$ 且 $W$ 与 $Y$ 不相邻 $\Rightarrow X \rightarrow Y \leftarrow Z$

**R4**: 若存在 $X - Y - Z$ 且存在 $X \rightarrow W \rightarrow Z$ 且 $W$ 与 $Y$ 不相邻 $\Rightarrow X \rightarrow Y \rightarrow Z$

循环应用直到没有新箭头可添加，最终输出CPDAG。

## 使用方法

### Python实现示例

```python
import numpy as np
from scipy import stats
from itertools import combinations
import networkx as nx

def conditional_independence_test(X, Y, Z=None, data=None, alpha=0.05):
    """
    条件独立性检验
    """
    if Z is None or len(Z) == 0:
        # 无条件独立性检验
        corr, p_value = stats.pearsonr(data[:, X], data[:, Y])
        return p_value > alpha
    else:
        # 条件独立性检验 - 偏相关
        try:
            from sklearn.linear_model import LinearRegression
            # 对X和Y分别回归Z
            lr_X = LinearRegression()
            lr_Y = LinearRegression()
            
            lr_X.fit(data[:, Z], data[:, X])
            lr_Y.fit(data[:, Z], data[:, Y])
            
            res_X = data[:, X] - lr_X.predict(data[:, Z])
            res_Y = data[:, Y] - lr_Y.predict(data[:, Z])
            
            corr, p_value = stats.pearsonr(res_X, res_Y)
            return p_value > alpha
        except:
            return False

def pc_algorithm(data, alpha=0.05):
    """
    PC算法实现
    """
    n_vars = data.shape[1]
    G = nx.complete_graph(n_vars)
    sep_set = {(i, j): set() for i in range(n_vars) for j in range(n_vars)}
    
    # 阶段1: 骨架学习
    for k in range(n_vars):
        edges_to_check = list(G.edges())
        removed_edges = []
        
        for (i, j) in edges_to_check:
            if (i, j) not in G.edges():
                continue
                
            neighbors_i = set(G.neighbors(i)) - {j}
            
            if len(neighbors_i) >= k:
                for S in combinations(neighbors_i, k):
                    if conditional_independence_test(i, j, list(S), data, alpha):
                        G.remove_edge(i, j)
                        sep_set[(i, j)] = set(S)
                        sep_set[(j, i)] = set(S)
                        removed_edges.append((i, j))
                        break
        
        if not removed_edges:
            break
    
    # 阶段2: v-结构识别
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            if not G.has_edge(i, j):
                common_neighbors = set(G.neighbors(i)) & set(G.neighbors(j))
                for k in common_neighbors:
                    if k not in sep_set[(i, j)]:
                        G.remove_edge(i, k)
                        G.remove_edge(j, k)
                        G.add_edge(i, k, directed=True)
                        G.add_edge(j, k, directed=True)
    
    return G, sep_set
```

### 参数选择

1. **显著性水平α**: 通常选择0.05，但可根据数据噪声调整
2. **独立性检验方法**: 
   - 连续变量: 偏相关检验、G-检验
   - 离散变量: 卡方检验、G-检验
3. **最大条件集大小**: 可设置上限以提高效率

## 应用示例

### 3变量因果结构

**真实结构**: $A \rightarrow C \leftarrow B$

**独立性表**:
- $A \perp\!\!\!\perp B | \emptyset$: 不独立
- $A \perp\!\!\!\perp C | \emptyset$: 不独立  
- $A \perp\!\!\!\perp C | \{B\}$: 独立 → 删除A–C，SepSet(A,C)={B}
- $B \perp\!\!\!\perp C | \emptyset$: 不独立
- $B \perp\!\!\!\perp C | \{A\}$: 独立 → 删除B–C，SepSet(B,C)={A}

**算法过程**:
1. 骨架: A–B (因为A⊥B|∅不成立)
2. v-结构: 对三元组A–C–B，SepSet(A,B)=∅，而C∉SepSet(A,B)，定向为A→C←B
3. 方向传播: 无其他边可加方向

**最终CPDAG**: A→C←B, A–B (无向)

### 实际应用场景

1. **基因网络分析**: 发现基因间的调控关系
2. **经济学**: 识别经济变量间的因果关系
3. **医学**: 分析疾病风险因素间的因果结构
4. **社会科学**: 研究社会现象间的因果机制

## 算法优缺点

### 优点
- 理论基础扎实，基于因果图的d-分离理论
- 不需要预设函数形式
- 能够处理混合类型变量
- 输出为CPDAG，表示Markov等价类

### 缺点
- 计算复杂度高，变量多时难以应用
- 依赖条件独立性检验的准确性
- 忠诚性假设在实际中可能不成立
- 只能识别Markov等价类，无法确定所有边的方向

## 改进变体

1. **FCI算法**: 处理隐变量和选择偏差
2. **PC-Stable**: 稳定版本的PC算法
3. **MPC算法**: 改进的PC算法，提高效率
4. **Fast Causal Inference (FCI)**: 扩展到隐变量情况

## 参考资源

- [【因果系列】PC 算法 —— 一种基于约束的因果发现算法](https://zhuanlan.zhihu.com/p/452724126)
- [PC 算法 - 贝叶斯网络与其结构学习算法](https://zhuanlan.zhihu.com/p/368010458)
- Spirtes, P., Glymour, C., & Scheines, R. (2000). Causation, Prediction, and Search. MIT Press.