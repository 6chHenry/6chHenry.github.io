# Cosine Similarity

## 1. 基本概念

余弦相似度（Cosine Similarity）通过计算两个向量的夹角余弦值来衡量它们的相似性，常用于文本、图像等向量化数据的相似度比较。

### 核心特点：

- 范围：[-1, 1]（实际场景通常为[0,1]）
- 对向量幅度不敏感（仅考虑方向）
- 适合高维稀疏数据

## 2. 计算公式

### 原始定义

对于向量A和B：
$$
\text{cosine}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}
$$

### 标准化版本（向量已单位化）

$$
\text{cosine}(A, B) = A \cdot B
$$

## 3. 计算示例

### 示例1：简单数值向量

```python
import numpy as np

# 定义两个向量
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])

# 手动计算
dot_product = np.sum(A * B)          # 1*4 + 2*5 + 3*6 = 32
norm_A = np.sqrt(np.sum(A**2))       # sqrt(1+4+9) ≈ 3.7417
norm_B = np.sqrt(np.sum(B**2))       # sqrt(16+25+36) ≈ 8.7750
cos_sim = dot_product / (norm_A * norm_B)  # 32 / (3.7417*8.7750) ≈ 0.9746

# 使用库函数验证
from sklearn.metrics.pairwise import cosine_similarity
cos_sim_lib = cosine_similarity([A], [B])[0][0]  # 输出0.9746
```

### 示例二：文本向量（TF-IDF）:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "apple banana fruit",
    "banana orange fruit",
    "car vehicle engine"
]

# 生成TF-IDF向量
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# 计算相似度矩阵
cosine_sim_matrix = cosine_similarity(tfidf_matrix)
print(cosine_sim_matrix)
```
