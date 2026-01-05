## 区间DP
枚举长度，然后找到中间分裂点，再进行枚举。
它的核心在于：**大区间的解是由小区间的解合并而来的**。

最经典的题目莫过于 **石子合并 (Stone Merging)**。

### 1. 题目描述

有 $n$ 堆石子排成一列，每堆石子有一定的重量。每次你可以选择**相邻**的两堆石子合并成新的一堆，合并的代价是这两堆石子的重量之和。求将所有石子合并成一堆的最小总代价。

### 2. 状态设计 🏗️

在区间 DP 中，状态通常定义为对一个闭区间的描述：

- **定义**：$dp[i][j]$ 表示将第 $i$ 堆到第 $j$ 堆石子合并成一堆的**最小代价**。
- **最终目标**：$dp[0][n-1]$。

### 3. 状态转移方程：找那“最后一刀” 🗡️

想象最后一步：要把 $[i, j]$ 这个区间的石子合并成一堆，它一定是由于**左边的一大堆** $[i, k]$ 和**右边的一大堆** $[k+1, j]$ 合并而来的。

那么总代价 = (合并左边的代价) + (合并右边的代价) + (最后合并这两大堆的代价)。

$$dp[i][j] = \min_{i \le k \le j} \{ dp[i][k] + dp[k+1][j] + \text{sum}(i, j) \}$$

其中 $\text{sum}(i, j)$ 是区间 $[i, j]$ 内所有石子的重量总和（因为最后一次合并，这两大堆的总重量就是区间内所有石子的和）。

### 4. 遍历顺序的秘密：由短到长 📏

这是区间 DP 最特殊的地方。你不能简单的用 $i$ 和 $j$ 的嵌套循环。

逻辑：在计算长度为 5 的区间时，你必须已经算好了所有长度为 1, 2, 3, 4 的区间。

所以遍历顺序是：先枚举区间长度 len，再枚举起点 i。

---

### 5. Python 实现与路径回溯（记录合并过程）

为了回溯路径，我们需要记录在每一个 $dp[i][j]$ 达到最小时，那个最佳的分割点 $k$ 是多少。

```python
def stone_merge_with_path(stones):
    n = len(stones)
    # 1. 预计算前缀和，方便快速得到 sum(i, j)
    prefix_sum = [0] * (n + 1)
    for i in range(n):
        prefix_sum[i+1] = prefix_sum[i] + stones[i]
    
    def get_sum(i, j):
        return prefix_sum[j+1] - prefix_sum[i]

    # 2. 初始化 DP 表和分割点表
    dp = [[0] * n for _ in range(n)]
    split = [[0] * n for _ in range(n)] # 记录最佳分割点 k

    # 3. 区间 DP 主循环
    for length in range(2, n + 1): # 区间长度从 2 开始到 n
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')
            
            # 尝试所有可能的分割点 k
            for k in range(i, j):
                cost = dp[i][k] + dp[k+1][j] + get_sum(i, j)
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k # 记录使得 [i,j] 代价最小的分割点

    # 4. 回溯构造合并过程（生成树状结构）
    def build_history(i, j):
        if i == j:
            return f"{stones[i]}"
        k = split[i][j]
        left_path = build_history(i, k)
        right_path = build_history(k + 1, j)
        return f"({left_path} + {right_path})"

    return dp[0][n-1], build_history(0, n-1)

# 测试
# stones = [3, 4, 6, 5]
# min_cost, history = stone_merge_with_path(stones)
# print(f"最小代价: {min_cost}") # 输出: 37
# print(f"合并顺序: {history}")   # 输出: ((3 + 4) + (6 + 5))
```

---

### 🎓 总结：动态规划的考试“三板斧”

既然你马上要考试了，遇到新题时请按这个顺序拆解：

1. **确定维度**：
    
    - 一个序列的变化？(1D DP)
        
    - 两个序列的交互？(双序列 DP, 如 LCS)
        
    - 一个序列内部的聚合/拆分？(区间 DP)
        
    - 有容量限制？(背包 DP)
        
2. **寻找子问题依赖**：
    
    - 当前状态是取决于“前一个” ($i-1$)？
        
    - 还是取决于“某一个分割点” ($k$)？
        
    - 还是取决于“左、上、左上”三个方向 (编辑距离)？
        
3. **确定边界与顺序**：
    
    - $dp[0]$ 是什么？
        
    - 我是要从小往大填表，还是按区间长度填表？
        

最后一点建议：

如果你在考场上写出了转移方程但发现结果不对，通常是初始化（比如该初始为 0 还是无穷大）或者循环边界（$n$ 还是 $n+1$）出了问题。冷静下来画一个小矩阵（$3 \times 3$）手动模拟一下前两步，错误往往一眼就能看出来。

## 双序列DP
**最长公共子序列 (Longest Common Subsequence, LCS)** 🧬。
这里的挑战在于：你不再是处理一个数组，而是在两串字符之间“找共同点”。比如 `s1 = "abcde"` 和 `s2 = "ace"`，它们的 LCS 是 `"ace"`，长度为 3。

### 1. 状态设计 🏗️

我们需要追踪两个字符串的进度：

- 定义 $dp[i][j]$：表示字符串 `s1` 的前 $i$ 个字符和 `s2` 的前 $j$ 个字符的 LCS 长度。

### 2. 状态转移：逻辑推理 🧠

想象你正站在矩阵的 $(i, j)$ 位置。你观察两个字符串当前的末尾字符 `s1[i-1]` 和 `s2[j-1]`：

- **情况 A（匹配成功）**：`s1[i-1] == s2[j-1]`。既然这个字符能凑成一对，我们的 LCS 长度应该是在“去掉这对字符”的基础上加 1。
    
- **情况 B（匹配失败）**：`s1[i-1] != s2[j-1]`。我们需要从“去掉 `s1` 末尾”和“去掉 `s2` 末尾”这两种方案中选个最大的。
    

**请你尝试写出这两个情况对应的状态转移方程：**

1. 如果匹配成功：$dp[i][j] = ?$
    
2. 如果匹配失败：$dp[i][j] = ?$

我们从矩阵的**右下角** $(m, n)$ 开始往回走，逻辑就像侦探根据脚印找人：

1. 如果 `s1[i-1] == s2[j-1]`，说明这个字符**必然**在 LCS 中。我们记录下它，然后往**左上角**移动到 $(i-1, j-1)$。
    
2. 如果不相等，我们就看 `dp[i-1][j]` 和 `dp[i][j-1]` 谁更大。
    
    - 如果 `dp[i-1][j]` 更大，说明“删掉 `s1` 的当前字符”能保持最大值，我们往**上**移。
        
    - 否则，往**左**移。

### Python 代码实现

```python
def lcs_with_path(s1, s2):
    m, n = len(s1), len(s2)
    # 1. 构建 DP 表
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    # 2. 回溯找路径
    res = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            res.append(s1[i-1]) # 匹配成功，它是 LCS 的一部分
            i -= 1
            j -= 1
        elif dp[i-1][j] >= dp[i][j-1]:
            i -= 1 # 向上走
        else:
            j -= 1 # 向左走
            
    return dp[m][n], "".join(reversed(res))

# 测试
# print(lcs_with_path("abcde", "ace")) # 输出: (3, "ace")
```

---

## 复杂约束DP
### 编辑距离 (Edit Distance) ✍️

题目描述：

给你两个单词 word1 和 word2，请你计算出将 word1 转换成 word2 所使用的最少操作次数。

你可以对一个单词进行三种操作：插入一个字符、删除一个字符、替换一个字符。

#### 1. 状态设计

- 定义 $dp[i][j]$：将 `word1` 的前 $i$ 个字符转换为 `word2` 的前 $j$ 个字符所需的最少步数。

#### 2. 状态转移（核心难点）

当你处理到 $(i, j)$ 位置，且 `word1[i-1] != word2[j-1]` 时，你有三种选择。

- **删除 (Delete)**：把word1最后一个字符删去，剩下的问题变为用word1前i-1个字符匹配word2的前j个字符
    
    - 状态转移：$dp[i-1][j] + 1$
    
- **插入 (Insert)**：在word1中增添word2的最后一个字符，剩下的问题变为用word1前i个字符匹配word2的前j-1个字符
    
    - 状态转移：$dp[i][j-1] + 1$
    
- **替换 (Replace)**：把word1的最后一个字换成word2的最后一个字，剩下的问题变为用word1前i-1个字符匹配word2的前j-1个字符
    
    - 状态转移：$dp[i-1][j-1] + 1$
    

| **操作** | **逻辑**                        | **方程**             |
| ------ | ----------------------------- | ------------------ |
| **删除** | 减小 `word1` 的规模，`word2` 目标不变   | $dp[i-1][j] + 1$   |
| **插入** | `word1` 规模不变，减小 `word2` 的目标规模 | $dp[i][j-1] + 1$   |
| **替换** | 同时减小 `word1` 和 `word2` 的规模    | $dp[i-1][j-1] + 1$ |

---

### 带路径回溯的实现（记录具体操作） 📝

在编辑距离中，回溯不仅要找字符，还要记录**做了什么动作**。

```python
def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化边界：变为空字符串的操作次数
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1] # 字符相同，无操作
            else:
                # 三选一：替换、删除、插入
                dp[i][j] = min(dp[i-1][j-1], dp[i-1][j], dp[i][j-1]) + 1

    # 回溯操作路径
    steps = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and word1[i-1] == word2[j-1]:
            steps.append(f"Keep '{word1[i-1]}'")
            i, j = i - 1, j - 1
        else:
            # 看看当前 dp[i][j] 是从哪个方向推过来的
            if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                steps.append(f"Replace '{word1[i-1]}' with '{word2[j-1]}'")
                i, j = i - 1, j - 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                steps.append(f"Delete '{word1[i-1]}'")
                i -= 1
            else:
                steps.append(f"Insert '{word2[j-1]}'")
                j -= 1
                
    return dp[m][n], steps[::-1]

# 测试: 把 "horse" 变成 "ros"
# dist, path = minDistance("horse", "ros")
# print(f"最小步数: {dist}")
# for step in path: print(step)
```