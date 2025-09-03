# 第3课-贝尔曼最优公式（最优策略和公式推导）知识点整理
## 一、最优策略的定义与核心问题
### 1. 策略优劣的比较标准
- 若存在两个策略$\pi_1$和$\pi_2$，对于**所有状态$s$**，$\pi_1$对应的状态价值$v_{\pi_1}(s)$均大于$\pi_2$对应的状态价值$v_{\pi_2}(s)$，则称$\pi_1$优于$\pi_2$。
- 最优策略$\pi^*$的定义：对**任意状态$s$** 和**任意其他策略$\pi$**，均满足$v_{\pi^*}(s) \geq v_{\pi}(s)$，即$\pi^*$在所有状态下的价值都不低于其他任何策略。

### 2. 最优策略的四大核心问题
1. **存在性**：是否存在满足上述定义的最优策略$\pi^*$？（理想中“全状态优于其他策略”的策略是否真实存在？）
2. **唯一性**：最优策略是唯一的，还是存在多个不同但均满足“最优”条件的策略？
3. **策略类型**：最优策略是确定性策略（某状态下固定选择一个动作），还是随机性策略（某状态下按概率选择多个动作）？
4. **求解方法**：如何通过数学工具推导并得到最优策略$\pi^*$？（核心问题，需通过贝尔曼最优公式解答）


## 二、贝尔曼最优公式
### 1. 公式形式与核心改动
贝尔曼最优公式是在**普通贝尔曼公式**（依赖给定策略$\pi$）的基础上，增加了“策略最大化”操作，具体形式如下：

| 公式类型       | 表达式核心逻辑                                               | 关键区别                                                     |
| -------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 普通贝尔曼公式 | $v_{\pi}(s) = \mathbb{E}_{\pi}\left[ r + \gamma v_{\pi}(s') \mid s \right]$ | 策略$\pi$是**给定的**，仅需计算该策略下的状态价值$v_{\pi}(s)$ |
| 贝尔曼最优公式 | $v_*(s) = \max_{\pi} \mathbb{E}_{\pi}\left[ r + \gamma v_*(s') \mid s \right]$ | 策略$\pi$是**待优化的**，需先找到使价值最大的$\pi$，再计算最优状态价值$v_*(s)$ |

- 符号说明：$v_*(s)$表示“最优状态价值”，即最优策略$\pi^*$对应的状态价值；$\max_{\pi}$表示对所有可能的策略取最大值。
- 简化表达：公式中期望项可缩写为动作价值$q(s,a)$（即状态$s$下选择动作$a$的价值），因此最优公式可进一步简化为$v_*(s) = \max_a q(s,a)$。

### 2. 已知条件与求解目标
| 类别     | 具体内容                                                     |
| -------- | ------------------------------------------------------------ |
| 已知条件 | 1. 状态转移概率$p(s' \mid s,a)$（系统模型参数，描述“状态$s$选动作$a$后转移到$s'$”的概率）<br>2. 即时奖励$r$（环境反馈，如“选动作$a$后获得的收益/惩罚”）<br>3. 折扣因子$\gamma$（控制未来奖励的权重，$0 \leq \gamma \leq 1$） |
| 求解目标 | 1. 最优状态价值$v_*(s)$（所有状态下的最优价值向量）<br>2. 最优策略$\pi^*$（使$v(s)$达到$v_*(s)$的策略） |


### 3. 公式的“优美性”与“复杂性”
- **优美性**：形式简洁，仅通过“$\max_{\pi}$”操作，就将“最优策略”与“最优状态价值”的关系刻画清楚，统一了“策略优化”与“价值计算”两个问题。
- **复杂性**：
  1. 嵌套优化：公式右侧包含“对策略$\pi$求最大值”的优化问题，需先解决优化问题才能计算$v_*(s)$；
  2. 双未知量：表面上需同时求解$v_*(s)$（价值向量）和$\pi^*$（策略），初学者易困惑“如何用一个公式解两个未知量”。


## 三、贝尔曼最优公式的求解思路（核心推导）
### 1. 求解逻辑：分两步拆解“双未知量”问题
贝尔曼最优公式的求解核心是**先固定价值求策略，再代入策略求价值**，通过“分步拆解”解决“双未知量”困境，具体以两个例子说明：

目标:

$v(s) = \max_\pi \sum_a \pi(a|s)(\sum_r (p|s,a)r+\gamma \sum_{s'}p(s'|s,a)v(s')),\forall s \in \mathcal{S} \\ = \max_\pi \sum_a \pi(a|s)q(s,a)$

#### 例1：How to solve 2 unknowns from 1 equation

假设存在公式$x = \max_a (2x - 1 - a^2)$，需求解$x$（对应$v_*(s)$）和$a$（对应策略$\pi^*$的动作选择）：
1. **第一步：固定$x$，求最优$a$**：  
   对$a$求最大值，因$-a^2$的最大值在$a=0$时取得（此时$-a^2=0$），因此$\max_a (2x - 1 - a^2) = 2x - 1$，最优$a=0$。
2. **第二步：代入$a=0$，求$x$**：  
   公式变为$x = 2x - 1$，解得$x=1$。  
   最终结果：$x=1$（最优价值），$a=0$（最优动作）。

#### 例2 How to solve $\max_{\pi} \sum_{a} \pi(a|s)q(s,a)$

Suppose $q_1, q_2, q_3 \in \mathbb{R}$ are given. Find $c_1^*, c_2^*, c_3^*$ solving
$$
\max_{c_1,c_2,c_3} c_1q_1 + c_2q_2 + c_3q_3.
$$
where $c_1 + c_2 + c_3 = 1$ and $c_1, c_2, c_3 \geq 0$.

Without loss of generality, suppose $q_3 \geq q_1, q_2$. Then, the optimal solution is $c_3^* = 1$ and $c_1^* = c_2^* = 0$. That is because for any $c_1, c_2, c_3$
$$
q_3 = (c_1 + c_2 + c_3)q_3 = c_1q_3 + c_2q_3 + c_3q_3 \geq c_1q_1 + c_2q_2 + c_3q_3.
$$

#### 强化学习场景（映射到贝尔曼最优公式）

假设某状态$s$下有5个可能动作$a_1 \sim a_5$，对应动作价值$q(s,a_1) \sim q(s,a_5)$，需求解$\pi^*(s,a)$（策略）和$v_*(s)$（最优价值）：
1. **第一步：固定$v_*(s')$，求最优$\pi$**：  
   动作价值$q(s,a) = r + \gamma \sum_{s'} p(s' \mid s,a) v_*(s')$，若暂时固定$v_*(s')$（可先设初始值），则$q(s,a)$可计算。  
   最优策略需满足：对“使$q(s,a)$最大的动作$a^*$”，取$\pi^*(s,a^*) = 1$（确定性选择该动作）；对其他动作$a \neq a^*$，取$\pi^*(s,a) = 0$。  
   此时$\max_{\pi} \mathbb{E}_{\pi}[q(s,a)] = \max_a q(s,a)$（即最优动作对应的动作价值）。
2. **第二步：代入最优$\pi$，求$v_*(s)$**：  
   最优状态价值$v_*(s) = \max_a q(s,a)$，即“最优动作的动作价值”就是该状态的最优状态价值。

Inspired by the above example, considering that $\sum_a \pi(a|s) = 1$, 

we have   $\underbrace{\max_{\pi} \sum_a \pi(a|s) q(s, a)} = \underbrace{\max_{a \in \mathcal{A}(s)} q(s, a)},$   

where the optimality is achieved when   $\pi(a|s) = \begin{cases}  1 & a = a^* \\ 0 & a \neq a^*  \end{cases}$

   where $a^* = \arg \max_a q(s, a)$.


### 2. 核心结论
- 贝尔曼最优公式的求解本质是“**贪心策略**”：在每个状态下，选择能使“即时奖励+未来最优价值”最大的动作，该动作对应的策略即为局部最优策略，所有状态的局部最优策略构成全局最优策略$\pi^*$。
- 最优策略的类型：通过上述推导可知，最优策略可表示为**确定性策略**（选择$q(s,a)$最大的动作），即使存在多个动作的$q(s,a)$相等，也可通过“确定性选择其中任意一个”实现最优，无需随机性策略。

## 四、公式与最优策略的关联

贝尔曼最优公式是连接“最优价值”与“最优策略”的桥梁：
1. 若已知最优状态价值$v_*(s)$，可通过$q(s,a) = r + \gamma \sum_{s'} p(s' \mid s,a) v_*(s')$计算所有动作价值，再选择$q(s,a)$最大的动作，得到最优策略$\pi^*$；
2. 若已知最优策略$\pi^*$，可通过普通贝尔曼公式$v_*(s) = \mathbb{E}_{\pi^*}\left[ r + \gamma v_*(s') \mid s \right]$计算最优状态价值$v_*(s)$。



# 第3课-贝尔曼最优公式（公式求解以及最优性）知识点整理
## 一、贝尔曼最优公式的简化形式
1. **核心转化逻辑**：将贝尔曼最优公式中右侧的$maxπ$（对策略的最大化）定义为函数$f(v)$，其中$v$为状态值函数（State Value）。  
   - 关键前提：固定$v$后，可求解出对应策略$π$，最终最优值仅与$v$相关，因此$maxπ$的结果可表示为$v$的函数$f(v)$。  
   - 简化后公式：$v = f(v)$（$f(v)$为向量，向量中对应状态$s$的元素即该状态下的最优值计算项）。

2. **$f(v)$的向量属性**：$f(v)$的每个元素对应一个状态$s$，描述该状态在最优策略下的价值，需结合状态转移概率、奖励等强化学习核心要素计算（承接前序课程中贝尔曼公式的基础定义）。

## 二、核心数学工具：收缩映射定理（Contraction Mapping Theorem）

### （一）前置基础概念
1. **不动点（Fixed Point）**  
   - 定义：若在集合$X$上存在点$x$，映射（函数）$f$满足$f(x) = x$，则$x$称为$f$的不动点。  
   - 直观解释：点$x$经过函数$f$映射后仍回到自身，“位置不变”。

2. **收缩映射（Contraction Mapping / Contractive Function）**  
   - 定义：对任意两个点$x₁$、$x₂$，若存在常数$γ < 1$（收缩因子），使得$||f(x₁) - f(x₂)|| ≤ γ·||x₁ - x₂||$（$||·||$表示范数，衡量“距离”），则$f$为收缩映射。  
   - 直观解释：任意两点经过$f$映射后，它们的“距离”会被缩小（收缩因子$γ$控制缩小比例），因此函数具有“压缩”特性。

3. **示例验证**  
   | 案例                             | 不动点验证                          | 收缩映射验证                                                 |
   | -------------------------------- | ----------------------------------- | ------------------------------------------------------------ |
   | 标量函数$f(x) = 0.5x$            | $f(0) = 0.5×0 = 0$，故$x=0$是不动点 | 对任意$x_1、x_2$，$||0.5x_1 - 0.5x_2|| = 0.5×||x_1 - x_2||$，取$γ=0.5 < 1$，满足定义 |
   | 向量函数$f(x) = Ax$（$A$为矩阵） | $f(0) = A×0 = 0$，故$x=0$是不动点   | 若矩阵$A$的范数$||A|| < 1$，则$||Ax_1 - Ax_2|| ≤ ||A||·||x_1 - x_2||$，满足定义 |


### （二）收缩映射定理的核心结论
若$f$是收缩映射，则方程$x = f(x)$满足以下3个关键性质（为贝尔曼最优公式求解提供理论保障）：
1. **存在性（Existence）**：必然存在至少一个不动点$x_*$，使得$f(x_*) = x_*$（无需关注$f$的具体表达式，仅需确认其为收缩映射）。  
2. **唯一性（Uniqueness）**：上述不动点$x_*$是唯一的（排除“多解”问题，确保最优值的确定性）。  
3. **可解性（Solvability）**：可通过迭代算法求解$x_*$，迭代公式为$x_{k+1} = f(x_k)$（$x_0$为初始值）：  
   - 收敛性：当迭代次数$k→∞$时，$x_k$会收敛到不动点$x_*$；  
   - 实用性：实际计算中无需迭代至无穷次，迭代若干步后即可得到满足精度的近似解；  
   - 收敛速度：指数级收敛（收敛速度快，计算效率高）。

> [!INFO] # 压缩映射定理的证明
>
> ### 部分1：证明序列 $\{x_k\}_{k=1}^\infty$（其中 $x_k = f(x_{k-1})$）收敛
>
> 证明依赖**柯西序列**的概念：若序列 $x_1, x_2, \dots$ 满足“对任意小的 $\varepsilon > 0$，存在整数 $N$，使得对所有 $m, n > N$，有 $\|x_m - x_n\| < \varepsilon$”，则称该序列为柯西序列。其直观含义是“$N$ 之后的所有元素都足够接近”。柯西序列的核心性质是：**柯西序列必收敛到一个极限**，这一性质是证明的关键。
>
> 注意：仅满足“相邻项差趋于0（$x_{n+1} - x_n \to 0$）”不足以说明是柯西序列（例如 $x_n = \sqrt{n}$，虽 $x_{n+1} - x_n \to 0$，但 $x_n = \sqrt{n}$ 发散）。
>
>
> 接下来证明 $\{x_k = f(x_{k-1})\}_{k=1}^\infty$ 是柯西序列（从而收敛）：
>
> 1. **压缩映射的递推估计**：
>    因 $f$ 是压缩映射（存在常数 $\gamma$，$0 \leq \gamma < 1$，使得对任意 $x, y$，有 $\|f(x) - f(y)\| \leq \gamma \|x - y\|$），故：
>    $$
>    \|x_{k+1} - x_k\| = \|f(x_k) - f(x_{k-1})\| \leq \gamma \|x_k - x_{k-1}\|
>    $$
>    递推可得：
>    $$
>    \|x_k - x_{k-1}\| \leq \gamma \|x_{k-1} - x_{k-2}\|, \quad \dots, \quad \|x_2 - x_1\| \leq \gamma \|x_1 - x_0\|
>    $$
>    因此，通过迭代放缩：
>    $$
>    \|x_{k+1} - x_k\| \leq \gamma^k \|x_1 - x_0\|
>    $$
>    由于 $\gamma < 1$，$\|x_{k+1} - x_k\|$ 随 $k \to \infty$ 指数级收敛到0，但这**不足以直接推出 $\{x_k\}$ 收敛**，需进一步分析任意两项的差。
>
>
> 2. **任意两项差的估计（柯西序列判定）**：
>    对任意 $m > n$，将 $\|x_m - x_n\|$ 拆分为相邻项的累加和：
>    $$
>    \|x_m - x_n\| = \|x_m - x_{m-1} + x_{m-1} - \dots - x_{n+1} + x_{n+1} - x_n\|
>    $$
>    由范数的三角不等式，得：
>    $$
>    \|x_m - x_n\| \leq \|x_m - x_{m-1}\| + \dots + \|x_{n+1} - x_n\|
>    $$
>    代入“$\|x_{k+1} - x_k\| \leq \gamma^k \|x_1 - x_0\|$”的估计，得：
>    $$
>    \|x_m - x_n\| \leq \gamma^{m-1} \|x_1 - x_0\| + \dots + \gamma^n \|x_1 - x_0\|
>    $$
>    这是公比为 $\gamma$（$\gamma < 1$）的等比数列求和，因此：
>    $$
>    \|x_m - x_n\| \leq \gamma^n \cdot \frac{1 - \gamma^{m-n}}{1 - \gamma} \cdot \|x_1 - x_0\| \leq \frac{\gamma^n}{1 - \gamma} \|x_1 - x_0\| \tag{3.4}
>    $$
>
>
> 3. **柯西序列的结论**：
>    对任意 $\varepsilon > 0$，由于 $\gamma^n \to 0$（当 $n \to \infty$），总能找到 $N$，使得对所有 $m, n > N$，有 $\|x_m - x_n\| < \varepsilon$。因此，$\{x_k\}$ 是柯西序列，故必收敛到极限点 $x^* = \lim_{k \to \infty} x_k$。
>
>
> ### 部分2：证明极限 $x^* = \lim_{k \to \infty} x_k$ 是不动点
>
> 由压缩映射的估计，$\|f(x_k) - x_k\| = \|x_{k+1} - x_k\| \leq \gamma^k \|x_1 - x_0\|$。由于 $\gamma < 1$，$\|f(x_k) - x_k\|$ 随 $k \to \infty$ 指数级收敛到0。
>
> 对 $f(x_k) - x_k$ 取极限（因 $f$ 连续，极限可交换），得：
> $$
> \lim_{k \to \infty} \|f(x_k) - x_k\| = \|f(x^*) - x^*\| = 0
> $$
> 故 $f(x^*) = x^*$，即 $x^*$ 是不动点。
>
>
> ### 部分3：证明不动点唯一
>
> 假设存在另一个不动点 $x'$（满足 $f(x') = x'$），则由压缩映射的定义：
> $$
> \|x' - x^*\| = \|f(x') - f(x^*)\| \leq \gamma \|x' - x^*\|
> $$
> 由于 $\gamma < 1$，上述不等式成立当且仅当 $\|x' - x^*\| = 0$，故 $x' = x^*$，即不动点唯一。
>
>
> ### 部分4：证明 $x_k$ 指数级收敛到 $x^*$
>
> 回顾式 (3.4)：对任意 $m > n$，有 $\|x_m - x_n\| \leq \frac{\gamma^n}{1 - \gamma} \|x_1 - x_0\|$。令 $m \to \infty$（利用极限的保序性），则：
> $$
> \|x^* - x_n\| = \lim_{m \to \infty} \|x_m - x_n\| \leq \frac{\gamma^n}{1 - \gamma} \|x_1 - x_0\|
> $$
> 由于 $\gamma < 1$，当 $n \to \infty$ 时，误差 $\|x^* - x_n\|$ 随 $\gamma^n$ 指数级收敛到0。

## 三、贝尔曼最优公式的求解（基于收缩映射定理）

### （一）关键前提：证明$f(v)$是收缩映射
贝尔曼最优公式中的$f(v)$满足收缩映射定义，核心依据是**折扣因子$γ < 1$**（强化学习中$γ$用于衡量未来奖励的权重，通常取$0 < γ < 1$），由此可推导出$||f(v_1) - f(v_2)|| ≤ γ·||v_1 - v_2||$，故$f(v)$是收缩映射。

> [!info]  Proof of Theorem 3.2
>
> Consider any two vectors  $v_1, v_2 \in \mathbb{R}^{|\mathcal{S}|} $, and suppose that $ \pi_1^* \doteq \arg\max_\pi (r_\pi + \gamma P_\pi v_1)  $and$  \pi_2^* \doteq \arg\max_\pi (r_\pi + \gamma P_\pi v_2) $. Then,
>
> $$
> f(v_1) = \max_\pi (r_\pi + \gamma P_\pi v_1) = r_{\pi_1^*} + \gamma P_{\pi_1^*} v_1 \geq r_{\pi_2^*} + \gamma P_{\pi_2^*} v_1,
> $$
>
> $$
> f(v_2) = \max_\pi (r_\pi + \gamma P_\pi v_2) = r_{\pi_2^*} + \gamma P_{\pi_2^*} v_2 \geq r_{\pi_1^*} + \gamma P_{\pi_1^*} v_2,
> $$
>
> where \( \geq \) is an elementwise comparison. As a result,
>
> $$
> \begin{align*}
> f(v_1) - f(v_2) &= r_{\pi_1^*} + \gamma P_{\pi_1^*} v_1 - \left( r_{\pi_2^*} + \gamma P_{\pi_2^*} v_2 \right) \\
> &\leq r_{\pi_1^*} + \gamma P_{\pi_1^*} v_1 - \left( r_{\pi_1^*} + \gamma P_{\pi_1^*} v_2 \right) \\
> &= \gamma P_{\pi_1^*} (v_1 - v_2).
> \end{align*}
> $$
>
> Similarly, it can be shown that $ f(v_2) - f(v_1) \leq \gamma P_{\pi_2^*} (v_2 - v_1) $. Therefore,
>
> $$
> \gamma P_{\pi_2^*} (v_1 - v_2) \leq f(v_1) - f(v_2) \leq \gamma P_{\pi_1^*} (v_1 - v_2).
> $$
>
> Define
>
> $$
> z \doteq \max \left\{ |\gamma P_{\pi_2^*} (v_1 - v_2)|, |\gamma P_{\pi_1^*} (v_1 - v_2)| \right\} \in \mathbb{R}^{|\mathcal{S}|},
> $$
>
> where $ \max(\cdot) ,  |\cdot| , and  \geq $ are all elementwise operators. By definition, $ z \geq 0 $. On the one hand, it is easy to see that
>
> $$
> -z \leq \gamma P_{\pi_2^*} (v_1 - v_2) \leq f(v_1) - f(v_2) \leq \gamma P_{\pi_1^*} (v_1 - v_2) \leq z,
> $$
>
> which implies
>
> $$
> |f(v_1) - f(v_2)| \leq z.
> $$
>
> It then follows that
>
> $$
> \| f(v_1) - f(v_2) \|_\infty \leq \| z \|_\infty, \tag{3.5}
> $$
>
> where $ \| \cdot \|_\infty $ is the maximum norm.
>
> On the other hand, suppose that $ z_i $ is the $ i $-th entry of $ z $, and $ p_i^T $ and $ q_i^T $ are the $ i $-th row of $ P_{\pi_1^*} $ and $ P_{\pi_2^*} $, respectively. Then,
>
> $$
> z_i = \max \left\{ |\gamma p_i^T (v_1 - v_2)|, |\gamma q_i^T (v_1 - v_2)| \right\}.
> $$
>
> Since $p_i$ is a vector with all nonnegative elements and the sum of the elements is equal to one, it follows that
>
> $$
> |p_i^T (v_1 - v_2)| \leq p_i^T |v_1 - v_2| \leq \| v_1 - v_2 \|_\infty.
> $$
>
> Similarly, we have $ |q_i^T (v_1 - v_2)| \leq \| v_1 - v_2 \|_\infty $. Therefore, $ z_i \leq \gamma \| v_1 - v_2 \|_\infty $ and hence
>
> $$
> \| z \|_\infty = \max_i |z_i| \leq \gamma \| v_1 - v_2 \|_\infty.
> $$
>
> Substituting this inequality gives
>
> $$
> \| f(v_1) - f(v_2) \|_\infty \leq \gamma \| v_1 - v_2 \|_\infty,
> $$
>
> which concludes the proof of the contraction property of  $f(v)$.


### （二）求解结论（对应收缩映射定理的3个性质）
1. **解的存在性**：贝尔曼最优公式$v = f(v)$必然存在解，记为最优状态值函数$v*$。  
2. **解的唯一性**：最优状态值函数$v^*$是唯一的（不存在多个“最优值”）。  
3. **迭代求解方法**：通过$v_{k+1} = f(v_k)$迭代计算，$v_k$最终会收敛到$v^*$（后续课程中的“值迭代算法”即基于此原理）。


## 四、贝尔曼最优公式解的最优性（与最优策略的关联）
### （一）最优策略的定义与推导

1. **最优策略$π^*$的定义**：当状态值函数固定为$v^*$时，能使$f(v^*) = v^*$成立的策略，即对每个状态$s$，选择能最大化“即时奖励 + 折扣后未来价值”的动作。  
2. **公式转化**：将贝尔曼最优公式中的$\max π$替换为$π^*$，得到$v^* = r + γP_{π^*}v^*$（$r$为奖励向量，$P_{π^*}$为$π^*$对应的状态转移概率矩阵），该式本质是**对应最优策略$π^*$的贝尔曼公式**。  
   - 结论：$v^*$就是最优策略$π^*$对应的状态值函数，即$v^* = v_{π^*}$（$v^π*$表示策略$π*$的状态值）。


### （二）$v*$的最优性证明
$v*$是**所有可能策略对应的状态值函数中的最大值**：  
- 对任意非最优策略$π$，其状态值函数$v_π$满足$v_π ≤ v^*$（即$v*$优于所有非最优策略的价值）；  
- 因此，$π^*$是最优策略（其对应的$v_{π^*} = v^*$为最大价值）。

> [!info]  Proof of Theorem 3.4
>
> For any policy $ \pi $, it holds that
>
> $$
> v_\pi = r_\pi + \gamma P_\pi v_\pi.
> $$
>
> Since
>
> $$
> v^* = \max_\pi (r_\pi + \gamma P_\pi v^*) = r_{\pi^*} + \gamma P_{\pi^*} v^* \geq r_\pi + \gamma P_\pi v^*,
> $$
>
> we have
>
> $$
> v^* - v_\pi \geq (r_\pi + \gamma P_\pi v^*) - (r_\pi + \gamma P_\pi v_\pi) = \gamma P_\pi (v^* - v_\pi).
> $$
>
> Repeatedly applying the above inequality gives $ v^* - v_\pi \geq \gamma P_\pi (v^* - v_\pi) \geq \gamma^2 P_\pi^2 (v^* - v_\pi) \geq \cdots \geq \gamma^n P_\pi^n (v^* - v_\pi) $. It follows that
>
> $$
> v^* - v_\pi \geq \lim_{n \to \infty} \gamma^n P_\pi^n (v^* - v_\pi) = 0,
> $$
>
> where the last equality is true because $\gamma < 1 $ and $ P_\pi^n $ is a nonnegative matrix with all its elements less than or equal to 1 (because $ P_\pi^n \mathbf{1} = \mathbf{1} $) Therefore, $ v^* \geq v_\pi $ for any $ \pi $.

### （三）最优策略$π*$的形式

$π*$是**确定性（Deterministic）贪心策略（Greedy Policy）**：  
- 对每个状态$s$，$π^*$会选择使“动作值函数（Action Value）$q^*(s,a)$”最大的动作$a^*$；  
- 选择概率：$π^*(a^*|s) = 1$（必然选择最优动作），$π^*(a|s) = 0$（不选择其他动作）。



# 第3课-贝尔曼最优公式（最优策略的有趣性质）知识点整理
## 一、贝尔曼最优公式核心逻辑
1. **公式作用**：作为求解强化学习中**最优策略（π*）** 和**最优状态价值（v*）** 的核心工具，能直接关联已知条件与目标解，为策略优化提供数学依据。
2. **变量关系**：
   - **已知量（红色变量）**：决定最优策略的关键输入，是构建贝尔曼最优公式的基础，包括三类：
     - 系统模型（概率P）：描述状态转移规律的量化指标，即从当前状态s执行动作a后，转移到下一状态s'的概率，反映环境本身的动态特性。
     - 奖励函数（r）：人工定义的奖惩规则，用于引导智能体行为，例如到达目标区域奖励为正（如+1）、进入禁止区域或撞边界奖励为负（如-1）。
     - 折扣因子（γ）：调节智能体对短期奖励与长期奖励重视程度的参数，取值范围为[0,1)，是平衡即时收益与未来收益的核心。
   - **求解量（黑色变量）**：通过贝尔曼最优公式计算得出的目标结果，即能使长期收益最大化的最优策略（π*），以及对应策略下各状态的最优状态价值（v*）。


## 二、影响最优策略的关键因素（实验验证）
以“网格世界”为实验场景（包含目标区域、禁止区域与边界），验证**奖励函数（r）** 和**折扣因子（γ）** 对最优策略的直接影响（系统模型通常由环境决定，难以改变，故暂不讨论）。

### 1. 折扣因子（γ）的影响：控制智能体“短视/远视”特性
折扣因子的大小直接决定智能体对未来奖励的权重分配，进而改变策略选择，实验分三种典型场景：

| 折扣因子（γ） | 智能体特性               | 策略表现（网格世界案例）                             | 核心原因                                                     |
| ------------- | ------------------------ | ---------------------------------------------------- | ------------------------------------------------------------ |
| 0.9（较大）   | 远视：重视长期总收益     | 主动穿过禁止区域（短期承受-1惩罚），快速抵达目标区域 | 虽然穿过禁止区域会有短期惩罚，但能更早获得目标区域的长期奖励；γ较大时，未来奖励折扣幅度小，总收益高于绕路方案。 |
| 0.5（中等）   | 中性：平衡短期与长期收益 | 绕开禁止区域，选择更长路径前往目标区域               | 此时绕路的短期无惩罚，叠加未来奖励的折扣效应后，总收益超过穿过禁止区域的方案，策略自然倾向规避风险。 |
| 0（极小）     | 极端短视：仅关注即时奖励 | 原地不动或仅选择即时奖励≥0的动作，无法到达目标区域   | 当γ=0时，未来奖励经折扣后全部为0，总收益等价于即时奖励；智能体仅规避即时惩罚（如撞边界、进禁止区），完全忽略长期目标。 |


### 2. 奖励函数（r）的影响：调整奖惩权重引导行为
通过改变“禁止区域”的惩罚力度，可直接反转智能体的策略选择：
- 原设置（禁止区域r=-1，γ=0.9）：智能体选择穿过禁止区域，以短期小惩罚换取长期高收益。
- 调整后（禁止区域r=-10，γ=0.9）：智能体选择绕开禁止区域，因短期大惩罚的代价远超长期奖励收益，策略随奖惩权重变化而反转。


## 三、最优策略的不变性：奖励线性变换不改变策略
### 1. 核心结论
若对所有奖励进行**线性变换**（即$$r' = \alpha \cdot r + \beta$$，其中$\alpha > 0$为正的缩放因子，$\beta$为偏置量），则**最优策略（π*）保持不变**，仅最优状态价值（v\*）会同步发生线性变换。

变换公式为$v' = \alpha \cdot v^* + \frac{\beta}{1-\gamma}\mathbb{1}$,其中$\gamma$为discounted rate，$\mathbb{1} =[1,1,...,1]^T$

> [!info]### Box 3.5: Proof of Theorem 3.6
>
> For any policy $ \pi $, define $ r_\pi = [\ldots, r_\pi(s), \ldots]^T $ where
>
> $$
> r_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \sum_{r \in \mathcal{R}} p(r|s,a) r, \quad s \in \mathcal{S}.
> $$
>
> If $ r \to \alpha r + \beta $, then $ r_\pi(s) \to \alpha r_\pi(s) + \beta $ and hence $ r_\pi \to \alpha r_\pi + \beta \mathbf{1} $, where $ \mathbf{1} = [1, \ldots, 1]^T $. In this case, the BOE becomes
>
> $$
> v' = \max_{\pi \in \Pi} (\alpha r_\pi + \beta \mathbf{1} + \gamma P_\pi v'). \tag{3.9}
> $$
>
> We next solve the new BOE in (3.9) by showing that $ v' = \alpha v^* + c \mathbf{1} $ with $ c = \beta/(1-\gamma) $ is a solution of (3.9). In particular, substituting $ v' = \alpha v^* + c \mathbf{1} $ into (3.9) gives
>
> $$
> \alpha v^* + c \mathbf{1} = \max_{\pi \in \Pi} (\alpha r_\pi + \beta \mathbf{1} + \gamma P_\pi (\alpha v^* + c \mathbf{1})) = \max_{\pi \in \Pi} (\alpha r_\pi + \beta \mathbf{1} + \alpha \gamma P_\pi v^* + c \gamma \mathbf{1}),
> $$
>
> where the last equality is due to the fact that $ P_\pi \mathbf{1} = \mathbf{1} $. The above equation can be reorganized as
>
> $$
> \alpha v^* = \max_{\pi \in \Pi} (\alpha r_\pi + \alpha \gamma P_\pi v^*) + \beta \mathbf{1} + c \gamma \mathbf{1} - c \mathbf{1},
> $$
>
> which is equivalent to
>
> $$
> \beta \mathbf{1} + c \gamma \mathbf{1} - c \mathbf{1} = 0.
> $$
>
> Since $ c = \beta/(1-\gamma) $, the above equation is valid and hence $ v' = \alpha v^* + c \mathbf{1} $ is the solution of (3.9). Since (3.9) is the BOE, $ v' $ is also the unique solution. Finally, since $ v' $ is an affine transformation of $ v^*$, the relative relationships between the action values remain the same. Hence, the greedy optimal policy derived from  $ v' $is the same as that from $ v^* $: $ \arg\max_{\pi \in \Pi} (r_\pi + \gamma P_\pi v') $ is the same as $ \arg\max_\pi (r_\pi + \gamma P_\pi v^*) $.

### 2. 原理分析

最优策略的选择依赖**动作价值（Q值）的相对大小**，而非绝对数值：
- 线性变换后，所有动作的Q值会同步进行缩放和偏移（例如原Q值为[1,2,1]，变换后可能为[100,200,100]），但“最大值对应的动作”始终不变，因此策略不会改变。
- 示例验证：
  - 原奖励规则：边界/禁止区r=-1，目标区域r=+1，其他步骤r=0。
  - 线性变换（r'=r+1）：边界/禁止区r'=0，目标区域r'=2，其他步骤r'=1。
  - 结果：最优策略完全一致，仅状态价值数值发生变化（如原v=5，变换后$$v' = 5 + \frac{1}{1-0.9} = 15$$）。


## 四、最优策略的“避绕性”：为何不做无意义绕路？
### 1. 问题背景
若“非目标/非禁止区域”的奖励r=0（无步数惩罚），理论上智能体可选择绕路（如s1→s2→s1→目标），但最优策略仍倾向最短路径，核心原因与折扣因子（γ）相关。

### 2. 核心原因：折扣因子的“时间惩罚”效应
绕路会延迟智能体到达目标区域的时间，导致“目标奖励”的折扣次数增加：
- 假设γ=0.9，最短路径2步到达目标，目标奖励折后为$$1 \cdot \gamma^2 = 0.81$$；若绕路4步到达，目标奖励折后为$$1 \cdot \gamma^4 \approx 0.656$$，后者收益显著更低。
- 结论：即使无明确“步数惩罚”，γ的存在也会对“延迟奖励”产生天然的折扣效应，使最优策略倾向选择最短路径，避免无意义绕路。


## 五、贝尔曼最优公式的基础性质（回顾与补充）
1. **解的存在性与唯一性**：
   - 最优状态价值（v*）：在马尔可夫决策过程（MDP）框架下，由**压缩映射定理（Contraction Mapping Theorem）** 可证明其**存在且唯一**。
   - 最优策略（π*）：对应最优状态价值（v*）的最优策略**不一定唯一**，可能存在多个策略能使状态价值达到v*。
2. **求解方法**：已介绍的迭代类算法（值迭代、策略迭代）可通过反复更新状态价值（v）和策略（π），逐步收敛到最优解（v*和π*）。


## 六、关键总结
1. 最优策略由**奖励函数（r）、折扣因子（γ）、系统模型（P）** 共同决定，其中r和γ是人工可调控的核心参数，需根据任务目标设计。
2. 折扣因子（γ）控制智能体“短视/远视”特性：γ越大，智能体越重视长期收益；γ越小，越关注即时收益。
3. 奖励的线性变换不改变最优策略，仅影响状态价值数值，可利用此特性简化奖励设计（如将负奖励调整为非负，降低计算复杂度）。
4. 折扣因子（γ）具有天然的“反绕路”作用，无需额外设计“步数惩罚”，即可引导智能体选择最短路径。