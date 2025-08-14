# Peter-Clark Algorithm

阶段 0　数据准备  
• 输入：n 个观测样本、变量集 V = {X₁,…,Xₚ}  
• 输出：一个 p×p 的“条件独立性表”I，其中 I(i,j|S)=1 表示在给定集合 S 的条件下 Xi ⊥ Xj。  
（实际实现时用统计检验或阈值化的偏相关系数即可，但手算时可直接给出“上帝视角”的独立性表。）

━━━━━━━━━━━━━━━━━━  
阶段 1　骨架学习（Skeleton Identification）  

1.1 初始化  
 完全无向图 G₀：所有节点两两之间都有边。

1.2 迭代删边（从 k=0 开始）  
 for k = 0,1,2,…,p−2           // k 为待检验条件集 S 的基数  
  for 每条边 (Xi, Xj) 仍在图中  
   for 每一个大小为 k 的子集 S ⊆ adj(G, Xi)\{Xj}  
    若检验 Xi ⊥ Xj | S 成立：  
     删除边 Xi–Xj；  
     记录 SepSet(i,j) = S；  
     break  // 只要找到 1 个分离集即可  
  until 没有边可再删  
 end for  

注意：adj(G, Xi) 表示当前图中 Xi 的邻居。  
复杂度：最坏 O(p²·2^{p−2})，但在稀疏图上远小于此。

━━━━━━━━━━━━━━━━━━  
阶段 2　识别 v-结构（Collider Detection）  

2.1 遍历所有三元组  
 for 所有无向路径 Xi – Xk – Xj 且 Xi 与 Xj 不相邻  
  若 Xk ∉ SepSet(i,j)  
   则将 Xk 标记为对撞节点，定向为 Xi → Xk ← Xj  

━━━━━━━━━━━━━━━━━━  
阶段 3　方向传播（Meek Rules）  

用到 4 条 Meek 规则（按优先级一次扫描即可收敛）：  
R1  若存在 X → Y – Z 且 X 与 Z 不相邻 → Y → Z  
R2  若存在 X → Y → Z 且 X – Z → X → Z  
R3  若存在 X – Y – Z 且存在 X → W ← Z 且 W 与 Y 不相邻 → X → Y ← Z  
R4  若存在 X – Y – Z 且存在 X → W → Z 且 W 与 Y 不相邻 → X → Y → Z  
循环应用直到没有新箭头可添加。  
最终输出：一个完全部分有向无环图（CPDAG）。

━━━━━━━━━━━━━━━━━━  
补充 3 变量示例  

真实结构：A → C ← B  
独立性表：  
• A ⊥ B | ∅：不独立  
• A ⊥ C | ∅：不独立  
• A ⊥ C | {B}：独立 → 删除 A–C；SepSet(A,C)={B}  
• B ⊥ C | ∅：不独立  
• B ⊥ C | {A}：独立 → 删除 B–C；SepSet(B,C)={A}  

骨架：A–B（因为 A ⊥ B | ∅ 不成立）。  
v-结构：对三元组 A – C – B，SepSet(A,B)=∅，而 C 在路径中且 C ∉ SepSet(A,B)，因此定向为 A → C ← B。  
方向传播：无其他边可加方向。  
最终 CPDAG：A → C ← B, A – B（无向）。  

这个 3 变量例子恰好展示了 PC 算法能够正确识别对撞结构 C → A ← B（若检验功效足够）。

还可参考 [【因果系列】PC 算法 —— 一种基于约束的因果发现算法 - 根植星空的文章 - 知乎](https://zhuanlan.zhihu.com/p/452724126)
[PC 算法 - 贝叶斯网络与其结构学习算法 - 西伯利亚大恶龙的文章 - 知乎](https://zhuanlan.zhihu.com/p/368010458)