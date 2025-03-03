一元微积分中的导数与微分的关系: $df = f'(x)dx$

多元微积分中的梯度(列向量)与微分的关系: $df = \sum_{i=1}^n \frac{\partial f}{\partial x_i}dx_i=\frac{\partial f}{\partial \textbf{x}}^Td\textbf{x}$ 第一个等号是全微分公式，第二个等号表达了梯度向量($n \times 1$)与微分向量($n \times 1$)的**内积**是全微分。由此，我们知道$df=\sum_{i=1}^m\sum_{j=1}^n\frac{\partial f}{\partial X_{ij}}dX_{ij} = \tr(\frac{\partial f}{\partial X}^TdX)$ 

第一个等号是全微分公式，第二个等号建议在纸上推导一遍，因为$\tr(A^TB)=\sum_{i,j}A_{ij}B_{ij}=\sum_{i=1}^m\sum_{j=1}^nA_{ij}B_{ij}$ 所以说照葫芦画瓢能够得到。（这里可以侧面说明线性代数中定义$tr(A^TB)$为矩阵**内积**的合理性）

 创建矩阵微分的运算法则[Click here](assets/矩阵求导.pdf)

