# torch.einsum

torch.einsum(equation, *operands) → Tensor

Einsum 可以通过基于爱因斯坦求和约定法的简写格式来计算许多常见的多维线性代数数组操作，格式为 equation 。基本思想是为输入 operands 的每个维度标记下标，并定义哪些下标是输出的一部分。然后通过沿着那些下标不属于输出的维度对 operands 的元素求和来计算输出。例如，矩阵乘法可以使用 einsum 计算，如 torch.einsum("ij,jk->ik", A, B)。这里，j 是求和下标，而 i 和 k 是输出下标

```python
# trace
torch.einsum('ii', torch.randn(4, 4))

# diagonal
torch.einsum('ii->i', torch.randn(4, 4))

# outer product
x = torch.randn(5)
y = torch.randn(4)
torch.einsum('i,j->ij', x, y)

# batch matrix multiplication
As = torch.randn(3, 2, 5)
Bs = torch.randn(3, 5, 4)
torch.einsum('bij,bjk->bik', As, Bs)



# with sublist format and ellipsis
torch.einsum(As, [..., 0, 1], Bs, [..., 1, 2], [..., 0, 2])



# batch permute
A = torch.randn(2, 3, 4, 5)
torch.einsum('...ij->...ji', A).shape

# equivalent to torch.nn.functional.bilinear
A = torch.randn(3, 5, 4)
l = torch.randn(2, 5)
r = torch.randn(2, 4)
torch.einsum('bn,anm,bm->ba', l, A, r)


```

For example:
x = torch.einsum('...ij, ijk -> ...ik', x, self.__weight)

参数分解：

第一个张量 x 的维度模式是 ...ij

... 表示任意数量的额外维度（可能没有）

ij 表示最后两个维度

第二个张量 self.__weight 的维度模式是 ijk

输出张量的维度模式是 ...ik

运算含义：

对第二个张量j维度中的每个值和第一个张量后两个维度构成的矩阵沿j方向逐项相乘然后求和。

等价于先广播再乘。

$C_{*ik} = A_{*ij}B_{ijk}$

$C_{*ik} = \sum_j A_{*ij}B_{ijk}$

(i,j) -> broadcast -> (i,j,1)  -> 在第三维度复制k次 (i,j,k) -> $\odot$（逐元素相乘）(i,j,k) -> 在j维度求和 -> (i,k)

A simple test:

```python
A = torch.Tensor(range(1*2*3*4)).view(1, 2, 3 , 4)
B = torch.Tensor(range(3*4*5)).view(3,4,5)
C = torch.einsum('...ij,ijk->...ik',A , B)
C_ = torch.zeros((1,2,3,5))
p , q , i , j , k = A.shape[0], A.shape[1], A.shape[2], A.shape[3], B.shape[2]
for p_ in range(p):
    for q_ in range(q):
        for i_ in range(i):
            for j_ in range(j):
                for k_ in range(k):
                    C_[p_, q_, i_, k_] += A[p_, q_, i_, j_] * B[i_, j_, k_]
assert C == C_
```

[博客参考](https://iiosnail.blogspot.com/2024/10/einsum.html)

[einsum is all you need](https://www.youtube.com/watch?v=pkVwUVEHmfI)