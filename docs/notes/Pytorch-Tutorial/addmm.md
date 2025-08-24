# torch.addmm

$out = \beta \times mat + \alpha \times (mat1 @ mat2)$

换句话说，就是需要传入5个参数，mat里的每个元素乘以beta，mat1和mat2进行矩阵乘法（左行乘右列）后再乘以alpha，最后将这2个结果加在一起。

$\beta,\alpha$默认为1

```python

a = torch.addmm(input,mat1,mat2)
b = input.addmm(mat1,mat2)

# In-place
inputs.addmm_(1,-2,mat1,mat2)
# inputs =  1 * inputs - 2 * (mat1 @ mat2)


```