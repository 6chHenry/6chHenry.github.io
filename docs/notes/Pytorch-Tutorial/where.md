# torch.where

## **1. 基本语法**
```python
torch.where(condition, x, y)
```
- `condition`：布尔条件（`torch.BoolTensor`），决定选择 `x` 还是 `y`。
- `x`：如果 `condition` 为 `True`，则选择 `x` 对应位置的元素。
- `y`：如果 `condition` 为 `False`，则选择 `y` 对应位置的元素。

**返回值**：
- 一个新的张量，其元素来自 `x` 或 `y`，取决于 `condition`。

---

## **2. 使用示例**
### **示例 1：基本条件选择**
```python
import torch

a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([10, 20, 30, 40])

condition = torch.tensor([True, False, True, False])

result = torch.where(condition, a, b)
print(result)  # 输出: tensor([ 1, 20,  3, 40])
```
**解释**：
- `condition` 为 `True` 的位置选择 `a` 的值（`1` 和 `3`）。
- `condition` 为 `False` 的位置选择 `b` 的值（`20` 和 `40`）。

---

### **示例 2：基于张量的条件**
```python
x = torch.tensor([[1, 2], [3, 4]])
y = torch.tensor([[10, 20], [30, 40]])

condition = x > 2  # 返回布尔张量
print(condition)
# tensor([[False, False],
#         [ True,  True]])

result = torch.where(condition, x, y)
print(result)
# tensor([[10, 20],
#         [ 3,  4]])
```
**解释**：
- `x > 2` 的位置选择 `x` 的值（`3` 和 `4`）。
- 其余位置选择 `y` 的值（`10` 和 `20`）。

---

### **示例 3：仅用 `condition`（类似 `np.where(condition)`）**
如果只传入 `condition`，`torch.where()` 会返回满足条件的 **索引**（类似于 `np.where(condition)`）：
```python
a = torch.tensor([1, 2, 3, 4, 5])
indices = torch.where(a > 3)
print(indices)  # 输出: (tensor([3, 4]),)
```
**解释**：
- 返回 `a > 3` 的索引位置（`3` 和 `4`）。

---

### **示例 4：用于替换特定值**
```python
x = torch.tensor([1, -2, 3, -4])
# 将负数替换为 0
result = torch.where(x < 0, torch.tensor(0), x)
print(result)  # 输出: tensor([1, 0, 3, 0])
```
**解释**：
- `x < 0` 的位置替换为 `0`，否则保留原值。

---

## **3. 实际应用场景**
### **(1) 三元组损失（Triplet Loss）中的难样本挖掘**
在 `TripletLoss` 中，可以用 `torch.where` 替代 `mask` 操作：
```python
# 原始代码：
dist_ap = dist[i][mask[i]].max()
dist_an = dist[i][mask[i] == 0].min()

# 用 torch.where 实现：
dist_ap = torch.where(mask[i], dist[i], -torch.inf).max()  # 仅计算正样本
dist_an = torch.where(~mask[i], dist[i], torch.inf).min()  # 仅计算负样本
```

### **(2) 梯度裁剪**
```python
grad = torch.randn(5)
clipped_grad = torch.where(grad > 1.0, torch.tensor(1.0), grad)
```

### **(3) 动态调整学习率**
```python
lr = torch.where(epoch < 10, 0.1, 0.01)  # 前 10 轮 lr=0.1，之后 lr=0.01
```

---

## **4. 注意事项**
1. **`x` 和 `y` 的形状必须相同**（或可广播）。
2. **`condition` 必须是 `torch.BoolTensor`**（可以用 `>`, `<`, `==` 等生成）。
3. **`torch.where(condition)` 返回的是索引**，而不是值。

---

## **总结**
| 用法 | 说明 |
|------|------|
| `torch.where(cond, x, y)` | 类似 `if-else`，选择 `x` 或 `y` |
| `torch.where(cond)` | 返回 `cond` 为 `True` 的索引 |
| 适用场景 | 条件选择、掩码操作、动态调整参数 |

`torch.where()` 在 PyTorch 中非常灵活，可以用于 **条件计算、索引查找、动态调整张量值** 等任务。
