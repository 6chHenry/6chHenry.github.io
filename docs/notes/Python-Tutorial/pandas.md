# Pandas库

## Series：带标签的一维数组

Series 类似于 NumPy 的一维数组，但有索引（标签）。

```python
s = pd.Series([10,20,30],index=['a','b','c'])
print(s)
print(s['a']) #类似字典一样访问值
```

## DataFrame

DataFrame 是 Pandas 最核心的数据结构，可以类比为一个带行列标签的二维表格（就像 Excel 表）。你可以把它想成是多个 Series 按列组合而成。

```python
data = {
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [25, 30, 35],
    'Score': [85.5, 90.0, 95.5]
}

df = pd.DataFrame(data)
print(df)
```

你可以像字典一样取列：

```python
print(df['Name'])  # 输出 Name 列（一个 Series）
```

也可以按行索引：

```python
print(df.loc[1])  # 按标签索引第 1 行
"""
name      Bob
age        30
Score    90.0
Name: 1, dtype: object
"""
print(df.iloc[2])  # 按位置索引第 2 行
```

!!! info

    loc 用名字（像字典）

    iloc 用数字（像数组）

    loc[a:b] 会包含 b，iloc[a:b] 不会包含 b

> **提示**  
> - `loc` 用名字（像字典）  
> - `iloc` 用数字（像数组）  
> - `loc[a:b]` 包含 `b`，`iloc[a:b]` 不包含 `b`

```python
import pandas as pd

data = {
    'Name': ['Tom', 'Lily', 'Jack'],
    'Math': [88, 95, 70],
    'English': [78, 85, 90]
}
df = pd.DataFrame(data, index=['s1', 's2', 's3'])
# 访问标签范围（包括末尾）
print(df.loc['s1':'s2'])

# 访问第 0~1 行（不含第 2 行）
print(df.iloc[0:2])
```

| 属性 / 方法       | 作用                    |
| ------------- | --------------------- |
| `.shape`      | 返回形状（行数，列数）           |
| `.columns`    | 返回所有列名（Index 对象）      |
| `.index`      | 返回所有行索引               |
| `.dtypes`     | 返回每一列的数据类型            |
| `.head(n)`    | 查看前 n 行数据（默认前 5 行）    |
| `.tail(n)`    | 查看后 n 行数据             |
| `.info()`     | 查看总体结构（非空数、类型、内存）     |
| `.describe()` | 数值列的汇总统计（均值、标准差、最大最小） |

## 按条件筛选DataFrame的行

```python
import pandas as pd

data = {
    'Name': ['Tom', 'Lily', 'Jack', 'Lucy', 'Eric'],
    'Math': [88, 95, 70, 82, 91],
    'English': [78, 85, 90, 88, 76],
    'Age': [20, 22, 21, 23, 21],
    'Gender': ['M', 'F', 'M', 'F', 'M']
}
df = pd.DataFrame(data)
```

我们要筛选Math 成绩大于 80 且 Age 小于 23 的学生：

```python
# 筛选 Math > 80 且 Age < 23
filtered_df = df[(df['Math'] > 80) & (df['Age'] < 23)]
print(filtered_df)
```

```javascript
    Name  Math  English  Age Gender
1   Lily    95       85   22      F
4   Eric    91       76   21      M

```

注意：

- 条件之间要用 &（与）和 |（或）连接。

- 每个条件要用圆括号括起来。

- 比较时，可以使用常见的运算符：==, !=, <, >, <=, >=。

## 选择特定列，添加，删除列

```txt
- 单列选择：df['column_name']
- 多列选择：df[['col1', 'col2']]
- 添加列：df['new_column'] = value
- 删除列，返回副本：df.drop('column_name', axis=1)
- 删除列，在原始数据上修改：df.drop('column_name', axis=1,inplace=True)
```

## DataFrame 的排序（按值或按索引排序）

| 方法名             | 作用说明        |
| --------------- | ----------- |
| `sort_values()` | 按列的值排序（最常用） |
| `sort_index()`  | 按行或列的索引排序   |

```python
import pandas as pd

data = {
    'Name': ['Tom', 'Lily', 'Jack', 'Lucy', 'Eric'],
    'Math': [88, 95, 70, 82, 91],
    'English': [78, 85, 90, 88, 76],
    'Age': [20, 22, 21, 23, 21]
}

df = pd.DataFrame(data)
```

按列值排序（`sort_values`）

### 单列排序

```python
# 按 Math 升序排序
df.sort_values(by='Math')
```

```python
# 按 Age 降序排序
df.sort_values(by='Age', ascending=False)
```

### 多列排序

```python
# 先按 Age 升序，再按 English 降序
# 如果同 Age 才按照 English 降序
df.sort_values(by=['Age', 'English'], ascending=[True, False])
```

---

按索引排序（`sort_index`）

### 按行索引排序

```python
df.sort_index()
```

### 按列名排序（横向）

```python
df.sort_index(axis=1)
```

---

## 缺省值处理

- isna() / isnull()：检查缺失值
- dropna()：删除含 NaN 的行或列
- fillna()：填充缺失值（0、前值、后值、特定值等）
- dropna(thresh=...)：根据非 NaN 值的个数删除行或列
- fillna(method='ffill'/'bfill')：前后值填充
- 按列使用不同的填充值：
df_filled_col = df.fillna({'Math': 80, 'English': 85})

## 分组操作

| 方法名              | 作用说明                                          |
|--------------------|---------------------------------------------------|
| `groupby()`         | 将数据按某些列进行分组                             |
| `agg()`             | 聚合操作，可以对每个组应用多个聚合函数             |
| `sum()`             | 按组求和                                          |
| `mean()`            | 按组计算均值                                      |
| `count()`           | 按组计算非 NaN 值的个数                           |
| `transform()`       | 对每个组进行变换，返回与原数据结构相同的 DataFrame |

```python

# 按 Gender 列分组
grouped = df.groupby('Gender')

# 查看每个组的内容
for name, group in grouped:
    print(name)
    print(group)
#聚合操作（agg()）
#计算每个性别组的数学成绩平均值和总和

# 按 Gender 分组，聚合 Math 和 English 列
grouped_agg = grouped.agg({
    'Math': ['mean', 'sum'],
    'English': 'mean'
})
print(grouped_agg)
```
