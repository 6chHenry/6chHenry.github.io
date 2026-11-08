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

## unique() & nunique()

`df["stock_id"].unique()`可以去重，`df["stock_id"].nunique()`计算独特值的个数

## `parse_dates` 

在 `pd.read_csv(DATA_PATH, parse_dates=["date"])` 中，`parse_dates` 的作用是**告诉 Pandas 在读取 CSV 时，自动将指定的列从字符串解析为 `datetime64` 类型**。

### 为什么需要它？

CSV 文件是纯文本格式，所有数据本质上都是字符串。如果不指定 `parse_dates`，日期列会被读入为普通字符串（`object` 类型）：

```python
# 不使用 parse_dates
df = pd.read_csv("data.csv")
print(df["date"].dtype)  # object (即字符串)
print(df["date"][0])     # '2023-01-15' (字符串)

# 使用 parse_dates
df = pd.read_csv("data.csv", parse_dates=["date"])
print(df["date"].dtype)  # datetime64[ns]
print(df["date"][0])     # Timestamp('2023-01-15 00:00:00')
```

### 解析为 datetime 后能做什么？

字符串做不到、但 datetime 能做的事情非常多：

```python
# 时间筛选
df[df["date"] > "2023-06-01"]

# 提取年/月/日/星期
df["date"].dt.year
df["date"].dt.month
df["date"].dt.dayofweek

# 时间差计算
df["date"].diff()  # 相邻行的时间间隔

# 按时间重采样
df.set_index("date").resample("M").mean()  # 月度平均

# 时间排序
df.sort_values("date")
```

### `parse_dates` 的几种用法

| 用法 | 含义 |
|------|------|
| `parse_dates=True` | 尝试解析索引列为日期 |
| `parse_dates=["date"]` | 将 `"date"` 列解析为日期（最常用） |
| `parse_dates=["date", "time"]` | 将多列分别解析为日期 |
| `parse_dates=[["date", "time"]]` | 将多列**合并**后解析为一个日期列 |

### 如果自动解析失败怎么办？

当日期格式比较特殊时（如 `15/01/2023` 或中文日期），可以配合 `date_format` 参数（Pandas 2.0+）或读取后手动转换：

```python
# 方法1: 指定格式 (Pandas 2.0+)
df = pd.read_csv("data.csv", parse_dates=["date"], date_format="%d/%m/%Y")

# 方法2: 读入后手动转换
df = pd.read_csv("data.csv")
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
```

### 小结

`parse_dates=["date"]` 本质上等价于读入后执行 `df["date"] = pd.to_datetime(df["date"])`，只是更简洁，且在读取阶段就完成了类型转换。

## `cumcount()` 

`g.groupby("stock_id").cumcount()` 的作用是：**在每个分组内，给每一行一个从 0 开始的递增编号**。

### 直观例子

假设数据如下：

```python
import pandas as pd

df = pd.DataFrame({
    "stock_id": ["AAPL", "AAPL", "AAPL", "GOOG", "GOOG"],
    "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-01", "2023-01-02"],
    "price": [150, 152, 151, 100, 102]
})

df["i"] = df.groupby("stock_id").cumcount()
```

结果：

| stock_id | date | price | **i** |
|----------|------|-------|-------|
| AAPL | 2023-01-01 | 150 | **0** |
| AAPL | 2023-01-02 | 152 | **1** |
| AAPL | 2023-01-03 | 151 | **2** |
| GOOG | 2023-01-01 | 100 | **0** |
| GOOG | 2023-01-02 | 102 | **1** |

每只股票的行都从 0 开始重新计数。

### 关键特点

- **分组内计数**：每个 group 独立从 0 开始
- **按出现顺序**：编号按行在 DataFrame 中的顺序分配（所以通常先 `sort_values` 再 `cumcount`）
- **返回的是 Series**：索引与原 DataFrame 对齐，可以直接赋值为新列

### 典型用途

在场景（股票收益预测）中，`cumcount()` 通常用来表示**时间步索引**——即这是该股票的第几个观测点：

```python
# 先按日期排序，确保 cumcount 反映时间顺序
g = g.sort_values(["stock_id", "date"])
g["i"] = g.groupby("stock_id").cumcount()
# i=0 是该股票最早的记录，i=1 是第二早...
```

这样做的好处：
- 方便做 train/test 按时间切分（如 `i < 200` 训练，`i >= 200` 测试）
- 方便对齐不同股票的时间窗口
- 方便做滚动窗口特征工程

### 与 `rank()` 的区别

```python
# cumcount(): 严格从 0 开始递增，不考虑值
# rank(): 根据某列的值大小排名

df.groupby("stock_id").cumcount()          # 0, 1, 2... (按行顺序)
df.groupby("stock_id")["price"].rank()     # 按 price 大小排名
```

## `transform()` 

`g.groupby("stock_id")["i"].transform("max")` 的作用是：**计算每个分组内 `"i"` 列的最大值，然后把结果"广播"回原 DataFrame 的每一行**。

### 核心概念：`transform` vs `agg`

这是理解 `transform` 的关键对比：

```python
# agg: 每组返回一个值，结果变短
g.groupby("stock_id")["i"].max()
# stock_id
# AAPL    249
# GOOG    199

# transform: 每组返回一个值，但广播回每一行，长度不变
g.groupby("stock_id")["i"].transform("max")
# 0    249
# 1    249
# 2    249   ← AAPL 的每一行都填 249
# 3    199
# 4    199   ← GOOG 的每一行都填 199
```

### 直观例子

```python
df = pd.DataFrame({
    "stock_id": ["AAPL", "AAPL", "AAPL", "GOOG", "GOOG"],
    "i":        [0,      1,      2,      0,      1]
})

df["n_per"] = df.groupby("stock_id")["i"].transform("max")
```

| stock_id | i | **n_per** |
|----------|---|-----------|
| AAPL | 0 | **2** |
| AAPL | 1 | **2** |
| AAPL | 2 | **2** |
| GOOG | 0 | **1** |
| GOOG | 1 | **1** |

每行都能看到"自己这只股票一共有多少个时间步（最大索引）"。

### 你代码中的用途

```python
n_per = g.groupby("stock_id")["i"].transform("max")
```

这里 `n_per` 表示每只股票的**最大时间索引**（即总观测数 - 1）。后续可能用来做：

```python
# 按比例划分 train/test
g["is_train"] = g["i"] <= n_per * 0.8  # 前 80% 训练，后 20% 测试
```

### `transform` 能传什么？

```python
# 字符串：内置聚合函数
.transform("max")
.transform("min")
.transform("mean")
.transform("std")
.transform("sum")
.transform("count")

# 自定义函数：接收一个 Series，返回同长度 Series
.transform(lambda x: (x - x.mean()) / x.std())   # 组内标准化
.transform(lambda x: x.fillna(x.median()))        # 组内中位数填充
```

### 为什么不用 `merge`？

`transform` 本质上是 `groupby` + `agg` + `merge` 的快捷方式：

```python
# 等价写法（但更麻烦）
max_i = g.groupby("stock_id")["i"].max().rename("n_per")
g = g.merge(max_i, on="stock_id")

# 用 transform 一行搞定
g["n_per"] = g.groupby("stock_id")["i"].transform("max")
```

### 一句话总结

> `transform` = 分组计算 + 结果广播回原表每一行，保持 DataFrame 形状不变。
