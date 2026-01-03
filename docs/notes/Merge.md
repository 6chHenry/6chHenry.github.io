
少年，你已经掌握了分治法的精髓！
主函数里有两个副函数，一个是merge_sort，一个是helper.
helper将问题分为三个部分，左区间，右区间以及跨区间
merge_sort接收传入的左区间和右区间，在这个过程中先计算跨区间的满足题意的序列对数，再进行归并排序。
helper中接受merge_sort返回的排序后的数组以及跨区间对数，最后将左中右的总数加起来并返回。
个人认为难点在于如何实现快速计算跨区间的满足题意的序列对数。这里主要要利用每个子区间的单调性以简化计算（每次不需要重头开始，而是在前面已有的数字上累加，类似前缀和，因为如果这个数满足某个条件，则在此之前/后的所有数也满足这个条件）
这里贴几个：
1. $i < j$
$nums[i] > 2 \times nums[j]$
```python
for num in left:
            while j <len(right) and num > 2 * nums [j]:
                j += 1
            count += j
```
2. `i < j`（日期 `i` 在日期 `j` 之前）；
- `profits[j] < 0`（后一天为亏损）；
- `|profits[j]| > 2 * profits[i]`（后期亏损绝对值超过前期盈利的两倍）。
这道题是有难度的变式，因为它还要求后面的数小于0并且绝对值后面大于前面。如果我们仍然按照全部升序来排列可能会出错。对于赚了天数，可以用升序归并。但是对于亏了的天数，我们知道，我们希望绝对值是越大越好的，也就是对应原来的数值（负数）是越来越小的，就应该是降序。并且这里遍历的是右半区间（当然也可以遍历左半区间，但我们更熟悉正数，因此吧threshold调整为正数比大小更容易）
```python
for loss in right_losses:
            threshold = -loss / 2  # 盈利值需小于该阈值
            # 移动 p 直到 left_profits[p] >= threshold
            while p < len(left_profits) and left_profits[p] < threshold:
                p += 1
            # 此时 p 是第一个不满足条件的盈利日索引，前面的都满足
            cross_cnt += p
```
3.$i < j$
$nums[i] > nums[j]$
这里可以直接在归并的时候计算，因为我们排序是严格按照递增的，也就是key=lambda x:x，而之前的变式题目，我们需要计算的对满足的关系并非简单的lambda x:x，所以无法在归并过程中一起计算！
```python
while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                cross_inversions += len(left) - i # 如果right[j]小于left[i]，则left[i]及其后面的元素都形成逆序对
                j += 1
```
# 翻转对 (Reverse Pairs)
题目描述
给定一个数组 nums，我们称 (i, j) 为一个重要翻转对，如果满足以下条件：
$i < j$
$nums[i] > 2 \times nums[j]$
请返回给定数组中的重要翻转对的数量。

示例
输入：[1, 3, 2, 3, 1]
输出：2

代码：
```python
import sys
import ast

def reverse_pair(nums):
    def merge_sort(left, right):
        if left >= right:
            return 0
        mid = (left + right) // 2
        count = merge_sort(left,mid) + merge_sort(mid+1,right)
        j = mid + 1
        for i in range(left,mid+1):
            while j <= right and nums[i] > 2 * nums[j]:
                j += 1
            count += j - (mid + 1)
        merged = []
        i , j = left , mid + 1
        while i <= mid and j <= right:
            if nums[i] <= nums[j]:
                merged.append(nums[i])
                i += 1
            else:
                merged.append(nums[j])
                j += 1
        merged.extend(nums[i:mid+1])
        merged.extend(nums[j:right+1])
        nums[left:right+1] = merged
        return count
    return merge_sort(0,len(nums)-1)
if __name__ == "__main__":
    nums = ast.literal_eval(sys.stdin.readline().strip())
    nums = list(nums)
    ans = reverse_pair(nums)
    print(ans)
```


# 重大亏损风险对

描述：

## 题目描述

在金融风险管理中，我们不仅关注单日的亏损，更关注“收益高点后的断崖式下跌”。给定一个整数数组 `profits`，表示每日的资金净变动（正数表示盈利，负数表示亏损）。

我们定义一个“重大风险对”为满足以下条件的下标对 `(i, j)`：

- `i < j`（日期 `i` 在日期 `j` 之前）；
- `profits[j] < 0`（后一天为亏损）；
- `|profits[j]| > 2 * profits[i]`（后期亏损绝对值超过前期盈利的两倍）。

换句话说，若后期某一天亏损额超过前期的一天盈利的两倍，则认为该对 `(i, j)` 是一个重大风险对。请返回数组中所有满足条件的下标对数量。

## 示例

示例 1：

1. `输入：profits = [1, -3, 2, -2]`
2. `输出：1`
3. `解释：`
4. `有效的重大风险对为 (i=0, j=1)。`
5. `对于 j=3（profits[3] = -2），任意 i < 3 的 profits[i] 都不满足 2 > 2 * profits[i]，因此不计入。`

输入示例： 1, -3, 2, -2 输出示例： 1

```python
def emergencypair(profits):
    n = len(profits)
    def solve(l, r):
        if r - l == 1:  
            if profits[l] > 0:          # 盈利日
                return 0, [profits[l]], []
            elif profits[l] < 0:        # 亏损日
                return 0, [], [profits[l]]
            else:                       # 0 忽略
                return 0, [], []
        mid = (l + r) // 2
        left_cnt, left_profits, left_losses = solve(l, mid)
        right_cnt, right_profits, right_losses = solve(mid, r)
        # 计算跨越左右的风险对数量（核心）
        cross_cnt = 0
        p = 0  # 左盈利列表指针
        # 遍历右亏损列表（降序排列，即亏损从大到小）
        for loss in right_losses:
            threshold = -loss / 2  # 盈利值需小于该阈值
            # 移动 p 直到 left_profits[p] >= threshold
            while p < len(left_profits) and left_profits[p] < threshold:
                p += 1
            # 此时 p 是第一个不满足条件的盈利日索引，前面的都满足
            cross_cnt += p
        total_cnt = left_cnt + right_cnt + cross_cnt
        # 合并盈利日列表（升序）
        merged_profits = []
        i, j = 0, 0
        while i < len(left_profits) and j < len(right_profits):
            if left_profits[i] <= right_profits[j]:
                merged_profits.append(left_profits[i])
                i += 1
            else:
                merged_profits.append(right_profits[j])
                j += 1
        merged_profits.extend(left_profits[i:])
        merged_profits.extend(right_profits[j:])
        # 合并亏损日列表（降序，保证亏损绝对值小的在前）
        merged_losses = []
        i, j = 0, 0
        while i < len(left_losses) and j < len(right_losses):
            # 注意：降序排列，所以较大的（即亏损较小的）在前
            if left_losses[i] >= right_losses[j]:
                merged_losses.append(left_losses[i])
                i += 1
            else:
                merged_losses.append(right_losses[j])
                j += 1
        merged_losses.extend(left_losses[i:])
        merged_losses.extend(right_losses[j:])
        return total_cnt, merged_profits, merged_losses
    return solve(0, n)[0]
if __name__ == "__main__":
    profits = list(map(int,input().split()))
    pairs = emergencypair(profits)
    print(pairs)
```

#  信号衰减异常对（分治）

题目描述：

在无线通信中，我们需要监测信号的异常衰减。给定一个时间序列数组 power，表示每日的信号发射强度。我们定义一个“异常衰减对”为满足以下条件的下标对 (i, j)：

- i < j（天数 i 在 j 之前）；
- power[i] > 3 \times power[j]（前期的信号强度超过后期强度的三倍）。
    请计算序列中“异常衰减对”的总数。

**输入描述：**

- 输入一行整数表示 `power` 数组。
**输出描述：**

- 输出一个整数表示异常对数量。


示例：

输入：

10 2 5 3 1

输出：

4（解释：(10,2), (10,3), (10,1), (5,1) 共4对）

```python
def abnormal_pair(power):
    abnormal = []
    def merge(left,right):
        merged = []
        j = 0
        count = 0
        for num in left:
            while j < len(right) and num > 3 * right[j]:
                abnormal.append((num,right[j]))
                j += 1
            count += j
        j = 0
        i = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
        merged.extend(left[i:])
        merged.extend(right[j:])
        return merged,count
    def merge_sort_helper(arr):
        if len(arr) < 2:
            return arr , 0
        mid = len(arr) // 2
        left_merge , left_count = merge_sort_helper(arr[:mid])
        right_merge , right_count = merge_sort_helper(arr[mid:])
        merged , cross_count = merge(left_merge,right_merge)
        count = left_count + right_count + cross_count
        return merged , count
    return merge_sort_helper(power),abnormal

if __name__ == "__main__":
    power = list(map(int,input().split()))
    print(abnormal_pair(power))
```
# 逆序对数目

```python
def merge_sort_with_inversion_count(arr):

    """

    归并排序并统计逆序对数量

    参数: arr - 待排序的数组

    返回: (sorted_arr, inversion_count) - 排序后的数组和逆序对数量

    """

    def merge(left, right):

        """

        合并两个有序数组，并统计跨越逆序对

        参数: left, right - 两个有序数组

        返回: (merged_array, cross_inversions) - 合并后的数组和跨越逆序对数量

        """

        # TODO: 实现合并逻辑，同时统计逆序对

        merged = []

        i = j = 0

        cross_inversions = 0

        while i < len(left) and j < len(right):

            if left[i] <= right[j]:

                merged.append(left[i])

                i += 1

            else:

                merged.append(right[j])

                cross_inversions += len(left) - i # 如果right[j]小于left[i]，则left[i]及其后面的元素都形成逆序对

                j += 1

        merged.extend(left[i:])

        merged.extend(right[j:])

        return merged, cross_inversions

    # 1  2  3 10

    # 4 6  7

    def merge_sort_helper(arr):

        """

        递归实现归并排序并统计逆序对

        参数: arr - 待排序的数组

        返回: (sorted_arr, inversion_count) - 排序后的数组和逆序对数量

        """

        # TODO: 实现归并排序的递归逻辑

        if len(arr) < 2:

            return arr , 0

        mid = len(arr) // 2

        left, left_inv = merge_sort_helper(arr[:mid])

        right, right_inv = merge_sort_helper(arr[mid:])

        merged, cross_inv = merge(left, right)

        total_inv = left_inv + right_inv + cross_inv

        return merged, total_inv

    return merge_sort_helper(arr)
```