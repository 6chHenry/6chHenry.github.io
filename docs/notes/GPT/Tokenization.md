一个困扰我很久的点：encode和decode到底是什么意思？

```python
# 原始字符串
original_string = "hello"

# 编码为 UTF-8 字节序列
encoded_bytes = original_string.encode('utf-8')
print(f"编码后的字节序列: {encoded_bytes}")   
print(f"变为列表形式:{list(encoded_bytes)}")
print(f"类型: {type(encoded_bytes)}")

# 解码为字符串
decoded_string = encoded_bytes.decode('utf-8')
print(f"解码后的字符串: {decoded_string}")
print(f"类型: {type(decoded_string)}")
```

输出：

编码后的字节序列: b'hello' 

变为列表形式:[104, 101, 108, 108, 111] (原理：十六进制-->十进制)

类型: <class 'bytes'>

 解码后的字符串: hello 

类型: <class 'str'>

我们看到encode之后还是hello，这是为什么呢？因为hello是英文字符，utf-8和ASCII编码相同，因此电脑上会直接显示英文字符。但是如果我们用中文,比如text=“你好” text.encode(“utf-8”) 就会显示编码

```python
# 原始字符串
original_string = "你好"

# 编码为 UTF-8 字节序列
encoded_bytes = original_string.encode('utf-8')
print(f"编码后的字节序列: {encoded_bytes}")
print(f"变为列表形式:{list(encoded_bytes)}")
print(f"类型: {type(encoded_bytes)}")

# 解码为字符串
decoded_string = encoded_bytes.decode('utf-8')
print(f"解码后的字符串: {decoded_string}")
print(f"类型: {type(decoded_string)}")
```

输出：

编码后的字节序列: b'\xe4\xbd\xa0\xe5\xa5\xbd'
变为列表形式:[228, 189, 160, 229, 165, 189]
类型: <class 'bytes'>
解码后的字符串: 你好
类型: <class 'str'>

而bytes呢？

在 Python 中，`bytes` 类用于表示**不可变的字节序列**。它是一个专门用于处理二进制数据的数据类型。`bytes` 对象可以通过多种方式创建，比如直接调用 `bytes()` 构造函数、使用字节字面量（如 `b'hello'`）或从其他数据类型（如列表、字符串）转换而来。

1.当你使用 `bytes(n)` 这种方式创建 `bytes` 对象时，它会生成一个长度为 `n` 的字节序列，其中每个字节的值默认是 `0`。

2. **从列表创建**

通过传递一个整数列表（每个整数范围是 0-255）来创建 `bytes` 对象。



比如bytes([104, 101, 108, 108, 111] )  会输出b’hello’



那么decode呢？

bytes([158] ).decode("utf-8") 会报错 

**字节值 `158` 不是一个有效的 UTF-8 编码的字节**。

UTF-8 是一种可变长度的字符编码方式，它对字节的取值有严格的规则。具体来说：

- 对于单字节字符（ASCII 字符），字节值的范围是 `0x00` 到 `0x7F`（即 0 到 127）。
- 对于多字节字符，UTF-8 使用特定的字节模式来标识字符的长度和编码。

字节值 `158`（十六进制为 `0x9E`）超出了单字节字符的范围，但它也不符合 UTF-8 的多字节字符的规则，因此会被认为是无效的 UTF-8 编码。

------

### 为什么字节值 `158` 无效？

在 UTF-8 中：

- 多字节字符的第一个字节（起始字节）必须以 `110`、`1110` 或 `11110` 开头（二进制表示）。
- 后续字节必须以 `10` 开头（二进制表示）。

字节值 `158` 的二进制表示是 `10011110`：

- 它不符合起始字节的规则（必须以 `110`、`1110` 或 `11110` 开头）。
- 它也不符合后续字节的规则（必须以 `10` 开头）。

解决方案:bytes([158] ).decode("utf-8",errors="replace") 会用'�'替代

