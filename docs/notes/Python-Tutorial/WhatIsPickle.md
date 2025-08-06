# Pickle

pickle实际上是一个字典，并且可以多层嵌套。

# 通过类别转换深刻认识Pickle文件

## 图片转pickle

如果直接将一个图像文件采用Python内置的read()方法，如以二进制形式读取：

```python
with open('image.jpg', 'rb') as f:
    raw_data = f.read()
```

raw_data是一个包含图像文件原始字节的bytes对象

注：什么是bytes对象？

不可修改！每个byte存储0~255的整数。可以直接用`b'hello'`(Ascii编码)或`b'\x41\x42'`(十六进制，相当于`b'AB'`)或从整数数组生成

可以将字符串编码为bytes对象：`text="你好"  b_text = text.encode('utf-8')`

将bytes对象转换成字符串: `b = b'\xe4\xbd\xa0\xe5\xa5\xbd'  text = b.decode('utf-8')  # "你好"`


一些用法：索引和切片

```python
b = b'ABCD'
print(b[0])    # 65（返回整数字节值）
print(b[1:3])  # b'BC'（返回新的 bytes 对象）
```



如果直接print出来，是以b开头的，并不直接描述图片像素，而还包括了一些信息，比如创建这张图的作者是谁



PIL: (行，列)

Numpy：（列，行）

可以写一个简单的Image2Pickle函数来测试对上述知识的理解。