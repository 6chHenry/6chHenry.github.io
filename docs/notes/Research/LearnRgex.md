# Regex

## 核心概念：原始字符串 (Raw Strings)

在定义正则表达式模式时，强烈建议使用 Python 的**原始字符串**，即在字符串前加上 `r`。

**为什么？** 因为正则表达式中大量使用反斜杠 `\` 来表示特殊字符（如 `\d` 表示数字），而 Python 的普通字符串也使用 `\` 来进行转义（如 `\n` 表示换行）。这就可能导致冲突。

- **普通字符串**：`pattern = "\\d"` (你需要两个反斜杠才能表示一个真正的反斜杠)
- **原始字符串**：`pattern = r"\d"` (所见即所得，`\` 就是 `\`)

使用 `r` 可以让你写的模式更清晰，并避免很多不必要的错误。

## `re` 库最常用的函数

### `re.search(pattern, string)` - 查找第一个匹配项

该函数会扫描整个字符串，找到**第一个**符合模式的匹配项。如果找到，它会返回一个“匹配对象”（Match Object）；如果找不到，则返回 `None`。

```python
import re

text = "My phone number is 415-555-1234, and my friend's is 415-555-9876."
pattern = r"\d{3}-\d{3}-\d{4}" # \d 表示数字, {3} 表示重复3次

match = re.search(pattern, text)

if match:
    print(f"找到了电话号码: {match.group(0)}")
else:
    print("没有找到匹配的号码。")

# 输出:
# 找到了电话号码: 415-555-1234
```

- `match.group(0)` 或 `match.group()` 返回完整匹配的字符串。

### `re.match(pattern, string)` - 从字符串开头匹配

`re.match()` 与 `re.search()` 非常相似，但它只在**字符串的开头**进行匹配。如果字符串的起始位置不符合模式，即使后面有匹配的内容，它也会返回 `None`。

```python
import re

text1 = "123 is a number."
text2 = "A number is 123."
pattern = r"\d+" # \d+ 表示一个或多个数字

match1 = re.match(pattern, text1)
match2 = re.match(pattern, text2)

print(f"匹配text1的结果: {match1.group(0) if match1 else 'None'}")
print(f"匹配text2的结果: {match2.group(0) if match2 else 'None'}")

# 输出:
# 匹配text1的结果: 123
# 匹配text2的结果: None
```

### 3. `re.findall(pattern, string)` - 查找所有匹配项

这是非常有用的一个函数，它会找到字符串中**所有**符合模式的非重叠匹配项，并以**列表（list）** 的形式返回。

```python
import re

text = "Emails: user1@example.com, user2@test.com, user3@domain.org"
pattern = r"[\w.-]+@[\w.-]+\.[\w-]+" # 一个简单的邮件模式

emails = re.findall(pattern, text)
print(emails)

# 输出:
# ['user1@example.com', 'user2@test.com', 'user3@domain.org']
```

### `re.sub(pattern, replacement, string)` - 替换匹配

该函数用于查找并替换。它会找到所有匹配 `pattern` 的子串，并将它们替换为 `replacement`。

```python
import re

text = "Hello, my name is John. Hello John!"
# 将 "John" 替换为 "Jane"
new_text = re.sub(r"John", "Jane", text)
print(new_text)

# 输出:
# Hello, my name is Jane. Hello Jane!

# 移除所有数字
text_no_digits = re.sub(r"\d", "", "My phone is 123-456-7890")
print(text_no_digits)
# 输出: My phone is ---
```

### `re.split(pattern, string)` - 分割字符串

比 `string.split()` 更强大，`re.split()` 可以使用复杂的模式作为分隔符来分割字符串。

```python
import re

text = "apple,banana;orange strawberry"
# 根据逗号、分号或空格进行分割
parts = re.split(r"[,;\s]+", text) # \s 表示空白字符, + 表示一个或多个
print(parts)

# 输出:
# ['apple', 'banana', 'orange', 'strawberry']
```

### `re.compile(pattern)` - 编译正则表达式

如果一个正则表达式需要被重复使用多次，最好先用 `re.compile()` 将它编译成一个“模式对象”（Pattern Object）。这样做可以大大提高执行效率，因为编译只需要进行一次。

```python
import re

# 编译一次模式
phone_pattern = re.compile(r"\d{3}-\d{3}-\d{4}")

text1 = "Call me at 415-555-1234."
text2 = "His number is 212-555-9876."

# 多次使用编译好的模式对象
match1 = phone_pattern.search(text1)
match2 = phone_pattern.search(text2)

if match1:
    print(f"Text1中找到: {match1.group(0)}")
if match2:
    print(f"Text2中找到: {match2.group(0)}")

# 输出:
# Text1中找到: 415-555-1234
# Text2中找到: 212-555-9876
```



## 正则表达式基本元字符

| 字符 | 描述                                    | 示例            | 匹配                    |
| ---- | --------------------------------------- | --------------- | ----------------------- |
| .    | 匹配除换行符外的任意单个字符            | a.c             | "abc", "a_c", "a2c"     |
| \d   | 匹配任意一个数字 (0-9)                  | \d+             | "123", "4"              |
| \D   | 匹配任意一个非数字                      | \D              | "a", "_", " "           |
| \w   | 匹配字母、数字、下划线 (word character) | \w+             | "hello", "user_123"     |
| \W   | 匹配非字母、数字、下划线                | \W              | "@", "!", " "           |
| \s   | 匹配任意空白字符 (空格, tab, 换行)      | \s+             | " ", " \t"              |
| \S   | 匹配任意非空白字符                      | \S+             | "word", "!"             |
| ^    | 匹配字符串的开头                        | ^Hello          | "Hello world"           |
| $    | 匹配字符串的结尾                        | world$          | "Hello world"           |
| []   | 字符集，匹配方括号中的任意一个字符      | [aeiou]         | "a", "e", "i", "o", "u" |
| [^]  | 否定字符集，匹配不在括号中的任意字符    | [^0-9]          | "a", "b", "c" (非数字)  |
| `    | `                                       | 或运算符        | cat\|dog                |
| ()   | 分组，捕获括号内的模式                  | (\d{3})-(\d{4}) | "555-1234"              |

**量词 (Quantifiers):**

| 字符  | 描述                       |
| ----- | -------------------------- |
| *     | 匹配前面的元素 0 次或多次  |
| +     | 匹配前面的元素 1 次或多次  |
| ?     | 匹配前面的元素 0 次或 1 次 |
| {n}   | 匹配前面的元素恰好 n 次    |
| {n,}  | 匹配前面的元素至少 n 次    |
| {n,m} | 匹配前面的元素 n 到 m      |

## Exercise

```python
pattern = r"([\w.-]+)@([\w.-]+\.[\w-]+)" # 在@前后加上括号

text = "Send your resume to careers@my-startup.com"
match = re.search(pattern, text)

if match:
    print(f"完整邮箱: {match.group(0)}")
    print(f"用户名: {match.group(1)}")
    print(f"域名部分: {match.group(2)}")
    
# 输出：
# 完整邮箱: careers@my-startup.com
# 用户名: careers
# 域名部分: my-startup.com
```

解析：

| 部分    | 含义               | 详细说明                                                     |
| ------- | ------------------ | ------------------------------------------------------------ |
| [\w.-]+ | 第一部分：用户名   | 匹配邮箱的 @ 前面的部分。                                    |
|         | \w                 | 匹配任何“单词字符”（字母 a-z, A-Z，数字 0-9，以及下划线 _）。 |
|         | . 和 -             | 匹配字面上的点号和连字符。                                   |
|         | [...]              | 将以上组合成字符集。                                         |
|         | +                  | 表示这个字符集里的字符可以出现一次或多次。                   |
| @       | 字面字符           | 精确匹配 @ 符号，它是用户名和域名的分界线。                  |
| [\w.-]+ | 第二部分：域名     | 匹配 @ 之后，顶级域名（如 .com）之前的部分。规则和用户名部分一样，可以匹配 gmail 或 sub-domain.example 这样的域名。 |
| \.      | 转义的点           | 这是关键！ . 在正则表达式中是元字符，代表“任何字符”。为了匹配一个真正的点，我们必须用反斜杠 \ 对它进行转义。 |
| [\w-]+  | 第三部分：顶级域名 | 匹配最后的部分，如 com, org, co-uk。                         |
|         | [\w-]              | 注意，这里通常不包含点 .，因为顶级域名本身一般不带点。       |
|         | +                  | 出现一次或多次。                                             |



```python
# 模式：http(s)://(域名和路径)
url_pattern = r"https?://[\w./-]+"
# 解析：
# http  -> 匹配字面上的 "http"
# s?    -> "s" 是可选的（? 代表0次或1次），所以 http 和 https 都能匹配
# ://   -> 匹配字面上的 "://"
# [\w./-]+ -> 匹配由单词字符、点、斜杠、连字符组成的任意长序列

web_text = "Visit our site at http://example.com/home or the secure portal https://portal.my-company.com/login."
urls = re.findall(url_pattern, web_text)
print(urls)

# 输出：
# ['http://example.com/home', 'https://portal.my-company.com/login']
```

