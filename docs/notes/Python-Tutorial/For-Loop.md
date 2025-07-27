# Pythonic

## 原来我可以少些这么多For-Loop

List Comprehension: `lst = [i for i in range(100)]`

Generator: `(i for i in range(100))`,含有`yield()`方法

使用built-in函数会更方便，同时，可读性更强

`any(i > 10 for i in range(100))` : Syntax Sugar for `any((i > 100 for i in range(100)))`

`all()`同理

注意：如果写 `any([i > 10 for i in range(100)])`会慢不少，因为List Comprehension会把所有的列表元素全部拿出来，而写成Generator的格式会更快，因为Generator是一个一个生成的（惰性的），只要找到一个不满足的立刻返回False。

如果要写一个满足`good()`函数的List Comprehension，除了`[i for i in slt if good(i)]`

还可以用内置函数`filter(good,lst)`，不过这个会返回Generator而不是一个List。所以说要在外面套一个`list()`

`map(Function,List)`：按照Function的对应法则映射List中的所有元素。返回一个Generator。还有一个好处：如果Function需要多个参数时，map可以接受多个参数，长度取第一个List的长度。

`zip(Lst1,Lst2)`：将Index相同的两个Lisat中的元素打包，新的生成器中的每个元素都是一个Tuple。
