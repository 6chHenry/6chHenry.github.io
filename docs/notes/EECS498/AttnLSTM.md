分为两个部分：

- Attention模块（类）
- 带Attention的LSTM解码器

# Attn 模块

## 定义

__init__参数：feature_dim（图像特征的维度，由预训练的CNN图像特征提取器提供），hidden_dim(LSTM隐藏状态的维度),attn_dim(自己设定的Attn内部计算空间的维度)

 `encoder_att`：把CNN提取出来的图像特征变换到attention空间。

`decoder_att`：把LSTM当前的隐藏状态也映射到attention空间。

再经过一个线性层，把每个位置的attention向量映射成一个标量，表示这个位置的重要性分数。

## 前向传播

参数：`features`：图像特征 (B, num_pixels, feature_dim)，比如(32, 196, 2048)

`hidden`：当前时刻LSTM的隐藏状态 (B, hidden_dim)

`att1`：每个图像位置的特征转成attention空间。

`att2`：把LSTM隐藏状态转成attention空间，并扩展一个维度，让它可以加到每个位置上。

att = F.relu(att1 + att2)
解释：

相加 → 表示当前语言上下文和图像每个位置的“匹配程度”。

加ReLU → 加一点非线性，通常能让模型表现更好。

`self.full_att(att)`：每个位置得到一个打分。

`softmax`：让打分归一化成概率分布（所有位置的权重加起来=1），表示注意力权重。

`features * alpha`：对图像特征加权求和，得到一个**上下文向量**（context vector）。

这个上下文是当前时间步 LSTM 需要关注的视觉信息。

# Decoder With Attention模块

## 定义

`embed_dim`：单词的嵌入向量维度。

`hidden_dim`：LSTM隐藏状态维度。

`vocab_size`：词汇表大小（比如10000）。

`feature_dim`和`attention_dim`：跟Attention模块一样。

把单词ID变成连续向量表示。-->输入：单词embedding 和 上下文向量（concatenate在一起）。输出：新的隐藏状态。-->把隐藏状态映射到每个单词的打分（未归一化），用于最后选词。-->用图像整体信息来初始化LSTM的初始隐藏状态`h0`和记忆单元`c0`。

## 前向传播

`features`：(B, num_pixels, feature_dim)  图像特征。

`captions`：(B, T)  单词ID序列。

- 把每一个单词的ID变成向量。-->用图像整体特征初始化LSTM的初始状态。-->准备收集每一个时间步的输出。-->每一步解码一个单词（循环）:当前的隐藏状态h，指导Attention模块，提取到的上下文特征context。把当前单词embedding 和 上下文向量拼接起来，作为LSTM的输入。输入到LSTM，更新隐藏状态和记忆单元。把LSTM隐藏状态映射到词表上，得到每个单词的打分。保存每一步的输出结果。

  循环结束后：把每步的输出沿着时间轴拼接起来，形成完整的预测序列。

## 初始化隐藏状态

- 对所有图像位置取平均，得到整体特征。
- 分别经过线性变换，得到LSTM的初始隐藏态和记忆态。