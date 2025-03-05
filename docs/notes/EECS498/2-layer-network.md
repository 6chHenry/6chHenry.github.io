4. # 用线性代数的知识证明 \( L \) 对 \( W1 \)、\( W2 \)、\( b1 \)、\( b2 \) 的梯度

  为了证明损失函数 \( L \) 对权重 \( W1 \)、\( W2 \) 和偏置 \( b1 \)、\( b2 \) 的梯度，我们可以使用链式法则（Chain Rule）和线性代数的知识。以下是详细的推导过程。

  ---

  ### **1. 定义符号和网络结构**

  假设网络结构如下：

  1. **输入层**：输入数据 \( X \)，维度为 \( N $\times$ D \)（\( N \) 是样本数量，\( D \) 是特征维度）。
  2. **隐藏层**：
     - 权重 \( W1 \)，维度为 \( D $\times$ H \)（\( H \) 是隐藏层神经元数量）。
     - 偏置 \( b1 \)，维度为 \( H \)。
     - 激活函数为 ReLU：\( h1 = $\text{ReLU}(X \cdot W1 + b1) $\)，维度为 \( N $\times$ H \)。
  3. **输出层**：
     - 权重 \( W2 \)，维度为 \( H $\times$ C \)（\( C \) 是类别数量）。
     - 偏置 \( b2 \)，维度为 \( C \)。
     - 输出 logits：\( scores = h1 $\cdot$ W2 + b2 \)，维度为 \( N $\times$ C \)。
  4. **Softmax 和损失函数**：
     - Softmax 输出：\( probs = $\text{Softmax}(scores) $\)。
     - 交叉熵损失：\( $L = -\frac{1}{N} \sum_{i=1}^N \log(probs_{i, y_i}) $\)，其中 \( y_i \) 是第 \( i \) 个样本的真实标签。

  ---

  ### **2. 计算梯度**

  我们需要计算 \( L \) 对 \( W1 \)、\( W2 \)、\( b1 \)、\( b2 \) 的梯度。

  #### **(1) 梯度 \($$ \frac{\partial L}{\partial W2} ) 和 ( \frac{\partial L}{\partial b2}$$ \)**

  - 根据链式法则：
    \[
    $\frac{\partial L}{\partial W2} = \frac{\partial L}{\partial scores} \cdot \frac{\partial scores}{\partial W2}$
    \]
    - \( $\frac{\partial L}{\partial scores} = probs - \text{one-hot label} $\)，维度为 \( N $\times$ C \)。
    - \($ \frac{\partial scores}{\partial W2} = h1^T$ \)，维度为 \( H $\times$ N \)。
    - 因此：
      \[
      $\frac{\partial L}{\partial W2} = h1^T \cdot (probs - \text{one-hot label})$
      \]
  - 对于 \( b2 \)：
    \[
    $\frac{\partial L}{\partial b2} = \sum_{i=1}^N (probs_i - \text{one-hot label}_i)$
    \]

  #### **(2) 梯度 $( \frac{\partial L}{\partial W1} ) 和 ( \frac{\partial L}{\partial b1} )$**

  - 根据链式法则：
    \[
    $\frac{\partial L}{\partial W1} = \frac{\partial L}{\partial h1} \cdot \frac{\partial h1}{\partial (X \cdot W1 + b1)} \cdot \frac{\partial (X \cdot W1 + b1)}{\partial W1}$
    \]
    - \( $\frac{\partial L}{\partial h1} = (probs - \text{one-hot label}) \cdot W2^T $\)，维度为 \( N $\times$ H \)。
    - \( $\frac{\partial h1}{\partial (X \cdot W1 + b1)} = \text{ReLU 的导数}$ \)，即 \( 1 \) 如果 \( h1 > 0 \)，否则 \( 0 \)。
    - \( $\frac{\partial (X \cdot W1 + b1)}{\partial W1} = X^T$ )，维度为 \( D $\times$ N \)。
    - 因此：
      \[
      $\frac{\partial L}{\partial W1} = X^T \cdot \left( (probs - \text{one-hot label}) \cdot W2^T \odot \text{ReLU 的导数} \right)$
      \]
  - 对于 \( b1 \)：
    \[
    $\frac{\partial L}{\partial b1} = \sum_{i=1}^N \left( (probs_i - \text{one-hot label}_i) \cdot W2^T \odot \text{ReLU 的导数} \right)$
    \]

  ---

  ### **3. 正则化项的梯度**

  如果损失函数包含 L2 正则化项（如 \( \text{reg} $\cdot$ (\|W1\|^2 + \|W2\|^2) \)），则梯度需要加上正则化项的导数：

  - 对于 \( W1 \)：
    \[
    $\frac{\partial L}{\partial W1} += 2 \cdot \text{reg} \cdot W1$
    \]
  - 对于 \( W2 \)：
    \[
    $\frac{\partial L}{\partial W2} += 2 \cdot \text{reg} \cdot W2$
    ]

  ---

  ### **4. 总结**

  通过链式法则和线性代数的知识，我们推导出了损失函数 \( L \) 对 \( W1 \)、\( W2 \)、\( b1 \)、\( b2 \) 的梯度公式。这些公式可以直接用于实现反向传播。