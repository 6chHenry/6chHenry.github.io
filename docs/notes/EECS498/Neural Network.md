Linear Classifiers cannot deal with non-linear boundaries.

such as two circles with different radius but the same center point.

--> use polar coordinate to create a feature space. $r$ is the x axis and the $\theta$ is the y axis.

all points on the same circle is seated on the same vertical line that is parallel to the y axis.

(Before) Linear score function: only a small part of Feature Extraction can adjust itself to better maximizing its ability.

Learn only one template of one category.

(After) Neural Network: raw picture pixel --> classification scores

Learn several templates of one category.

Linear score function: $f = Wx$

2-layer Neural Network:$f=W_2\max(0,W_1x)$

​				$W_2 \in \mathbb{R}^{C\times H} \:  W_1 \in \mathbb{R}^{H\times D}\:  x \in \mathbb{R}^D$

$h = W_1x = (\alpha_1 ,\alpha_2,\cdots,\alpha_H)^Tx$

Element $(i,j)$ of $W_1$ gives the effect on $h_i$ from $x_j$

Deep Neural Networks: Depth = number of layers = number of Matrix

​	Width = Size of each layer

Activation Functions:

Without the activation function,we will go back to $f=W_2W_1x=Wx$ which is linear classifiers.

| Activation Functions                          | Expression                     | Graph                                                  |
| --------------------------------------------- | ------------------------------ | ------------------------------------------------------ |
| Sigmoid                                       | $\sigma(x)=\frac{1}{1+e^{-x}}$ | ![Sigmoid Function](./Neural%20Network.assets/sigma.png) |
| tanh                                          | tanh(x)                        | ![tanhx](./Neural%20Network.assets/tanh.png)             |
| ReLU(A good default choice for most problems) | max(0,x)                       | ![ReLU](./Neural%20Network.assets/relu.png)           |

A simple achievement:

```python
import numpy as np
from numpy.random import randn

N,Din,H,Dout = 64,1000,100,10
x,y = randn(N,Din),randn(N,Dout)
w1,w2 = randn(Din,H),randn(H,Dout)
for t in range(10000):
    h = 1.0 / (1.0 + np.exp(-x.dot(w1)))
    y_pred = h.dot(w2)
    loss = np.square(y_pred - y).sum()
    dy_pred = 2.0 * (y_pred - y)
    dw2 = h.T.dot(dy_pred)
    dh = dy_pred.dot(w2.T)
    dw1 = x.T.dot(dh*h*(1-h))
    w1 -= 1e-4 * dw1
    w2 -= 1e-4 * dw2
```



Space warping:

Linear transform cannot linearly separate points even in feature space.

but with ReLU function,![Space Warping](./Neural%20Network.assets/spacewarping.jpg)

Universal Approximation:

​	use layer bias to move the graph

![UA](./Neural%20Network.assets/UA.png)

use many ReLU to approach the function.

to reach 0 or unchanged: slope should be opposite

let coefficient of x be 1,only change the shaping factor of MAX.

Convex Functions:

$f:X \subset \mathbb{R}^N \rightarrow \mathbb{R}$ is *convex* if for all $x_1,x_2 \in X,t\in[0,1],f(tx_1+(1-t)x_2)\leq tf(x_1)+(1-t)f(x_2)$

convex is easy to optimize

