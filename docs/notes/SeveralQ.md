# 我们到底应该如何理解BatchNorm和LayerNorm?

Reference: 1.   [machine learning - Why do transformers use layer norm instead of batch norm? - Cross Validated (stackexchange.com)](https://stats.stackexchange.com/questions/474440/why-do-transformers-use-layer-norm-instead-of-batch-norm)

2.  [BatchNorm和LayerNorm——通俗易懂的理解_layernorm和batchnorm-CSDN博客](https://blog.csdn.net/Little_White_9/article/details/123345062)

![img](https://miro.medium.com/v2/resize:fit:1400/0*K45DoPRbhC5-dqq1)

说人话： BN 是对每一个通道计算同一批样本所有对应特征的均值和标准差

LN 是对每一个样本计算所有特征的均值和标准差

BN 常用于 CV(CNN),因为我们见识不同的样本来了解“猫”“狗”的含义，也就是说不同样本对同一特征要有可比性。

LN 常用于NLP(Transformer),因为我们希望学习一个词是通过这一句话的上下文，也就是同一个样本的所有特征。

BN示意图：

![BN1示意图](https://i-blog.csdnimg.cn/blog_migrate/d5e54b2f27d60e4cd852cc35e22f4a09.jpeg)

---



LN示意图：

![LN示意图](https://i-blog.csdnimg.cn/blog_migrate/166896d1305f8859e931206ff127a383.jpeg)

# 我们为什么需要激活函数(比如ReLU)?

The purpose of the activation function is to introduce **non-linearity into the network**

in turn, this allows you to model a response variable (aka target variable, class label, or score) that varies non-linearly with its explanatory variables

*non-linear* means that the output cannot be reproduced from a linear combination of the inputs (which is not the same as output that renders to a straight line--the word for this is *affine*).

another way to think of it: without a *non-linear* activation function in the network, a NN, no matter how many layers it had, would behave just like a single-layer perceptron, because summing these layers would give you just another linear function (see definition just above).

---

# 我们为什么需要归一化？

Normalized data enhances model performance and improves the accuracy of a model. It aids algorithms that rely on distance metrics, such as [k-nearest neighbors](https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn) or [support vector machines](https://www.datacamp.com/tutorial/svm-classification-scikit-learn-python), by preventing features with larger scales from dominating the learning process.

Normalization fosters stability in the optimization process, promoting faster convergence during gradient-based training. It mitigates issues related to vanishing or exploding gradients, allowing models to reach optimal solutions more efficiently.

Normalized data is also easy to interpret and thus, easier to understand. When all the features of a dataset are on the same scale, it also becomes easier to identify and visualize the relationships between different features and make meaningful comparisons.

**1. With Unnormaized data:**

Since your network is tasked with learning how to combine inputs through a series of linear combinations and nonlinear activations, the parameters associated with each input will exist on different scales.

Unfortunately, this can lead toward an awkward loss function topology which places more emphasis on certain parameter gradients.

If the images are not normalized, the input pixels will range from [ 0 , 255 ]. These will produce huge activation values ( if you're using ReLU ). After the forward pass, you'll end up with a huge loss value and gradients.

**2. With Normalized data:**

By normalizing our inputs to a standard scale, we're allowing the network to more quickly learn the optimal parameters for each input node.

Additionally, it's useful to ensure that our inputs are roughly in the range of -1 to 1 to avoid weird mathematical artifacts associated with floating-point number precision. In short, computers lose accuracy when performing math operations on really large or really small numbers. Moreover, if your inputs and target outputs are on a completely different scale than the typical -1 to 1 range, the default parameters for your neural network (ie. learning rates) will likely be ill-suited for your data. In the case of image the pixel intensity range is bound by 0 and 1(mean =0 and variance =1).