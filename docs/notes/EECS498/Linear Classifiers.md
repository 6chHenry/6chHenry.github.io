Linear Classification: $f(x_i,W) = W \cdot x$

Matrix multiply: stretch x to a one-dimension vector,W is a matrix.

# Multiclass SVM Loss:

Let $f(x_i,W)$ be scores,then the SVM scores has the form: $L_i = \sum_{j\neq y_i}\max(0,s_j-s_{y_i}+1)$

$s_{y_i}$ is the correct label’s score,while $s_j$ is the wrong label’s scores. When $s_j$ is larger than $s_{y_i} - 1$

,that means it contributes to the loss,so that $L_i$ is greater than $0$.

Characteristics: 1.When give the $s_{y_i}$ a little bit change,the Loss function will not change. Because after change,$s_{y_i}$ is still 1 more than the wrong label’s scores.

min possible : 0 max:$+\infty$

When all scores are small random values,loss is $C - 1$($s_j \approx s_{y_i}$) where C stands for the number of categories.

# Regularization

$L(W)=\frac{1}{N}\sum_{i=1}^NL_i(f(x_i,W),y_i)+\lambda R(W)$  

The most common regularization: L2-norm $\sum_i\sum_jW_{i,j}^2$ 

Why we need that?:

- Express preferences in among models beyond “minimize training error”,allow people to integrate their wisdom and knowledge they’ve already obtained.

- Avoid *overfitting* 

  Example: $x = [1,1,1,1] \newline w_1=[1,0,0,0] \newline w_2=[0.25,0.25,0.25,0.25]$

  It’s obvious that $w_1^\mathrm T \cdot x = w_2^\mathrm T\cdot x = 1$

  L2-norm regularization prefer more balanced matrix,which is $w_2$ in this example. This implies that use as many functions as possible in this preference.”spread out the weights”

  prefer simple models: Occam's Razor reveals the truth that simplicity is much preferred.

# Cross Entropy Loss

SoftMax function: 

| cat  | 3.2  | 24.5  | 0.13 |
| ---- | ---- | ----- | ---- |
| car  | 5.1  | 164.0 | 0.87 |
| frog | -1.7 | 0.18  | 0.00 |

​				unnormalized log-prob/logits --exp--> unnormalized prob --normalize-->probabilities

$L_i = -\ln P(Y = y_i |X = x_i)$  Maximum Likelihood Estimation

min possible loss:0 (it can only approach to 0 but never truly reach)	max:$+\infty$

When all scores are small random values,loss is $-\ln C$ where C stands for the number of categories.

