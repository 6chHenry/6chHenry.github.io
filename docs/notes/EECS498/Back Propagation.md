Back Propagation:

Simple Example:$f(x,y,z)=(x+y)z$

1.Forward Pass: Compute outputs from left to right(from inputs to outputs)

$q=x+y$  $f=qz$

2.Backward Pass: Compute derivatives 

Want: $\frac{\part f}{\part x},\frac{\part f}{\part y},\frac{\part f}{\part z}$

Order: $\frac{\part f}{\part f}\rightarrow \frac{\part f}{\part z}\rightarrow \frac{\part f}{\part q}\rightarrow \frac{\part f}{\part y}=\frac{\part f}{\part q}\frac{\part q}{\part y}\rightarrow \frac{\part f}{\part x}=\frac{\part f}{\part q}\frac{\part q}{\part x}$

[Downstream] = [Local]*[Upstream]

sigmoid function:$\sigma(x)=\frac{1}{1+e^{-x}}$

$\frac{d\sigma(x)}{dx}=(1-\sigma(x))\sigma(x)$

add: don’t change derivatives so the elements linked by the “+” share the same gradient.

copy: one side’s derivatives add to the other side.

multiply: swap multiplier

max: gradient router,that reduces one element to $0$ and another,full gradient the same with other side’s element’s.

Flat: “Reverse Thinking Method”:

```python
#Forward
def f(w0,x0,w1,x1,w2):
    s0 = w0 * x0
    s1 = w1 * x1
    s2 = s0 + s1
    s3 = s2+ w2
    L = sigmoid(s3)
#Backward
	grad_L = 1.0
    grad_s3 = grad_L * (1-L) * L    #sigmoid function has a special form of derivatives
    grad_w2 = grad_s3	#gradient copier
    grad_s2 = grad_s3	#gradient copier
    grad_s0 = grad_s2	#gradient copier
    grad_s1 = grad_s2	#gradient copier
    grad_w1 = grad_s1 * x1	#gradient multiplier
    grad_x1 = grad_s1 * w1	#gradient multiplier
    grad_w0 = grad_s0 * x0	#gradient multiplier
    grad_x0 = grad_w0 * w0	#gradient multiplier
```

$y \in \mathbb{R},x \in \mathbb{R}^M,\frac{dy}{dx}\in \mathbb{R}^M$

$y \in \mathbb{R}^N,x \in \mathbb{R}^M,\frac{dy}{dx}\in \mathbb{R}^{M\times N}$ :Jacobian Matrix

4D INPUT $x$:[1,-2,3,-3]  $\rightarrow f(x)=\max(0,x)(elementwise)\rightarrow $  4D OUTPUT $y$ = [1,0,3,0]

4D $\frac{dL}{dy}$:[4,-1,5,9] $\rightarrow $ $\frac{dy}{dx}\frac{dL}{dy}$

$\frac{dy}{dx}$=$$\begin{bmatrix}1&0&0&0 \\0&0&0&0 \\0&0&1&0\\0&0&0&0  \end{bmatrix}$$positive: 1 negative: 0

Jacobian is sparse!: off-diagonal entries all zero! When doing a big Jacobian Matrix Multiply,it’ll cause large resources waste because almost every element is zero!Never explicitly form Jacobian,instead use implicit multiplication.

$y=xw$($x:[N\times D]$ $w:[D\times M]$  $y:[N \times M]$)

$\frac{dL}{dx_{i,j}}=\frac{dy}{dx_{i,j}}\cdot\frac{dL}{dy}=w_{j,:}\cdot\frac{dL}{dy_{i,:}}$

$\frac{dL}{dx}=\frac{dL}{dy}w^T$(How to remember? Use the shape!)

$\frac{dL}{dw}=x^T\frac{dL}{dy}$

Hint:     $\cdot$  means inner product while blank space means matrix multiply

Backward-Mode: A vector input and a scalar output

Forward-Mode: A scalar input and a vector output

Compute Higher-Order Derivatives(Cool!):

$x_0 --f1-->x1--f2-->L--f_2'-->\frac{dL}{dx_1}--f_1'-->\frac{dL}{dx_0}--\cdot v-->\frac{dL}{dx_0}\cdot v$

we want to calculate $\frac{\part^2L}{\part x_0^2}$  then we can calculate$\frac{\part^2L}{\part x_0^2}\cdot v$ ,  and surprisingly,$\frac{\part^2L}{\part x_0^2}\cdot v=\frac{\part}{\part x_0}[\frac{\part L}{\part x_0}\cdot v]$

($v$ is independent from $x_0$) 

use backprop we will get the answer(remember: backprop gets the derivatives of output with regard to the input)





