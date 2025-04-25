# One time setup

## Activation functions

Don’t use sigmoid or tanh because it will make gradients vanish.(they have flat region)

Sigmoid: non-zero-symmetric, it will be all positive/negative for gradients on W(each input is the activation output from the previous layer) So it will have zigzag shape in order to achieve the final optimization goal. Exponential is expensive for CPUs.

ReLU: converge very fast ; not zero-centered output; gradients are 0 if x is negative(dead ReLU: use a slightly positive biases ,say 0.01)

Leaky ReLU: will not die $f(x)=\max(0.01x,x)$

## Data Preprocessing

we want zero-centered data.

Common for images : - mean(per channel) / standard variation (per channel )

for non-images : whitening  covariance matrix is identity matrix

If we don’t normalize the data, classification loss vert sensitive to changes in weight matrix.

## Weight Initialization

Initializing matrix to all 0 is very bad! The network will not learn anything.

small random numbers(Gaussian with zero mean ) what about std?

If activation collapses to 0, it will make the local gradient to 0 because local gradient equals to the output activation from the previous layer. 

If activation all center around 1 and we use tanh as activation function, it will make the upstream gradient to be 0 because 1 is in the flat region of tanh gradient.

For tanh : Xavier initialization:$std = \sqrt{\frac{1}{Din}}$

We want the variance of output = variance of input

$y=Wx  \\ y_i = \sum_{j=1}^{Din}x_jw_j$

$Var(y_i)=Din\times Var(x_iw_i) \hfill \ [Assume\  x,w \ are \ iid] \\=Din\times (E[x_i^2]E[w_i^2]-E[x_i]^2E[w_i]^2) \hfill \ [Assume\ x,w\ independent]\\=Din\times Var(x_i)\times Var(w_i) \hfill  [Assume\ x_i,w_i \ are\  zero-mean]$ 

Note that $x_i,w_i$ are zero-mean,so $Var(x_i)=E[x_i^2]-E[x_i]^2=E[x_i^2]$

For ReLU: Kaiming initialization 

gain = $\sqrt2$  that is $std = \sqrt{\frac{2}{Din}}$

For residual networks, if we still use Kaiming initialization , we will double the variance every layer, because remember,

$Var(F(x)+x)=Var(F(x))+Var(x)=Var(x)+Var(x)=2Var(x)$

Solution: Initialize first conv with Kaiming, initialize second conv to zero.

![image-20250404181632626](./TrainingNN.assets/image-20250404181632626.png)

## Regularization

L2 regularization(Weight decay);

Dropout:

Why? Forces the network to have a redundant representation ; Prevents co-adaptation of features.

Dropout is training a large ensemble of models that share parameters. Each binary mask is one model.

At test time we need accurate(fixed) dropout, so we need to multiply a dropout probability, which is expected output in training = output in testing

Dropout only occurs in the fully-connected layers.

Batch normalization : Training: Normalize using stats from random minibatches

​				Testing: Use fixed stats to normalize

## Data Augmentation

Transform image: Horizontal Flips, random crops and scales

Cutout and mixup for small classification datasets.



## Learning Rate

 Set a high learning rate at the beginning end with a low lr.

Step Decay: 0.10 --> 0.01 --> 0.001 ...

Cosine Decay: $\alpha_{t} = \frac{1}{2}\alpha_0(1+\cos(t\pi/T))$    No hyperparameters!   Usually for CV fields

Linear Decay : $\alpha_t=\alpha_0(1-t/T)$	Usu. for NLP

Inverse Sqrt Decay : $\alpha_t=\alpha_0/\sqrt{t}$

Constant: $\alpha_t=\alpha_0$

Grid Search: [, , ,] * [, , ,]

What DeepSeek does: AdamW optimizer $\beta_1=0.9,\beta_2=0.95,Weight Decay = 0.1$

​				Warm-up & step decrease : At the beginning 2000 steps, learning rate increases to the maximum linearly. Then,after training about 80% tokens,learning rate times 0.316,90% tokens times 0.316,maximum is set to $4.2\times 10^{-4}$,gradient cut norm is set to 1.0

## Choosing Hyperparameters

Step 1 : Check initial loss

Turn off weight decay , we have some expectation on our loss

  For example : lnC for softmax with C classes

Step 2 : Overfit a small sample

Try to train to 100% training accuracy on a small sample of training data（~5-10 minibatches）；fiddle with architecture,learning rate, weight initialization .Turn off regularization.
Loss not going down？LR too low,bad initialization 

Loss explodes to Inf or NaN？LR too high,bad initialization

Step 3 : Find LR that makes loss go down.

Use the architecture from the previous step,use all training data, turn on small weight decay,find a learning rate that makes the loss drop significantly within ~100 iterations Good learning rates to try：1e-1,1e-2,1e-3,1e-4

Step 4：Coarse grid,train for ~1-5 epochs 

Choose a few values of learning rate and weight decay around wha worked from Step 3，train a few models for ~1-5 epochs.
Good weight decay to try：1e-4,1e-5,0

Observe the graph:

If train acc and val acc are nearly the same : underfitting train longer, use a bigger model

If train acc goes up and val acc goes down : overfitting

If there is smoe gap but they are all going up : good network!

## Model ensembles

Train multiple independent models --> at the test time average their results.

## Transfer Learning

pretrained models

1. Train on ImageNet
2. Use CNN as a feature extractor
3. Fine-tuning 

Some tricks：

- Train with feature extraction first before fine-tuning

- Lower the learning rate：
   use ~1/10 of LR used in original training

- Sometimes freeze lower layers to save computation

Paralleled: LR Warm-up

