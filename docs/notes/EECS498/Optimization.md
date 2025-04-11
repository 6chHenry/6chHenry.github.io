$w^* = \arg \min_wL(w)$

Idea #1 :Random Search(Bad Idea!)

```python
bestloss = float('inf')
for num in xrange(1000):
    W = np.random.randn(10,3073) * 0.001
    loss = L(X_train,Y_train,W) #L is the loss function
    if loss < bestloss:
        bestloss = loss
        bestW = W
    print(f'in attempt {num} the loss was {loss},best {bestloss}')
```



Batch Gradient Descent

$L(W) = \frac{1}{N}\sum_{i=1}^NL_i(x_i,y_i,W)+\lambda R(W)$

$\nabla_WL(W)=\frac{1}{N}\sum_{i=1}^N\nabla_WL_i(x_i,y_i,W)+\lambda\nabla_WR(W)$

Idea #2 : Stochastic Gradient Descent

```python
w = initialize_weights()
for t in range(num_steps):
	minibatch = sample_data(data,batch_size)
    dw = compute_gradient(loss_fn,minibatch,w)
    w- = learning_rate * dw
```

SGD: $x_{t+1}=x_t - \alpha \nabla f(x_t)$

Problems:1.overshoot and never get back

​		2.settle in local minimum and saddle point

SGD+Momentum: $v_{t+1}=\rho v_t -\alpha \nabla f (x_t)$

​				$x_{t+1}=x_t+v_{t+1}$ 

Nesterov Momentum:$v_{t+1}=\rho v_t-\alpha \nabla f(x_t+\rho v_t)$

​				$x_{t+1}=x_t+v_{t+1}$   Not that good :Not intuitively clear,because it uses the data of future status

​		or            $\tilde{x_t} =x_t + \rho v_t $

​				$v_{t+1}=\rho v_t -\alpha \nabla f (\tilde{x_t})$

​				$\widetilde{x_{t+1}}=\tilde{x_t}-\rho v_t + (1 + \rho) v_{t+1}$

​					$=\tilde{x_t}+v_{t+1}+\rho (v_{t+1}-v_t)$

AdaGrad: Progress along “steep” directions is damped;

​		progress along “flat” directions is accelerated.

```python
grad_squared = 0
for t in range(num_steps):
    dw = compute_gradient(w)
    grad_squared += dw*dw
    w -= learning_rate * dw / (grad_squared.sqrt() + 1e-7)
```

Problem: grad_squared is accumulative so that it will stop before getting to the bottom.(it can get very big)

RMSProp: a leaky version of Adaguard

```python
grad_square = 0
for t in range(num_steps):
    dw = compute_gradient(w)
    grad_squared = decay_rate * grad_squared + (1 - decay_rate) * dw * dw
    w -= learning_rate * dw / (grad_squared.sqrt() + 1e-7)
```



Adam: RMSProp + Momentum

```python
moment1 = 0
moment2 = 0
for t in range(num_steps):
    dw = compute_gradient(w)
    moment1 = beta1 * moment1 + (1-beta1) * dw #Momentum
    moment2 = beta2 * moment2 + (1-beta2) * dw * dw #RMSProp
    moment1_unbias = moment1 / (1 - beta1 ** t)
    moment2_unbias = moment2 / (1 - beta2 ** t)
    w -= learning_rate * moment1_unbias / (moment2_unbias.sqrt() + 1e-7)
    # Problem: when beta2 is approximately 1,momenent.sqrt() in the first several steps can be very small,thus leading to the */moment2.sqrt() very big.
    #We need to correct the bias.  
```

Adam with beta1 = 0.9,beta2 = 0.999,and learning_rate = 1e-3,5e-4,1e-4 is a great starting point for many models.

AdamW:(W stands for weight decay)
Only differs from Adam in the last step,
$w_{t+1}=w_t-r\frac{V_w^{correct}}{\sqrt{S_w^{correct}}+\epsilon}-r\lambda w_t$

