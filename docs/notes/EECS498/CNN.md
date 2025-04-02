Convolution Layer

Without padding: Feature maps shrink with each layer.

Input: W  Filter: K Output : W-K+1

padding = 1,usually zero-padding.

Why zero padding? it breaks the  translational equivalent(平移不变性)

Output: W-K+1+2P

Same padding: Set P = $\frac{K-1}{2}$ to make the out put same as input. 

Receptive Fields :

Each successive convolution adds K-1 to the receptive field size 

With L layers the receptive field size is 1+L*（K-1）

With Stride:(W-K+2P)/S + 1