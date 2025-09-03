
# 数学公式测试页面

## 行内公式测试

这是一个行内公式：$E = mc^2$

另一个行内公式：$\alpha + \beta = \gamma$

## 块级公式测试

### 使用 $$...$$ 格式

$$
f(x) = \int_{-\infty}^{\infty} \hat{f}(\xi) e^{2\pi i \xi x} d\xi
$$

### 使用 \[...\] 格式

\[
\frac{\partial f}{\partial t} + \frac{1}{2}\sigma^2\frac{\partial^2 f}{\partial x^2} = 0
\]

## 复杂公式测试

### 带有对齐环境的公式

#### 使用 \begin{aligned} 环境

$$
\begin{aligned}
\nabla \times \vec{\mathbf{B}} -\, \frac1c\, \frac{\partial\vec{\mathbf{E}}}{\partial t} &= \frac{4\pi}{c}\vec{\mathbf{j}} \\
\nabla \cdot \vec{\mathbf{E}} &= 4 \pi \rho \\
\nabla \times \vec{\mathbf{E}}\, +\, \frac1c\, \frac{\partial\vec{\mathbf{B}}}{\partial t} &= \vec{\mathbf{0}} \\
\nabla \cdot \vec{\mathbf{B}} &= 0
\end{aligned}
$$

#### 使用 \begin{align} 环境

$$
\begin{align}
\mathbb{E}[G_{t+1}|S_t = s] &= \sum_{s'} \mathbb{E}[G_{t+1}|S_t = s, S_{t+1} = s']p(s'|s) \\
&= \sum_{s'} \mathbb{E}[G_{t+1}|S_{t+1} = s']p(s'|s) \\
&= \sum_{s'} v_\pi(s')p(s'|s) \\
&= \sum_{s'} v_\pi(s') \sum_{a} p(s'|s,a)\pi(a|s) \\
&=\sum_a\pi(a|s)\sum_{s'}p(s'|s,a)v_\pi(s')
\end{align}
$$

### 带有矩阵的公式

$$
\mathbf{X} = \begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1n} \\
x_{21} & x_{22} & \cdots & x_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
x_{m1} & x_{m2} & \cdots & x_{mn}
\end{bmatrix}
$$

### 带有求和符号的公式

$$
\sum_{i=1}^{n} i = \frac{n(n+1)}{2}
$$

### 带有极限的公式

$$
\lim_{x \to \infty} \left(1 + \frac{1}{x}\right)^x = e
$$

### 带有积分的公式

$$
\int_{a}^{b} f(x) dx = F(b) - F(a)
$$

### 带有希腊字母的公式

$$
\Delta = b^2 - 4ac
$$

### 带有上下标的公式

$$
x_1^2 + x_2^2 + \cdots + x_n^2 = r^2
$$

### 带有分数的公式

$$
\frac{a}{b} = \frac{c}{d} \implies ad = bc
$$

### 带有根号的公式

$$
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$

### 带有集合符号的公式

$$
A \cup B = \{x \mid x \in A \text{ or } x \in B\}
$$

### 带有逻辑符号的公式

$$
P \land Q \implies R
$$

### 带有概率符号的公式

$$
\mathbb{P}(X = x) = \frac{e^{-\lambda}\lambda^x}{x!}
$$

### 带有期望符号的公式

$$
\mathbb{E}[X] = \sum_{x \in \mathcal{X}} x \cdot \mathbb{P}(X = x)
$$

### 带有方差符号的公式

$$
\mathrm{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
$$

### 带有协方差符号的公式

$$
\mathrm{Cov}(X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]
$$

### 带有 argmax 和 argmin 的公式

$$
\hat{\theta} = \arg\max_{\theta} \mathcal{L}(\theta)
$$

### 带有条件概率的公式

$$
\mathbb{P}(A \mid B) = \frac{\mathbb{P}(A \cap B)}{\mathbb{P}(B)}
$$

### 带有贝叶斯公式的公式

$$
\mathbb{P}(A \mid B) = \frac{\mathbb{P}(B \mid A) \cdot \mathbb{P}(A)}{\mathbb{P}(B)}
$$

### 带有向量符号的公式

$$
\vec{v} = \begin{pmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{pmatrix}
$$

### 带有梯度符号的公式

$$
\nabla f = \left( \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n} \right)
$$

### 带有散度符号的公式

$$
\nabla \cdot \vec{F} = \frac{\partial F_x}{\partial x} + \frac{\partial F_y}{\partial y} + \frac{\partial F_z}{\partial z}
$$

### 带有旋度符号的公式

$$
\nabla \times \vec{F} = \begin{vmatrix}
\hat{i} & \hat{j} & \hat{k} \\
\frac{\partial}{\partial x} & \frac{\partial}{\partial y} & \frac{\partial}{\partial z} \\
F_x & F_y & F_z
\end{vmatrix}
$$

### 带有拉普拉斯算子的公式

$$
\Delta f = \nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} + \frac{\partial^2 f}{\partial z^2}
$$

### 带有傅里叶变换的公式

$$
\hat{f}(\xi) = \int_{-\infty}^{\infty} f(x) e^{-2\pi i x \xi} dx
$$

### 带有拉普拉斯变换的公式

$$
F(s) = \mathcal{L}\{f(t)\} = \int_0^{\infty} f(t) e^{-st} dt
$$

### 带有泰勒级数的公式

$$
f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n
$$

### 带有麦克劳林级数的公式

$$
f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(0)}{n!}x^n
$$

### 带有欧拉公式的公式

$$
e^{i\theta} = \cos \theta + i \sin \theta
$$

### 带有欧拉恒等式的公式

$$
e^{i\pi} + 1 = 0
$$

### 带有二项式定理的公式

$$
(a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k} b^k
$$

### 带有组合数的公式

$$
\binom{n}{k} = \frac{n!}{k!(n-k)!}
$$

### 带有排列数的公式

$$
P(n, k) = \frac{n!}{(n-k)!}
$$

### 带有阶乘的公式

$$
n! = n \times (n-1) \times (n-2) \times \cdots \times 2 \times 1
$$

### 带有双阶乘的公式

$$
n!! = n \times (n-2) \times (n-4) \times \cdots \times 2 \text{ 或 } 1
$$

### 带有伽马函数的公式

$$
\Gamma(z) = \int_0^{\infty} t^{z-1} e^{-t} dt
$$

### 带有贝塔函数的公式

$$
B(x, y) = \int_0^1 t^{x-1} (1-t)^{y-1} dt = \frac{\Gamma(x)\Gamma(y)}{\Gamma(x+y)}
$$

### 带有黎曼 zeta 函数的公式

$$
\zeta(s) = \sum_{n=1}^{\infty} \frac{1}{n^s}
$$

### 带有误差函数的公式

$$
\operatorname{erf}(x) = \frac{2}{\sqrt{\pi}} \int_0^x e^{-t^2} dt
$$

### 带有互补误差函数的公式

$$
\operatorname{erfc}(x) = 1 - \operatorname{erf}(x) = \frac{2}{\sqrt{\pi}} \int_x^{\infty} e^{-t^2} dt
$$

### 带有正弦积分的公式

$$
\operatorname{Si}(x) = \int_0^x \frac{\sin t}{t} dt
$$

### 带有余弦积分的公式

$$
\operatorname{Ci}(x) = -\int_x^{\infty} \frac{\cos t}{t} dt
$$

### 带有指数积分的公式

$$
E_1(x) = \int_x^{\infty} \frac{e^{-t}}{t} dt
$$

### 带有对数积分的公式

$$
\operatorname{li}(x) = \int_0^x \frac{dt}{\ln t}
$$

### 带有菲涅尔积分的公式

$$
S(x) = \int_0^x \sin\left(\frac{\pi t^2}{2}\right) dt
$$

### 带有椭圆积分的公式

$$
F(\phi, k) = \int_0^{\phi} \frac{d\theta}{\sqrt{1 - k^2 \sin^2 \theta}}
$$

### 带有贝塞尔函数的公式

$$
J_n(x) = \frac{1}{\pi} \int_0^{\pi} \cos(n\theta - x \sin \theta) d\theta
$$

### 带有勒让德多项式的公式

$$
P_n(x) = \frac{1}{2^n n!} \frac{d^n}{dx^n} \left[(x^2 - 1)^n\right]
$$

### 带有埃尔米特多项式的公式

$$
H_n(x) = (-1)^n e^{x^2} \frac{d^n}{dx^n} e^{-x^2}
$$

### 带有拉盖尔多项式的公式

$$
L_n(x) = \frac{e^x}{n!} \frac{d^n}{dx^n} (x^n e^{-x})
$$

### 带有切比雪夫多项式的公式

$$
T_n(x) = \cos(n \arccos x)
$$

### 带有拉盖尔伴随多项式的公式

$$
L_n^{(k)}(x) = \frac{d^k}{dx^k} L_{n+k}(x)
$$

### 带有雅可比多项式的公式

$$
P_n^{(\alpha, \beta)}(x) = \frac{(\alpha + 1)_n}{n!} \sum_{k=0}^n \binom{n}{k} \frac{(\alpha + \beta + n + 1)_k}{(\alpha + 1)_k} \left(\frac{x-1}{2}\right)^k
$$

### 带有超几何函数的公式

$$
{}_2F_1(a, b; c; z) = \sum_{n=0}^{\infty} \frac{(a)_n (b)_n}{(c)_n} \frac{z^n}{n!}
$$

### 带有合流超几何函数的公式

$$
{}_1F_1(a; b; z) = \sum_{n=0}^{\infty} \frac{(a)_n}{(b)_n} \frac{z^n}{n!}
$$

### 带有广义超几何函数的公式

$$
{}_pF_q(a_1, \ldots, a_p; b_1, \ldots, b_q; z) = \sum_{n=0}^{\infty} \frac{(a_1)_n \cdots (a_p)_n}{(b_1)_n \cdots (b_q)_n} \frac{z^n}{n!}
$$

### 带有梅林变换的公式

$$
\mathcal{M}\{f(t)\}(s) = \int_0^{\infty} t^{s-1} f(t) dt
$$

### 带有汉克尔变换的公式

$$
F_\nu(k) = \int_0^{\infty} f(r) J_\nu(kr) r dr
$$

### 带有希尔伯特变换的公式

$$
\mathcal{H}\{f(t)\}(u) = \frac{1}{\pi} \text{p.v.} \int_{-\infty}^{\infty} \frac{f(t)}{t-u} dt
$$

### 带有哈特利变换的公式

$$
H(\omega) = \frac{1}{\sqrt{2\pi}} \int_{-\infty}^{\infty} f(t) (\cos(\omega t) + \sin(\omega t)) dt
$$

### 带有小波变换的公式

$$
W(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} f(t) \psi^*\left(\frac{t-b}{a}\right) dt
$$

### 带有Z变换的公式

$$
X(z) = \sum_{n=-\infty}^{\infty} x[n] z^{-n}
$$

### 带有离散时间傅里叶变换的公式

$$
X(e^{j\omega}) = \sum_{n=-\infty}^{\infty} x[n] e^{-j\omega n}
$$

### 带有离散傅里叶变换的公式

$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-j 2\pi kn/N}
$$

### 带有快速傅里叶变换的公式

$$
X[k] = \sum_{n=0}^{N-1} x[n] W_N^{kn}, \quad W_N = e^{-j 2\pi / N}
$$

### 带有余弦变换的公式

$$
X[k] = \sum_{n=0}^{N-1} x[n] \cos\left[\frac{\pi}{N}\left(n + \frac{1}{2}\right)k\right]
$$

### 带有正弦变换的公式

$$
X[k] = \sum_{n=0}^{N-1} x[n] \sin\left[\frac{\pi}{N}\left(n + \frac{1}{2}\right)k\right]
$$

### 带有沃尔什变换的公式

$$
X[k] = \sum_{n=0}^{N-1} x[n] \text{wal}(k, n)
$$

### 带有哈达玛变换的公式

$$
X[k] = \sum_{n=0}^{N-1} x[n] (-1)^{\sum_{i=0}^{m-1} k_i n_i}
$$

### 带有哈尔变换的公式

$$
X[k] = \sum_{n=0}^{N-1} x[n] h_k(n)
$$

### 带有斜变换的公式

$$
X[k] = \sum_{n=0}^{N-1} x[n] s(k, n)
$$

### 带有离散余弦变换的公式

$$
X[k] = c_k \sum_{n=0}^{N-1} x[n] \cos\left[\frac{\pi}{N}\left(n + \frac{1}{2}\right)k\right]
$$

### 带有离散正弦变换的公式

$$
X[k] = \sum_{n=0}^{N-1} x[n] \sin\left[\frac{\pi}{N+1}(n+1)(k+1)\right]
$$

### 带有离散小波变换的公式

$$
W_{j,k} = \sum_{n} x[n] \psi_{j,k}[n]
$$

### 带有连续小波变换的公式

$$
W(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \psi^*\left(\frac{t-b}{a}\right) dt
$$

### 带有短时傅里叶变换的公式

$$
X(\tau, \omega) = \int_{-\infty}^{\infty} x(t) w(t-\tau) e^{-j\omega t} dt
$$

### 带有维格纳分布的公式

$$
W_x(t, \omega) = \int_{-\infty}^{\infty} x\left(t + \frac{\tau}{2}\right) x^*\left(t - \frac{\tau}{2}\right) e^{-j\omega \tau} d\tau
$$

### 带有科恩类分布的公式

$$
C_x(t, \omega) = \frac{1}{2\pi} \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} A_x(\theta, \tau) \phi(\theta, \tau) e^{j(\theta t + \tau \omega - \theta \tau)} d\theta d\tau
$$

### 带有模糊函数的公式

$$
A_x(\theta, \tau) = \int_{-\infty}^{\infty} x\left(t + \frac{\tau}{2}\right) x^*\left(t - \frac{\tau}{2}\right) e^{j\theta t} dt
$$

### 带有谱图的公式

$$
S_x(t, \omega) = \left| \int_{-\infty}^{\infty} x(\tau) w(\tau - t) e^{-j\omega \tau} d\tau \right|^2
$$

### 带有标量图的公式

$$
S_x(t, \omega) = \left| STFT_x(t, \omega) \right|^2
$$

### 带有重排的公式

$$
R_x(t, \omega) = \iint S_x(t', \omega') \delta\left(t - \hat{t}(t', \omega')\right) \delta\left(\omega - \hat{\omega}(t', \omega')\right) dt' d\omega'
$$

### 带有同步压缩的公式

$$
T_x(t, \omega) = \frac{1}{g(\omega)} \iint S_x(t', \omega') \delta\left(\omega - \hat{\omega}(t', \omega')\right) dt' d\omega'
$$

### 带有经验模态分解的公式

$$
x(t) = \sum_{i=1}^{n} c_i(t) + r_n(t)
$$

### 带有希尔伯特-黄变换的公式

$$
H(t, \omega) = \operatorname{Re} \sum_{i=1}^{n} a_i(t) e^{j \int \omega_i(t) dt}
$$

### 带有变分模态分解的公式

$$
\min_{\{u_k\}, \{\omega_k\}} \left\{ \sum_{k=1}^{K} \left\| \partial_t \left[ \left( \delta(t) + \frac{j}{\pi t} \right) * u_k(t) \right) e^{-j\omega_k t} \right\|_2^2 \right\}
$$

### 带有经验小波变换的公式

$$
W(a, b) = \frac{1}{\sqrt{a}} \int_{-\infty}^{\infty} x(t) \psi^*\left(\frac{t-b}{a}\right) dt
$$

### 带有多重分形去趋势波动分析的公式

$$
F_q(s) = \left\{ \frac{1}{N_s} \sum_{v=1}^{N_s} \left[ F^2(v, s) \right]^{q/2} \right\}^{1/q}
$$

### 带有去趋势移动平均分析的公式

$$
F_2(s) = \sqrt{\frac{1}{N-s+1} \sum_{i=1}^{N-s+1} \left[ Y(i) - \tilde{Y}_s(i) \right]^2}
$$

### 带有递归定量分析的公式

$$
RR = \frac{1}{N^2} \sum_{i,j=1}^{N} R_{i,j}
$$

### 带有递归熵的公式

$$
H_{rec} = -\sum_{l=l_{\min}}^{N} p(l) \ln p(l)
$$

### 带有递归率的公式

$$
DET = \frac{\sum_{l=l_{\min}}^{N} l P(l)}{\sum_{l=1}^{N} l P(l)}
$$

### 带有层流率的公式

$$
LAM = \frac{\sum_{l=1}^{l_{\min}-1} l P(l)}{\sum_{l=1}^{N} l P(l)}
$$

### 带有递归时间分布熵的公式

$$
RTD = -\sum_{i=1}^{N} p_i \ln p_i
$$

### 带有递归时间分布的公式

$$
p_i = \frac{\tau_i}{\sum_{j=1}^{N} \tau_j}
$$

### 带有递归时间分布的公式

$$
\tau_i = t_i - t_{i-1}
$$

### 带有递归时间分布的公式

$$
t_i = \min \{ t > t_{i-1} : R_{i,j} = 1 \}
$$

### 带有递归时间分布的公式

$$
R_{i,j} = \Theta(\varepsilon - \| \vec{x}_i - \vec{x}_j \|)
$$

### 带有递归时间分布的公式

$$
\Theta(x) = \begin{cases} 1, & x \geq 0 \\ 0, & x < 0 \end{cases}
$$

### 带有递归时间分布的公式

$$
\| \vec{x}_i - \vec{x}_j \| = \sqrt{\sum_{k=1}^{d} (x_{i,k} - x_{j,k})^2}
$$

### 带有递归时间分布的公式

$$
\vec{x}_i = (x_{i,1}, x_{i,2}, \ldots, x_{i,d})
$$

### 带有递归时间分布的公式

$$
x_{i,k} = x(t_i + (k-1)\tau)
$$

### 带有递归时间分布的公式

$$
t_i = t_0 + i \Delta t
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\tau = \frac{1}{f_0}
$$

### 带有递归时间分布的公式

$$
f_0 = \frac{1}{\tau}
$$

### 带有递归时间分布的公式

$$
d = m \tau
$$

### 带有递归时间分布的公式

$$
m = \frac{d}{\tau}
$$

### 带有递归时间分布的公式

$$
\varepsilon = \alpha \sigma
$$

### 带有递归时间分布的公式

$$
\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2}
$$

### 带有递归时间分布的公式

$$
\bar{x} = \frac{1}{N} \sum_{i=1}^{N} x_i
$$

### 带有递归时间分布的公式

$$
x_i = x(t_i)
$$

### 带有递归时间分布的公式

$$
t_i = t_0 + i \Delta t
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
t_0 = 0
$$

### 带有递归时间分布的公式

$$
i = 1, 2, \ldots, N
$$

### 带有递归时间分布的公式

$$
N = \frac{T}{\Delta t}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{f_0}
$$

### 带有递归时间分布的公式

$$
f_0 = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
f_k = k \Delta f
$$

### 带有递归时间分布的公式

$$
k = 0, 1, \ldots, N-1
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
f_{\max} = \frac{f_s}{2}
$$

### 带有递归时间分布的公式

$$
f_{\min} = 0
$$

### 带有递归时间分布的公式

$$
f_{\text{Nyquist}} = \frac{f_s}{2}
$$

### 带有递归时间分布的公式

$$
f_s \geq 2 f_{\max}
$$

### 带有递归时间分布的公式

$$
f_{\max} = \frac{f_s}{2}
$$

### 带有递归时间分布的公式

$$
f_s = 2 f_{\max}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{1}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{f_s}{N}
$$

### 带有递归时间分布的公式

$$
f_s = \frac{1}{\Delta t}
$$

### 带有递归时间分布的公式

$$
\Delta t = \frac{T}{N}
$$

### 带有递归时间分布的公式

$$
T = N \Delta t
$$

### 带有递归时间分布的公式

$$
f_s = \frac{N}{T}
$$

### 带有递归时间分布的公式

$$
\Delta f = \frac{1}{T}
$$

### 带有递归时间分布的公式

$$
T = \frac{1}{\Delta f}
$$

### 带有递归时间分布的公式

$$
f_s = N \Delta f
$$

### 带有递归时间分布的公式

$$
N = \frac{f_s}{\Delta f}
$$

### 带有递归时间分布的公式

$$
N = f_s T
$$

### 带有递归时间分布的公式

$$
T = \frac{N}{f_s}
$$

### 带有递归时间分布的公式

$$
\Delta t =