# Overall Map

## Chapter 1: Basic Concepts

Concepts: state,action,reward,return,episode,policy

Grid-world example

Markov decision process(MDP)

## Chapter 2: Bellman Equation

One concept: state value 

$v_\pi(s)=\mathbb{E}[G_t|S_t=s]$

One tool: Bellman equation

$v_\pi = r_\pi + \gamma P_\pi v_\pi$

## Chapter 3: Bellman Optimality Equation

A special Bellman equation

Two concepts: optimal policy $\pi^*$ & optimal state value

One tool: Bellman optimality equation

$v = \max_\pi{r_\pi+\gamma P_\pi v} = f(v)$

1) Fixed-Point theorem
2) Fundamental problems
3) An algorithm solving the equation

## Chapter 4: Value Iteration & Policy Iteration

First algorithms for optimal policies

Three algorithms:

1) Value iteration(VI)
2) Policy iteration(PI)
3) Truncated policy iteration

Need the environment model

## Chapter 5: Monte Carlo Learning

Mean estimation with sampling data

$\mathbb{E}[X] \approx \bar{x} =\frac{1}{n}\sum_{i=1}^n x_i$

First model-free RL algorithms

1) MC Basic
2) MC Exploring Starts
3) MC $\epsilon$-greedy

## Chapter 6: Stochastic Approximation

Gap: from non-incremental to incremental

Mean estimation

Algorithm:

1) Robbins-Monro (RM) algorithm
2) Stochastic gradient descent(SGD)
3) SGD,BGD,MBGD

## Chapter 7: Temporal-Difference Learning

1) TD learning of state values

2) Sarsa: TD learning of action values

3) Q-learning : TD learning of optimal action values

   on-policy & off-policy

4) Unified point of view

## Chapter 8: Value Function Approximation

Gap: tabular representation to function representation

Algorithms:

1) State value estimation with value function approxmation(VFA):

2) $\min_w{J(w)} = \mathbb{E}[v_\pi(S)-\hat{v}(S,w)]$

   Sarsa/W-learning with VFA

   Deep W-learning

## Chapter 9: Policy Gradient Methods

Gap: From value-based to policy-based

1) Metrics to define optimal policies

   $J(\theta)=\bar{v_\pi},\bar{r_\pi}$

2) Policy gradient:

   $\nabla J(\theta)=\mathbb{E}[\nabla_\theta \ln{\pi(A|S,\theta)q_\pi(S,A)}]$

3) Gradient-ascent algorithm(REINFORCE)

   $\theta_{t+1}=\theta_t + \alpha \nabla_\theta \ln{\pi(a_t|s_t,\theta_t)q_t(s_t,a_t)}$

## Chapter 10: Actor-Critic Methods

Gap: policy-based + value-based

Algorithms:

1) The simplest actor-critic(QAC)

2) Advantage actor-critic (A2C)

3) Off-policy actor-critic

   Importance sampling

4) Deterministic actor-critic(DPG)

