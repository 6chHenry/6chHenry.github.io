# Pseudocode

## Value Iteration 

```python

import numpy as np

# Define environment
grid_size = 4
states = [(i, j) for i in range(grid_size) for j in range(grid_size)]
actions = ['up', 'down', 'left', 'right']
terminal_state = (grid_size - 1, grid_size - 1)

# Initialize state-value function V(s)
V = np.zeros((grid_size, grid_size))

# Discount factor
gamma = 0.9

# Define policy (for now it can be random)
policy = {state: np.random.choice(actions) for state in states}

def is_terminal(state):
    return state == terminal_state

def step(state, action):
    if state == terminal_state:
        return state, 0

    i, j = state
    reward = 0  # Default reward
    next_state = state  # Default to staying in same position
    
    if action == 'up':
        new_i = i - 1
        if new_i < 0:  # Hit boundary
            next_state = (i, j)  # Stay in same position
            reward = -1
        else:
            next_state = (new_i, j)
    elif action == 'down':
        new_i = i + 1
        if new_i >= grid_size:  # Hit boundary
            next_state = (i, j)  # Stay in same position
            reward = -1
        else:
            next_state = (new_i, j)
    elif action == 'left':
        new_j = j - 1
        if new_j < 0:  # Hit boundary
            next_state = (i, j)  # Stay in same position
            reward = -1
        else:
            next_state = (i, new_j)
    elif action == 'right':
        new_j = j + 1
        if new_j >= grid_size:  # Hit boundary
            next_state = (i, j)  # Stay in same position
            reward = -1
        else:
            next_state = (i, new_j)
    
    # Check if we reached the terminal state
    if next_state == terminal_state:
        reward = 1
    
    return next_state, reward

def value_iteration():
    theta = 0.001  # threshold for stopping iteration
    while True:
        delta = 0
        for state in states:
            if is_terminal(state):
                continue

            v = V[state]
            max_value = float('-inf')
            for action in actions:
                next_state, reward = step(state, action)
                value = reward + gamma * V[next_state]
                max_value = max(max_value, float(value))

            V[state] = max_value
            delta = max(delta, abs(v - V[state]))

        if delta < theta:
            break

    return V

optimal_V = value_iteration()

def extract_policy():
    for state in states:
        if is_terminal(state):
            continue

        max_value = float('-inf')
        best_action = None
        for action in actions:
            next_state, reward = step(state, action)
            value = reward + gamma * optimal_V[next_state]
            if value > max_value:
                max_value = value
                best_action = action

        policy[state] = best_action

    return policy

# Extract the optimal policy from the optimal state-value function
optimal_policy = extract_policy()

# Print results
print("Optimal Values:")
print(optimal_V)
print("\nOptimal Policy:")
for i in range(grid_size):
    for j in range(grid_size):
        if (i, j) != terminal_state:
            print(policy[(i, j)], end='\t')
        else:
            print("G", end='\t')  # Goal
    print("")

```

## Policy Iteration

```python

# Policy Iteration Example
import numpy as np

# 定义网格世界环境参数
n_states = 16  # 4x4 网格
n_actions = 4  # 上下左右
gamma = 0.9

# 初始化随机策略和价值函数
policy = np.ones([n_states, n_actions]) / n_actions
value_function = np.zeros(n_states)

# 定义收敛精度
theta = 1e-10

def policy_evaluation(policy, value_function, gamma, theta):
    while True:
        delta = 0
        for state in range(n_states):
            v = value_function[state]
            new_value = 0
            for action in range(n_actions):
                next_state_prob = transition_prob[state, action]
                reward = reward_function[state, action]
                new_value += policy[state, action] * (reward + gamma * next_state_prob * value_function)
            value_function[state] = new_value
            delta = max(delta, np.abs(v - new_value))
        if delta < theta:
            break
    return value_function

def policy_improvement(value_function, policy, gamma):
    policy_stable = True
    for state in range(n_states):
        chosen_action = np.argmax(policy[state])
        action_values = np.zeros(n_actions)
        for action in range(n_actions):
            next_state_prob = transition_prob[state, action]
            action_values[action] = reward_function[state, action] + gamma * next_state_prob * value_function

        best_action = np.argmax(action_values)
        if chosen_action != best_action:
            policy_stable = False
            policy[state] = np.eye(n_actions)[best_action]

    return policy, policy_stable

# 示例实现策略迭代方法
def policy_iteration():
    policy_stable = False
    while not policy_stable:
        value_function = policy_evaluation(policy, value_function, gamma, theta)
        policy, policy_stable = policy_improvement(value_function, policy, gamma)
    
    return policy, value_function

# 调用策略迭代函数
optimal_policy, optimal_value_function = policy_iteration()

print("Optimal Policy:")
print(optimal_policy)
print("Optimal Value Function:")
print(optimal_value_function)

```
