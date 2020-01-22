"""
Implementation 1: Tabular Q learning
Env: FrozenLake-V0 in https://gym.openai.com/envs/#toy_text
Reference: http://www.incompleteideas.net/book/RLbook2018.pdf
"""
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

gamma = 0.95
alpha = 0.05
max_steps = 1000
n_eps = 10000

epsilon_max = 1.0
epsilon_decay = 0.999

# # 1. random initialization
# q_table = np.random.rand(env.observation_space.n, env.action_space.n)
## Q(terminal, *) = 0
# terminal_s = [5,7,11,12,15]
# q_table[terminal_s] = np.array([0,0,0,0])
# 2. zero initialization
q_table = np.zeros((env.observation_space.n, env.action_space.n))

ep_reward = []
epsilon = epsilon_max

for ii in range(n_eps):
    ob = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        env.render()
        # epsilon-greedy
        if np.random.uniform(0,1) >= epsilon:
            a = np.argmax(q_table[ob])
        else:
            a = env.action_space.sample()
        ob_, r, done, _ = env.step(a)
        # update Q value
        q_table[ob][a] += alpha * (r + gamma * np.max(q_table[ob_]) - q_table[ob][a])
        ob = ob_
        total_reward += r

    epsilon *= epsilon_decay
    ep_reward.append(total_reward)
    print('total reward of one episode is: ', total_reward)

print('Q table \n', q_table)

# print smmothed reward
def smooth_reward(ep_reward, smooth_over):
    smoothed_r = []
    for ii in range(smooth_over, len(ep_reward)):
        smoothed_r.append(np.mean(ep_reward[ii-smooth_over:ii]))
    return smoothed_r

plt.plot(smooth_reward(ep_reward, 100))
plt.title('smoothed reward over 100 episodes')
plt.show()

print('average score of the last 100 eps: ', np.mean(ep_reward[-100:]))
print('average score of all episodes: ', np.mean(ep_reward))