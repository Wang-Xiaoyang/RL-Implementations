# Vanilla policy gradient using Pytorch
# CartPole-v0 in Gym environment
# Algorithm can be found in https://spinningup.openai.com/en/latest/algorithms/vpg.html, http://joschu.net/docs/thesis.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions.categorical import Categorical
import scipy.signal

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# policy network
class policy_network(nn.Module):
    def __init__(self, input=4, hidden=64, output=2):
        super(policy_network, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

# state-value estimator
class state_value_network(nn.Module):
    def __init__(self, input=4, hidden=64, output=1):
        super(state_value_network, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def pi_step(ob):
    # choose action using current policy pi (discrete action space)
    ob = torch.as_tensor(ob, dtype=torch.float32).to(device)
    probs = pi(ob).detach()
    pi_ = Categorical(probs)
    a = pi_.sample().unsqueeze(0)
    v = V(ob).detach()
    return a.numpy()[0], v.numpy()[0]

def gae_advantage(rewards, states, vals, gamma, lam):
    # get returns of states from a trajectory as state-value
    # R_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
    deltas = rewards[:-1] + gamma * vals[1:] - vals[:-1]
    advs = discount_cumsum(deltas, gamma * lam)
    return np.float32(advs)   

def reward_to_go(rewards, gamma):
    rtg = np.float32(discount_cumsum(rewards, gamma)[:-1])
    return rtg

def compute_loss_pi(states, actions, advs):
    probs = pi(states)
    # Note that this is equivalent to what used to be called multinomial
    m = Categorical(probs)
    loss = - torch.sum(torch.mul(m.log_prob(actions), advs))
    return loss

def model_validate():
    ep_reward = 0
    ob = env.reset()
    done = False
    while not done:      
        a, _ = pi_step(ob)
        ob_, r, done, _ = env.step(a)
        ep_reward += r
        ob = ob_
    return ep_reward

K = 2000
BATCH_SIZE = 100
GAMMA = 0.98
LAM = 0.97
LEARNING_RATE = 0.0005

env = gym.make('CartPole-v0')

n_actions = env.action_space.n
state_length = env.observation_space.shape[0]

# initialize policy network and state-value network
pi = policy_network(input=state_length, output=n_actions).to(device)
V = state_value_network(input=state_length).to(device)

# define optimizers
pi_optimizer = torch.optim.Adam(pi.parameters(), lr=LEARNING_RATE)
V_optimizer = torch.optim.Adam(V.parameters(), lr=LEARNING_RATE)

# define loss
loss_MSE = nn.MSELoss(reduction='sum').to(device)

training_rewards = []
for k in range(K):
    # save trajectories
    states = []
    actions = []
    rewards = []
    vals = []
    # collect trajectories
    ob = env.reset()
    done = False        
    while not done:
        a, v = pi_step(ob)
        ob_, r, done, _ = env.step(a)
        # save trajectories
        states.append(ob)
        actions.append(a)
        rewards.append(r)
        vals.append(v)
        ob = ob_
    
    # calculate discounted return and advantages
    rewards, states, vals = np.array(rewards), np.array(states), np.array(vals)
    advs = gae_advantage(rewards, states, vals, GAMMA, LAM)
    rtg = reward_to_go(rewards, GAMMA)

    states = torch.tensor(states[:-1], dtype=torch.float32).to(device)
    actions = torch.tensor(actions[:-1], dtype=torch.float32).to(device)
    vals = torch.tensor(vals[:-1], dtype=torch.float32).to(device)
    rtg = torch.tensor(rtg, dtype=torch.float32).to(device)
    advs = torch.tensor(advs, dtype=torch.float32).to(device)

    # re-fit state-value function as a baseline
    V_optimizer.zero_grad() # clear gradient
    v_loss = loss_MSE(rtg, V(states).squeeze())
    v_loss.backward(retain_graph=True)
    V_optimizer.step()
    # update policy pi
    pi_optimizer.zero_grad()
    pi_loss = compute_loss_pi(states, actions, advs)
    pi_loss.backward()
    pi_optimizer.step()

    # validate current policy
    training_reward = model_validate()
    training_rewards.append(training_reward)
    print('Step: ', k, '  Total reward: ', training_reward)

# smmothed reward
def smooth_reward(ep_reward, smooth_over):
    smoothed_r = []
    for ii in range(smooth_over, len(ep_reward)):
        smoothed_r.append(np.mean(ep_reward[ii-smooth_over:ii]))
    return smoothed_r

# plt.plot(smooth_reward(ep_rewards, 50))
plt.plot(training_rewards)
plt.show()

# final validation after training
ep_rewards = []
for ii in range(100):
    ep_reward = model_validate()
    ep_rewards.append(ep_reward)
print('Average rewards of last 100 eps: ', np.mean(ep_rewards))
