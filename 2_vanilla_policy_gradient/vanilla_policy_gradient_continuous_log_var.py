# Vanilla policy gradient using Pytorch, continuous action space
# MountainCarContinuous-v0 in Gym environment
# Algorithm can be found in https://spinningup.openai.com/en/latest/algorithms/vpg.html, http://joschu.net/docs/thesis.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import math
import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# policy network - gaussian mean and log(var)
class policy_network(nn.Module):
    def __init__(self, input=4, hidden=64, output=2):
        super(policy_network, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# state-value estimator (critic)
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

def choose_action(ob):
    # choose action using current policy pi (continuous action space)
    gau_pars = pi(torch.tensor(ob, dtype=torch.float32).to(device)).detach()
    gau_pars = gau_pars.cpu().numpy()
    std = math.sqrt(math.exp(gau_pars[1]))
    a = np.random.normal(loc=gau_pars[0], scale=std)
    # if np.random.uniform(0,1) >= epsilon:
    #     a = a
    # else:
    #     a = 2 * gau_pars[0] - a

    # if a < 0:
    #     a_env = 0
    # else:
    #     a_env = 1
    return np.array([a])


def compute_advantage(rewards, states, gamma):
    # get returns of states from a trajectory
    # R_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
    returns = rewards.copy()
    advs = rewards.copy()
    states_ep = states.copy()[-len(rewards):]
    for i in reversed(range(len(returns)-1)):
        r = returns[i] + gamma * returns[i+1]
        returns[i] = r
    states_ep = torch.tensor(states_ep, dtype=torch.float32).to(device)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    advs = returns - V(states_ep).squeeze(1)
    return advs    
    
def model_validate():
    ep_reward = 0
    ob = env.reset()
    done = False
    while not done:      
        a = choose_action(ob)
        ob_, r, done, _ = env.step(a)
        ep_reward += r
        ob = ob_
    return ep_reward

K = 1000000
BATCH_SIZE = 200
# epsilon_max = 0.2
# epsilon_min = 0.001
# epsilon_step = (epsilon_max - epsilon_min) / (1000)
GAMMA = 0.95
LEARNING_RATE = 0.005

env = gym.make('MountainCarContinuous-v0')

# n_actions = env.action_space.shape[0]
n_actions = 1
state_length = env.observation_space.shape[0]

# initialize policy network and state-value network
pi = policy_network(input=state_length, output=2).to(device)
V = state_value_network(input=state_length).to(device)

# define optimizers
pi_optimizer = torch.optim.Adam(pi.parameters(), lr=LEARNING_RATE)
V_optimizer = torch.optim.Adam(V.parameters(), lr=LEARNING_RATE)

# define loss
loss_MSE = nn.MSELoss(reduction='mean').to(device)
training_rewards = []
for k in range(K):
    # save trajectories
    states = []
    actions = []
    states_ = []
    advantages = torch.tensor([], dtype=torch.float32).to(device)
    # collect trajectories
    while len(states) < BATCH_SIZE:
        ob = env.reset()
        done = False
        rewards = [] 
        step_e = 0 # allowed steps in one episode       
        while not done:
            if step_e < 200:
                a = choose_action(ob)
                ob_, r, done, _ = env.step(a)
                # save trajectories
                states.append(ob)
                actions.append(a)
                rewards.append(r)
                states_.append(ob_)
                ob = ob_
                step_e += 1
            else:
                break
        # calculate discounted return and advantages
        advs = compute_advantage(rewards, states, GAMMA)
        advantages = torch.cat((advantages, advs), 0)

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.float32).to(device)
    # re-fit state-value function
    V_optimizer.zero_grad() # clear gradient
    v_loss = loss_MSE(advantages, V(states).squeeze())
    v_loss.backward(retain_graph=True)
    V_optimizer.step()
    # update policy pi
    pi_optimizer.zero_grad()
    vars_ = torch.exp(pi(states)[:,1]).unsqueeze(1)
    means =  pi(states)[:,0].unsqueeze(1)
    log_pi = - (((actions - means)**2)/vars_/2) - torch.log(torch.sqrt(2*np.pi*vars_))
    pi_loss = - torch.sum(torch.mul(log_pi.squeeze(), advantages)) # negative: .backward() use gradient descent, (-loss) with gradient descnet = gradient ascent
    pi_loss.backward()
    pi_optimizer.step()

    # validate current policy
    if k % 100 == 0:
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

ep_rewards = []
for ii in range(10):
    ep_reward = model_validate(0.0)
    ep_rewards.append(ep_reward)
print('Average rewards of last 10 eps: ', np.mean(ep_rewards))
