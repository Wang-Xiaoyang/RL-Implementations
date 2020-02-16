# Vanilla policy gradient using Pytorch
# CartPole-v0 in Gym environment
# Algorithm can be found in https://spinningup.openai.com/en/latest/algorithms/vpg.html, http://joschu.net/docs/thesis.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import matplotlib.pyplot as plt

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

def choose_action(ob):
    # choose action using current policy pi (discrete action space)
    probs = pi(torch.tensor(ob, dtype=torch.float32).to(device)).detach()
    probs = probs.cpu().numpy()
    a = np.random.choice(range(len(probs)), p=probs)
    return a

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
    # for j in range(len(states_ep)):
    #     adv = returns[j] - V(states_ep[j])
    #     advs[j] = adv
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

K = 1000
BUFFER_LEN = 10000
BATCH_SIZE = 50
GAMMA = 0.95
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
loss_NLL = nn.NLLLoss().to(device)

training_rewards = []
for k in range(K):
    # save trajectories
    states = []
    actions = []
    states_ = []
    advantages = torch.tensor([], dtype=torch.float32)
    # collect trajectories
    while len(states) < BATCH_SIZE:
        ob = env.reset()
        done = False
        rewards = []        
        while not done:
            a = choose_action(ob)
            ob_, r, done, _ = env.step(a)
            # save trajectories
            states.append(ob)
            actions.append(a)
            rewards.append(r)
            states_.append(ob_)
            ob = ob_
        # calculate discounted return and advantages
        advs = compute_advantage(rewards, states, GAMMA)
        advantages = torch.cat((advantages, advs), 0)

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).to(device).unsqueeze(1)
    # re-fit state-value function
    V_optimizer.zero_grad() # clear gradient
    v_loss = loss_MSE(advantages, V(states))
    v_loss.backward(retain_graph=True)
    V_optimizer.step()
    # update policy pi
    pi_optimizer.zero_grad()
    log_softmax = torch.log(pi(states).gather(1, actions))
    pi_loss = torch.sum(- torch.mul(log_softmax.squeeze(1), advantages))
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

ep_rewards = []
for ii in range(100):
    ep_reward = model_validate()
    ep_rewards.append(ep_reward)
print('Average rewards of last 100 eps: ', np.mean(ep_rewards))
