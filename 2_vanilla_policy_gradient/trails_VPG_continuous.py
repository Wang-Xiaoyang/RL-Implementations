# Vanilla policy gradient using Pytorch, continuous action space
# MountainCarContinuous-v0 in Gym environment
# Algorithm can be found in https://spinningup.openai.com/en/latest/algorithms/vpg.html, http://joschu.net/docs/thesis.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import matplotlib.pyplot as plt

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# policy network - gaussian mean and std (output=2)
# gaussian mean (output=1)
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

def choose_action(ob, policy_net):
    # choose action using current policy pi (continuous action space)
    # assume the variance of Gaussian policy \sigma = 1.0
    mean_a = policy_net(torch.tensor(ob, dtype=torch.float32).to(device)).detach()
    mean_a = mean_a.cpu().numpy()
    a = np.random.normal(loc=mean_a, scale=1.0)
    # if a < 0:
    #     a_env = 0
    # else:
    #     a_env = 1
    return np.array(a)

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
    
def model_validate(policy_net):
    ep_reward = 0
    ob = env.reset()
    done = False
    while not done:
        env.render()      
        a = choose_action(ob, policy_net)
        ob_, r, done, _ = env.step(a)
        ep_reward += r
        ob = ob_
    return ep_reward

K = 10000
# BATCH_SIZE = 500
GAMMA = 0.95
LEARNING_RATE = 0.0005

env = gym.make('MountainCarContinuous-v0')

n_actions = env.action_space.shape[0]
# n_actions = 1
state_length = env.observation_space.shape[0]

# initialize policy network and state-value network
pi = policy_network(input=state_length, output=1).to(device)
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
    ob = env.reset()
    done = False
    rewards = []
    ep_reward = 0.0       
    while not done:
        a = choose_action(ob, pi)
        ob_, r, done, _ = env.step(a)
        # save trajectories
        states.append(ob)
        actions.append(a)
        rewards.append(r)
        states_.append(ob_)
        ob = ob_
        ep_reward += r
    # calculate discounted return and advantages
    advs = compute_advantage(rewards, states, GAMMA)
    advantages = torch.cat((advantages, advs), 0)
    weight = torch.tensor([1.0] * len(rewards), dtype=torch.float32).to(device)
    advantages = torch.mul(advantages, weight)

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.float32).to(device)
    # re-fit state-value function
    V_optimizer.zero_grad() # clear gradient
    v_loss = loss_MSE(advantages, V(states).squeeze())
    v_loss.backward(retain_graph=True)
    V_optimizer.step()
    # update policy pi
    pi_optimizer.zero_grad()    
    log_pi = - (((actions - pi(states)))**2) / 2
    pi_loss = - torch.sum(torch.mul(log_pi.squeeze(), advantages)) # negative: .backward() use gradient descent, (-loss) with gradient descnet = gradient ascent
    pi_loss.backward()
    pi_optimizer.step()

    # print training process
    training_rewards.append(ep_reward)
    if k % 100 == 0:
        print('Step: ', k, '  Total reward: ', ep_reward)

plt.plot(training_rewards)
plt.title('VPG(weights)')
plt.show()

# save model
path = './vpg_w/vpg_w.pt'
torch.save({
            'policy_model': pi.state_dict(),
            'critic_model': V.state_dict(),
            'optimizer_policy_model': pi_optimizer.state_dict(),
            'optimizer_critic_model': V_optimizer.state_dict(),
            }, path)
# load model
pi_ = policy_network(input=state_length, output=1)
V_ = state_value_network(input=state_length, output=1)
pi_optimizer_ = torch.optim.Adam(pi_.parameters())
V_optimizer_ = torch.optim.Adam(V_.parameters())

checkpoint = torch.load(path)
pi_.load_state_dict(checkpoint['policy_model'])
V_.load_state_dict(checkpoint['critic_model'])
pi_optimizer_.load_state_dict(checkpoint['optimizer_policy_model'])
V_optimizer_.load_state_dict(checkpoint['optimizer_critic_model'])

# validate
ep_rewards = []
for ii in range(10):
    ep_reward = model_validate(pi_)
    ep_rewards.append(ep_reward)
print('Average rewards of last 10 eps: ', np.mean(ep_rewards))