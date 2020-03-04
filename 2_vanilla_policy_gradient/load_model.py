# load model and render
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

env = gym.make('MountainCarContinuous-v0')
n_actions = env.action_space.shape[0]
state_length = env.observation_space.shape[0]

# load model
path = './vpg/vpg.pt'
# path = './vpg_ICM/vpg_icm.pt'
# path = './vpg_w/vpg_w.pt'

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