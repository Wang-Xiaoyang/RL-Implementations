# Soft Actor Critic (SAC) using Pytorch, continuous action space
# MountainCarContinuous-v0 in Gym environment
# Algorithm can be found in https://arxiv.org/abs/1801.01290, https://spinningup.openai.com/en/latest/algorithms/sac.html

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from collections import deque
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

# state-action-value estimator (Q function)
class Q_network(nn.Module):
    def __init__(self, input=4, hidden=64, output=1):
        super(Q_network, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# replay buffer using deque
class Memory():
    def __init__(self, max_size=2000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]

def choose_action(ob, policy_net):
    # choose action using current policy pi (continuous action space)
    # assume the variance of Gaussian policy \sigma = 1.0
    mean_a = policy_net(torch.tensor(ob, dtype=torch.float32).to(device)).detach()
    # mean_a = mean_a.cpu().numpy()
    # a = np.random.normal(loc=mean_a, scale=1.0)
    dist = torch.distributions.normal.Normal(mean_a, 1.0)
    act = dist.sample().item()
    return [act]
    # if a < 0:
    #     a_env = 0
    # else:
    #     a_env = 1
    # return np.array(a)

def choose_action_batch(ob, policy_net):
    # to test
    # choose actions using current policy pi (continuous action space)
    # given a batch of observations
    # assume the variance of Gaussian policy \sigma = 1.0
    mean_a = policy_net(ob).detach()
    dist = torch.distributions.normal.Normal(mean_a, 1.0)
    act = dist.sample().item()
    return [act], dist.log_prob(act)

def squashed_Gaussian(ob, policy_net):
    mean_a = policy_net(ob).detach()
    dist = torch.distributions.normal.Normal(0.0, 1.0)
    noise = dist.sample().item()
    return np.tanh(mean_a + 1.0 * noise)

def update_Q_target(Q, Q_target, rho):
    Q_state_dict = Q.state_dict()
    Q_target_state_dict = Q_target.state_dict()
    for name, param in Q_target_state_dict.items():
        param_ = rho * param + (1-rho) * Q_state_dict[name]
        Q_target_state_dict[name].copy_(param_) 


    
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

K = 10000 # number of iterations
D = 10000 # buffer length
N = 5 # number of eps in each iteration
U = 3 # number of updates per iteration
BATCH_SIZE = 500
GAMMA = 0.95
LEARNING_RATE = 0.0005
ALPHA = 0.90
RHO = 0.90

env = gym.make('MountainCarContinuous-v0')

n_actions = env.action_space.shape[0]
# n_actions = 1
state_length = env.observation_space.shape[0]

# initialize policy network and Q networks
pi = policy_network(input=state_length, output=1).to(device)
Q_1 = Q_network(input=(state_length+n_actions), output=1)
Q_2 = Q_network(input=(state_length+n_actions), output=1)
Q_1_target = Q_network(input=(state_length+n_actions), output=1)
Q_2_target = Q_network(input=(state_length+n_actions), output=1)
# copy parameters
Q_1_target.load_state_dict(Q_1.state_dict())
Q_2_target.load_state_dict(Q_2.state_dict())

# initialize memory buffer
memory = Memory(max_size=BUFFER_LEN)

# define optimizer
Q_1_optimizer = torch.optim.Adam(Q_1.parameters(), lr=LEARNING_RATE)
Q_2_optimizer = torch.optim.Adam(Q_2.parameters(), lr=LEARNING_RATE)
pi_optimizer = torch.optim.Adam(pi.parameters(), lr=LEARNING_RATE)

# define loss
loss_MSE = nn.MSELoss(reduction='mean').to(device)

training_rewards = []

#######

#########


for k in range(K):
    # save trajectories
    states = []
    actions = []
    states_ = []

    # collect trajectories
    for n in range(N):
        ob = env.reset()
        done = False
        rewards = []
        ep_reward = 0.0       
        while not done:
            a = choose_action(ob, pi)
            ob_, r, done, _ = env.step(a)
            memory.add((ob, a, r, ob_, int(done)))
            ob = ob_
    # update
    if len(memory.buffer) >= BATCH_SIZE:
        for u in range(U):
            batch = memory.sample(BATCH_SIZE)
            states = torch.FloatTensor([item[0] for item in batch]).to(device)
            actions = torch.FloatTensor([item[1] for item in batch]).to(device)
            rewards = torch.FloatTensor([item[2] for item in batch]).to(device)
            states_ = torch.FloatTensor([item[3] for item in batch]).to(device)
            done = torch.FloatTensor([item[4] for item in batch]).to(device)

            next_as, log_next_as = choose_action_batch(states_) # given next states, sample next actions from current policy
            # concatenate states_ and next_as as: next_states_actions
            # concatenate states and actions as: states_actions
            min_q_target = min(Q_1_target(next_states_actions), Q_2_target(next_states_actions)
            # to test
            q_targets = rewards + GAMMA * (1-done) * (min_q_target - ALPHA * log_next_as)
            # one step of back prop for Qs
            Q_1_optimizer.zero_grad() # clear gradient
            loss = loss_MSE(Q_1(states_actions), q_targets)
            loss.backward(retain_graph=True)
            Q_1_optimizer.step()
            #
            Q_2_optimizer.zero_grad() # clear gradient
            loss = loss_MSE(Q_2(states_actions), q_targets)
            loss.backward(retain_graph=True)
            Q_2_optimizer.step()
            # update policy
            pi_optimizer.zero_grad()
            a_samples = squashed_Gaussian(states, pi)
            # states_a_samples
            min_q = min(Q_1(states_a_samples), Q_2(states_a_samples))
            log_pi_a_samples = - (((a_samples - pi(states)))**2) / 2
            loss = mean(min_q - ALPHA * log_pi_a_samples)
            loss.backward()
            pi_optimizer.step()
            # update target Qs
            update_Q_target(Q_1, Q_1_target, RHO)
            update_Q_target(Q_2, Q_2_target, RHO)

    # print training process
    training_rewards.append(ep_reward)
    if k % 100 == 0:
        print('Step: ', k, '  Total reward: ', ep_reward)

plt.plot(training_rewards)
plt.title('VPG(a)')
plt.show()

# save model
path = './vpg/vpg.pt'
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
