# Improved Deep Q Learning using Pytorch
# CartPole-v0 in Gym environment

# Extension 1: Double Q-learning (DeepMind, 2016)
# https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12389

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Q networks
class Q_network(nn.Module):
    def __init__(self, input=4, hidden=64, output=2):
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

def choose_action(ob, epsilon):
    # choose action using epsilon greedy given the current ob
    if np.random.uniform(0,1) >= epsilon:
        ob_tensor = torch.FloatTensor([ob])
        a = Q(ob_tensor).max(1)[1].detach()
        a = a.data.numpy()[0]
    else:
        a = env.action_space.sample()
    return a

def model_validate():
    ep_reward = 0
    ob = env.reset()
    done = False
    while not done:
        # validate model for current Q network        
        ob_tensor = torch.FloatTensor([ob])
        a = Q(ob_tensor).max(1)[1].detach()
        a = a.data.numpy()[0]
        ob_, r, done, _ = env.step(a)
        ep_reward += r
        ob = ob_
    return ep_reward

N_EPS = 5000
BUFFER_LEN = 5000
BATCH_SIZE = 100
GAMMA = 0.95
LEARNING_RATE = 0.001
UPDATE_STEP = 200
epsilon_max = 1.0
epsilon_min = 0.001
epsilon_step = (epsilon_max - epsilon_min) / N_EPS # how to decay epsilon

env = gym.make('CartPole-v0')

n_actions = env.action_space.n
state_length = env.observation_space.shape[0]

# initialize memory buffer
memory = Memory(max_size=BUFFER_LEN)

# initialize action-value and target action-value function: Q, Q^
Q = Q_network(input=state_length, output=n_actions)
Q_ = Q_network(input=state_length, output=n_actions)
# copy parameters
Q_.load_state_dict(Q.state_dict())
# define optimizer
optimizer = torch.optim.Adam(Q.parameters(), lr=LEARNING_RATE)
# define loss
loss_MSE = nn.MSELoss()

epsilon = epsilon_max

step = 0
ep_rewards = []
for ii in range(N_EPS):
    ob = env.reset()
    done = False
    while not done:
        a = choose_action(ob, epsilon)
        ob_, r, done, _ = env.step(a)
        # add to memory
        memory.add((ob, a, r, ob_, done))
        ob = ob_
    if len(memory.buffer) >= BATCH_SIZE:
        batch = memory.sample(BATCH_SIZE)
        # get state, action, reward and the next state
        states = torch.FloatTensor([item[0] for item in batch])
        actions = torch.LongTensor([item[1] for item in batch])
        rewards = torch.FloatTensor([item[2] for item in batch])
        states_ = torch.FloatTensor([item[3] for item in batch])
        done = [item[4] for item in batch]

        # if next state is the terminal state
        non_final_mask = torch.tensor(list(map(lambda s: s is not True, done)), dtype=torch.float32) #
        # q_targets is the main difference between DQN and double DQN
        next_as = Q(states_).max(1)[1].detach() # next actions according to Q
        next_q_values = Q_(states_).gather(1, next_as.unsqueeze(1))
        q_targets = rewards + GAMMA * torch.mul(next_q_values, non_final_mask)
        q_targets = torch.unsqueeze(q_targets, 1)
        q_values = Q(states).gather(1, actions.unsqueeze(1))

        # one step of back prop
        optimizer.zero_grad() # clear gradient
        loss = loss_MSE(q_targets, q_values)
        loss.backward()
        optimizer.step()
        
        if step % UPDATE_STEP == 0:
            Q_.load_state_dict(Q.state_dict())
        step += 1
    
    epsilon = max((epsilon-epsilon_step), epsilon_min)
    ep_reward = model_validate()
    ep_rewards.append(ep_reward)
    print('Episode: ', ii, '  Total reward: ', ep_reward, 'Epsilon: ', epsilon)

# smmothed reward
def smooth_reward(ep_reward, smooth_over):
    smoothed_r = []
    for ii in range(smooth_over, len(ep_reward)):
        smoothed_r.append(np.mean(ep_reward[ii-smooth_over:ii]))
    return smoothed_r

# plt.plot(smooth_reward(ep_rewards, 50))
plt.plot(ep_rewards)
plt.show()

for ii in range(100):
    ep_reward = model_validate()
    ep_rewards.append(ep_reward)
print('Average rewards of last 100 eps: ', np.mean(ep_rewards[-100:]))