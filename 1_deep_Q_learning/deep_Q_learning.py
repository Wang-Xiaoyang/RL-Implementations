# Deep Q Learning using Pytorch
# CartPole in Gym environment
# Algorithm can be found in https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Q networks
class Q_network(nn.Module):
    def __init__(self, input=4, hidden=50, output=2):
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
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, experience):
        self.buffer.append(experience)
            
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), 
                               size=batch_size, 
                               replace=False)
        return [self.buffer[ii] for ii in idx]

N_EPS = 1000
BUFFER_LEN = 1000
BATCH_SIZE = 100
GAMMA = 0.95
LEARNING_RATE = 0.01
UPDATE_STEP = 10
epsilon_s = 1.0
epsilon_decay = 0.995


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

epsilon = epsilon_s

step = 0
ep_rewards = []
for ii in range(N_EPS):
    ob = env.reset()
    done = False
    ep_reward = 0
    while not done:
        # epsilon greedy
        if np.random.uniform(0,1) >= epsilon:
            ob_tensor = torch.FloatTensor([ob])
            a = Q(ob_tensor).max(1)[1].data.numpy()[0]
        else:
            a = env.action_space.sample()
        ob_, r, done, _ = env.step(a)
        ep_reward += r
        # add to memory
        memory.add((ob, a, r, ob_))
        ob = ob_
        if len(memory.buffer) > BATCH_SIZE:
            batch = memory.sample(BATCH_SIZE)
            # get state, action, reward and the next state
            states = torch.FloatTensor([item[0] for item in batch])
            actions = torch.LongTensor([item[1] for item in batch])
            rewards = torch.FloatTensor([item[2] for item in batch])
            states_ = torch.FloatTensor([item[3] for item in batch])
            
            # if next state is the terminal state
            non_final_next_states = torch.cat([s for s in states_ if s is not None])
            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, states_))) # ?
            q_targets = rewards
            q_targets[non_final_mask] = rewards + GAMMA * Q_.forward(states_).max(1)[0]
            # q_targets = rewards + GAMMA * Q_.forward(states_).max(1)[0]
            q_targets = torch.unsqueeze(q_targets, 1)
            q_values = Q(states).gather(1, actions.unsqueeze(1))

            # one step of back prob
            optimizer.zero_grad() # clear gradient
            loss = loss_MSE(q_targets, q_values)
            loss.backward()
            optimizer.step()
            
            if step % UPDATE_STEP == 0:
                Q_.load_state_dict(Q.state_dict())
            step += 1
    
    epsilon *= epsilon_decay
    ep_rewards.append(ep_reward)
    print('Episode: ', ii, '  Total reward: ', ep_reward)

plt.plot(ep_rewards)
plt.show()