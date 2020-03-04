# Vanilla policy gradient (VPG) using Pytorch, continuous action space
# VPG Algorithm can be found in https://spinningup.openai.com/en/latest/algorithms/vpg.html, http://joschu.net/docs/thesis.pdf
# Intrinsic Curiosity Module (ICM): https://arxiv.org/abs/1705.05363
# MountainCarContinuous-v0 in Gym environment


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

# forward dynamics model - s_{t+1} = f(s_t, a_t)
class forward_dynamics_network(nn.Module):
    def __init__(self, input=5, hidden=64, output=4):
         super(forward_dynamics_network, self).__init__()
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
    # assume the variance of Gaussian policy \sigma = 1.0
    mean_a = pi(torch.tensor(ob, dtype=torch.float32).to(device)).detach()
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
    # advs = returns
    return advs

def compute_epsilon_adv(states, epsilon):
    # get another adv according to the discrepancy of current state to its average
    states_o = states.copy()
    states = np.array(states)
    means = []
    advs = []
    for i in range(len(states[0])):
        mean_ = np.mean(states[:,i])
        means.append(mean_)
    for i in range(len(states_o)):
        a = states_o[i]
        e = a - means
        cur_error = sum(t*t for t in e)
        advs.append(cur_error)
    if np.random.uniform() < epsilon:
        value = [t * 100 for t in advs]
    else:
        value = [t/t for t in advs]
    return value

def intrinsic_r(ob, a, ob_):
    # calculated the intrinsic reward as prediction error
    loss = nn.MSELoss(reduction='sum').to(device)
    input_tensor = torch.tensor(np.concatenate((ob, a)), dtype=torch.float32).to(device)
    ob_tensor = torch.tensor(ob, dtype=torch.float32).to(device)
    ob_p = forward_net(input_tensor).detach()
    pred_error = loss(ob_p, ob_tensor)
    return pred_error.item() # as a number
    
def model_validate():
    ep_reward = 0
    ob = env.reset()
    done = False
    while not done:
        env.render()      
        a = choose_action(ob)
        ob_, r, done, _ = env.step(a)
        ep_reward += r
        ob = ob_
    return ep_reward

K = 10000
BATCH_SIZE = 500
GAMMA = 0.95
ETA = 0.9 # regularizer of loss function
LEARNING_RATE = 0.0005
decay = 0.995

env = gym.make('MountainCarContinuous-v0')

n_actions = env.action_space.shape[0]
# n_actions = 1
state_length = env.observation_space.shape[0]

# initialize policy network, forward dynamics network and state-value network
pi = policy_network(input=state_length, output=1).to(device)
# forward_net = forward_dynamics_network(input=(n_actions+state_length), output=state_length).to(device)
V = state_value_network(input=state_length).to(device)

# define optimizers
# params = list(pi.parameters()) + list(forward_net.parameters())
# global_optimizer = torch.optim.Adam(params, lr=LEARNING_RATE)
V_optimizer = torch.optim.Adam(V.parameters(), lr=LEARNING_RATE)
pi_optimizer = torch.optim.Adam(pi.parameters(), lr=LEARNING_RATE)
# forward_net_optimizer = torch.optim.Adam(forward_net.parameters(), lr=LEARNING_RATE)

# define loss
loss_MSE = nn.MSELoss(reduction='mean').to(device)

training_rewards = []

# epsilon = 1.0
epsilon = 0.0

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
    step_e = 0 # allowed steps in one episode
    total_r = 0       
    while not done:
        a = choose_action(ob)
        ob_, r_e, done, _ = env.step(a)
        # r_i = intrinsic_r(ob, a, ob_) #add intrinsic reward
        # r = r_i + r_e
        r = r_e
        # save trajectories
        states.append(ob)
        actions.append(a)
        rewards.append(r)
        states_.append(ob_)
        ob = ob_
        step_e += 1
        total_r += r_e
    # calculate discounted return and advantages
    advs = compute_advantage(rewards, states, GAMMA)
    advantages = torch.cat((advantages, advs), 0)

    adv_epsilon = compute_epsilon_adv(states, epsilon)
    adv_epsilon = torch.tensor(adv_epsilon, dtype=torch.float32).to(device)
    advantages = torch.mul(advantages, adv_epsilon)
    # epsilon = np.max((epsilon*decay, 0.0001))

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.float32).to(device)
    states_actions = torch.cat((states, actions), 1).to(device)
    states_ = torch.tensor(states_, dtype=torch.float32).to(device)
    # update policy pi and forward dynamics model
    # global_optimizer.zero_grad()    
    # log_pi = - (((actions - pi(states)))**2) / 2
    # pi_loss = - torch.sum(torch.mul(log_pi.squeeze(), advantages)) # negative: .backward() use gradient descent, (-loss) with gradient descnet = gradient ascent
    # forward_net_loss = loss_MSE(states_, forward_net(states_actions))
    # global_loss = (1-ETA) * pi_loss + ETA * forward_net_loss
    # global_loss.backward(retain_graph=True)
    # global_optimizer.step()
    #
    pi_optimizer.zero_grad()    
    log_pi = - (((actions - pi(states)))**2) / 2
    pi_loss = - torch.sum(torch.mul(log_pi.squeeze(), advantages)) # negative: .backward() use gradient descent, (-loss) with gradient descnet = gradient ascent
    pi_loss.backward(retain_graph=True)
    pi_optimizer.step()

    V_optimizer.zero_grad() # clear gradient
    v_loss = loss_MSE(advantages, V(states).squeeze())
    v_loss.backward()
    V_optimizer.step()

    # validate current policy
    if k % 50 == 0:
        # if k % 500 == 0:
        #     training_reward = model_validate()
        #     training_rewards.append(training_reward)
        #     print('Step: ', k, '  Total reward (model validation): ', training_reward)
        print('Step: ', k, '  Total reward: ', total_r, 'epsilon', epsilon)
        # re-fit state-value network (critic)

    training_rewards.append(total_r)

# plt.plot(smooth_reward(ep_rewards, 50))
plt.plot(training_rewards)
plt.show()

ep_rewards = []
for ii in range(10):
    ep_reward = model_validate()
    ep_rewards.append(ep_reward)
print('Average rewards of last 10 eps: ', np.mean(ep_rewards))
# Vanilla policy gradient using Pytorch, continuous action space
