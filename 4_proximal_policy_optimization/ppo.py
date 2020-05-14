"""Proximal policy optimization.

PPO-clip
https://spinningup.openai.com/en/latest/algorithms/ppo.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from policy_gradient_entities import policy_network_continuous as policy_network
from policy_gradient_entities import policy_gradient_comm_func, state_value_network
from save_model import SaveModel
import wandb

wandb.init(project="rl-implementation-basics-ppo")
# wandb config parameters
wandb.config.training_eps = int(1000)
wandb.config.gamma = 0.99
wandb.config.lr = 1e-5
wandb.config.batch_size = 100
wandb.config.epsilon = 0.1 # clip ratio
wandb.config.N = 2 # iters to update pi
config = wandb.config

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
SAVE = True

K = config.training_eps
N = config.N # 
BATCH_SIZE = config.batch_size
GAMMA = config.gamma
EPSILON = config.epsilon
LEARNING_RATE = config.lr

env = gym.make('MountainCarContinuous-v0')

n_actions = env.action_space.shape[0]
state_length = env.observation_space.shape[0]

# initialize policy network and state-value network
pi = policy_network(input=state_length, output=n_actions).to(device)
V = state_value_network(input=state_length).to(device)
# pi_old
pi_old = policy_network(input=state_length, output=n_actions).to(device)
pi_old.load_state_dict(pi.state_dict())
# define optimizers
pi_optimizer = torch.optim.Adam(pi.parameters(), lr=LEARNING_RATE)
V_optimizer = torch.optim.Adam(V.parameters(), lr=LEARNING_RATE)
# define MSE loss
loss_MSE = nn.MSELoss(reduction='mean').to(device)
# policy gradient function class init
pg_comm = policy_gradient_comm_func(device=device, pi=pi)

# other functions
def compute_advantage(returns, states, V):
    """
    """
    advs = returns - V(states).squeeze(1)
    return advs

def state_action_prob(states, actions, pi):
    """Current policy: the state-action probablility.

    Args:

    Returns:

    """
    std = 1.0
    mean_a = pi(states).squeeze()
    actions = actions.squeeze()
    e = - torch.pow(torch.div(actions-mean_a, std), 2) / 2
    s_a_prob = torch.div(torch.exp(e), (std * np.sqrt(2*math.pi)))
    return s_a_prob

def clip(ratio, advs, epsilon):
    """clip function in PPO.

    Args:

    Returns:

    """
    clipped_advs = advs.clone()
    for i in range(len(advs)):
        if advs[i] >= 0:
            clipped_advs[i] *= min(ratio[i], 1+epsilon)
        else:
            clipped_advs[i] *= min(ratio[i], 1-epsilon)
    return clipped_advs

# training
for k in range(K):
    ob = env.reset()
    done = False
    states = []
    actions = []
    states_ = []
    rewards = []
    ep_reward = 0.0
    # collect samples        
    while not done:
        a = pg_comm.choose_action_continuous(ob)
        ob_, r, done, _ = env.step(a)
        # save trajectories
        states.append(ob)
        actions.append(a)
        rewards.append(r)
        states_.append(ob_)
        ep_reward += r
        ob = ob_
    # calculate discounted return and advantages
    returns = pg_comm.reward_to_go(rewards, GAMMA) # discount?
    # to tensor
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.float32).to(device)
    returns = torch.tensor(returns, dtype=torch.float32).to(device)
    advs = compute_advantage(returns, states, V)
    for j in range(N):
        s_a_old = state_action_prob(states, actions, pi_old)
        s_a_new = state_action_prob(states, actions, pi)
        ratio = s_a_new / (s_a_old + 1e-9)
        # update policy
        pi_optimizer.zero_grad()
        pi_loss = - torch.mean(clip(ratio, advs, EPSILON))    
        pi_loss.backward(retain_graph=True)
        pi_optimizer.step()
    # update pi_old
    pi_old.load_state_dict(pi.state_dict())
    # re-fit state-value function
    V_optimizer.zero_grad() # clear gradient
    v_loss = loss_MSE(returns, V(states).squeeze())
    v_loss.backward(retain_graph=True)
    V_optimizer.step()

    # print training process
    wandb.log({"ep reward(training)": ep_reward,
                "v_net loss": v_loss,
                "pi loss": pi_loss})

# save model
if SAVE:
    model_entities = SaveModel()
    path = './ppo/ppo.pt'
    networks = [pi, V]
    networks_name = ['policy_model', 'value_model']
    optims = [pi_optimizer, V_optimizer]
    optims_name = ['optimizer_policy_model', 'optimizer_value_model']
    model_entities.save_model(path, networks, networks_name, optims, optims_name)

    wandb.save('./ppo/ppo.pt')
    wandb.save('../logs/*ckpt*')
    wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))

# load and render
model_entities = SaveModel()
pi_ = policy_network(input=state_length, output=n_actions).to(device)
V_ = state_value_network(input=state_length).to(device)
pi_optimizer_ = torch.optim.Adam(pi.parameters(), lr=LEARNING_RATE)
V_optimizer_ = torch.optim.Adam(V.parameters(), lr=LEARNING_RATE)
path = './ppo/ppo.pt'
networks = [pi_, V_]
networks_name = ['policy_model', 'value_model']
optims = [pi_optimizer_, V_optimizer_]
optims_name = ['optimizer_policy_model', 'optimizer_value_model']
model_entities.load_model(path, networks, networks_name, optims, optims_name)
pg_comm_ = policy_gradient_comm_func(device=device, pi=pi_)
# validate trained model
def model_validate(policy_net):
    ep_reward = 0
    ob = env.reset()
    done = False
    while not done:
        env.render()      
        a = pg_comm_.choose_action_continuous(ob)
        ob_, r, done, _ = env.step(a)
        ep_reward += r
        ob = ob_
    return ep_reward

ep_rewards = []
for ii in range(5):
    ep_reward = model_validate(pi)
    ep_rewards.append(ep_reward)
print('Average rewards of last 10 eps: ', np.mean(ep_rewards))