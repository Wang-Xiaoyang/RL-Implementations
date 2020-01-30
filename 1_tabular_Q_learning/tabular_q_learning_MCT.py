# Solving FrozenLake problem using monte-carlo tree search (MCTS)
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

# initialize the tree as a tree_table
tree_table = []
for ii in range(env.observation_space.n):
    # possible actions [0 0 0 0]; time visited and value [0 0]; child states
    tree_table.append([0,0,0,0,0,0,[[],[],[],[]]])

def rollouts(num_eps, tree_table):
    # do a number of rollouts to update the tree
    for ii in range(num_eps):
        tree_table = one_rollout(tree_table)
    return tree_table

def one_rollout(tree_table):
    # do one rollout and backprob to update the tree
    ob = env.reset()
    trajectories_ = []
    rewards_ = []
    # do one rollout till end
    done  = False
    while not done:
        a = env.action_space.sample()
        ob_, r, done, _ = env.step(a)
        trajectories_.append([ob, a, r, ob_])
        rewards_.append(r)
        ob = ob_

    tree_table = backprob(tree_table, trajectories_, rewards_)
    return tree_table        

def backprob(tree_table, trajectories, rewards):
    # update nodes in reverse order
    cum_rewards = np.cumsum(rewards[::-1])[::-1] 
    for ii in reversed(range(len(trajectories))):
        # [ob, a, r, ob_]
        cur_traj = trajectories[ii]
        ob = cur_traj[0]
        a = cur_traj[1]
        tree_table[ob][4] += 1 # time visited update
        tree_table[ob][5] += cum_rewards[ii] # reward update
        tree_table[ob][a] = 1
        if cur_traj[3] not in tree_table[ob][6][a]:
            tree_table[ob][6][a].append(cur_traj[3])
    return tree_table

# do rollouts
tree_table = rollouts(1000, tree_table)
# count state values
state_value = []
for ii in range(env.observation_space.n):
    value = 1e-20
    if tree_table[ii][4] != 0:
        value = tree_table[ii][5] / tree_table[ii][4]
    state_value.append(value)
print('state value: \n', state_value)

# standardize
def standardize_data(data):
    min_ = np.min(data)
    max_ = np.max(data)
    data_ = data.copy()
    data_ = [(i - min_)/(max_ - min_) for i in data_]
    return data_

state_value_n = standardize_data(state_value)

# sort and visualization
reshaped = np.array(state_value_n).reshape((4,4))

fig, ax = plt.subplots()
min_val, max_val = 0, 4
ax.matshow(reshaped, cmap=plt.cm.Blues)

t = 0
for j in range(4):
    for i in range(4):
        ax.text(i, j, str(t), va='center', ha='center')
        t += 1
plt.show()

#-------------------------------------------------------------------------- 
# DON using confidence map (normalized or standardized) as instance reward
# not improving

gamma = 0.95
alpha = 0.05
max_steps = 1000
n_eps = 10000
beta = 0.10 # the weight of confidence map value in reward

epsilon_max = 1.0
epsilon_decay = 0.999


# zero initialization
q_table = np.zeros((env.observation_space.n, env.action_space.n))

ep_reward = []
epsilon = epsilon_max

for ii in range(n_eps):
    ob = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        # env.render()
        # epsilon-greedy
        if np.random.uniform(0,1) >= epsilon:
            a = np.argmax(q_table[ob])
        else:
            a = env.action_space.sample()
        ob_, r, done, _ = env.step(a)
        # update Q value
        q_table[ob][a] += alpha * ((r + beta * state_value_n[ob_]) + gamma * np.max(q_table[ob_]) - q_table[ob][a])
        ob = ob_
        total_reward += (r + beta * state_value_n[ob_])

    epsilon *= epsilon_decay
    ep_reward.append(total_reward)
    print('total reward of one episode is: ', total_reward)

print('Q table \n', q_table)

# print smmothed reward
def smooth_reward(ep_reward, smooth_over):
    smoothed_r = []
    for ii in range(smooth_over, len(ep_reward)):
        smoothed_r.append(np.mean(ep_reward[ii-smooth_over:ii]))
    return smoothed_r

plt.plot(smooth_reward(ep_reward, 100))
plt.title('smoothed reward over 100 episodes')
plt.show()

print('average score of the last 100 eps: ', np.mean(ep_reward[-100:]))
print('average score of all episodes: ', np.mean(ep_reward))

# game performance
ep_reward_validation = []
for ii in range(100):
    ob = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        a = np.argmax(q_table[ob])
        ob_, r, done, _ = env.step(a)
        total_reward += r
        ob = ob_
    ep_reward_validation.append(total_reward)

print('game score over 100 eps: ', np.mean(ep_reward_validation))
final_score.append(np.mean(ep_reward_validation))


#--------------------------------------------------------------------------
#----------------------------Not done--------------------------------------

# tree_table = []
# for ii in range(env.observation_space.n):
#     # unvisited? [[a, ob_], time visited, value]
#     tree_table.append([True, []])

# ob = env.reset()

# if tree_table[ob][0] == True:
#     # expand - add four new nodes
#     tree_table[ob][0] = False
#     for ii in range(4):
#         trajectories_ = []
#         rewards_ = []
#         ob_, r, done, _ = env.step(ii)
#         trajectories_.append([ob, a, r, ob_])
#         rewards_.append(r)
#         ob = ob_
#         # simulation
#         while not done:
#             a = env.action_space.sample()
#             ob_, r, done, _ = env.step(a)
#             trajectories_.append([ob, a, r, ob_])
#             rewards_.append(r)
#             ob = ob_
#             tree_table[ob][0] = False
#         tree_table = backprob(tree_table, trajectories_, rewards_)
# else:
#     # tree policy

# def rollouts(num_eps, tree_table):
#     # do a number of rollouts to initialize the tree
#     for ii in range(num_eps):
#         tree_table = one_rollout(tree_table)
#     return tree_table

# def one_rollout(tree_table):
#     # do one rollout and backprob to update the tree
#     ob = env.reset()
#     trajectories_ = []
#     rewards_ = []
#     # do one rollout till end
#     done  = False
#     while not done:
#         a = env.action_space.sample()
#         ob_, r, done, _ = env.step(a)
#         trajectories_.append([ob, a, r, ob_])
#         rewards_.append(r)
#         ob = ob_

#     tree_table = backprob(tree_table, trajectories_, rewards_)
#     return tree_table        

# def backprob(tree_table, trajectories, rewards):
#     # update nodes in reverse order
#     cum_rewards = np.cumsum(rewards[::-1])[::-1] 
#     for ii in reversed(range(len(trajectories))):
#         # [ob, a, r, ob_]
#         cur_traj = trajectories[ii]
#         ob = cur_traj[0]
#         ob_ = cur_traj[3]
#         a = cur_traj[1]
#         for jj in range(len(tree_table[ob])):
#             if [a, ob_] not in tree_table[ob][jj]:
#                 tree_table[ob].append([[a, ob_], 0, 0])
#         for jj in range(len(tree_table[ob])):
#             if tree_table[ob][jj][0] == [a, ob_]:
#                 tree_table[ob][jj][1] += 1 # time visited update
#                 tree_table[ob][jj][2] += cum_rewards[ii] # reward update
#     return tree_table

# tree_table = rollouts(1000, tree_table)

# # next time, choose action based on TreePolicy (Upper confidence bounds for trees, UCT here)
# ob = env.reset()
# while not done:
#     # get UCT value for possible four nodes
#     for ii in range(4):
#         env1
