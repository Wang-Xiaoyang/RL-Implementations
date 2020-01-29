# Solving FrozenLake problem using monte-carlo tree search (MCTS)
import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')

tree_table = []
for ii in range(env.observation_space.n):
    # possible actions [0 0 0 0]; time visited and value [0 0]; child state
    tree_table.append([0,0,0,0,0,0,[[],[],[],[]]])

def rollouts(num_eps, tree_table):
    # do a number of rollouts to initialize the tree
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

tree_table = rollouts(1000000, tree_table)

state_value = []
for ii in range(env.observation_space.n):
    value = 1e-20
    if tree_table[ii][4] != 0:
        value = tree_table[ii][5] / tree_table[ii][4]
    state_value.append(value)

print(state_value)

sorted_value = state_value.copy()
sorted_value.sort()

idx = []
for ii in range(len(sorted_value)):
    idx_ = state_value.index(sorted_value[ii])
    idx.append(idx_)

# reshaped = np.array(idx).reshape((4,4))
reshaped = np.array(state_value).reshape((4,4))

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
