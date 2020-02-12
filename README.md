# RL-Implementations
Implementations for classic reinforcement learning algorithms, using Gym environment.

## 0 - Tabular Q Learning
tabular_q_learning.py: solving FrozenLake-v0 using tabular Q learning.

tabular_q_learning_MCT.py: using Monto-Carlo Tree to collect samples and calculating the value of each state.

confidence_map_MCT_1M_rollouts.png: the Monte-Carlo confidence map (values) of each state after running 1M simulations.

## 1 - Deep Q Learning
deep_Q_learning.py: solving CartPole-v0 using deep Q learning, implemented with Pytorch.

rewards.png: episode rewards for 5k training episodes using DQN.

deep_Q_learning_extension.py: solving CartPole-v0 using improved DQN, here double DQN is implemented.

rewards_double_DQN.png: episode rewards for 10k training episodes using double DQN.


