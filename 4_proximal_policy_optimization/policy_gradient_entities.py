"""
For policy gradient family:

Policy network and state-value network.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# policy network
class policy_network_discrete(nn.Module):
    def __init__(self, input=4, hidden=64, output=2):
        super(policy_network_discrete, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

class policy_network_continuous(nn.Module):
    def __init__(self, input=4, hidden=64, output=2):
        super(policy_network_continuous, self).__init__()
        self.fc1 = nn.Linear(input, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, output)

    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# state-value estimator
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

class policy_gradient_comm_func():
    """Class for common functions used in policy gradient methods.

    Pytorch platform.
    """
    def __init__(self, device, pi):
        self.device = device
        self.pi = pi

    def choose_action_discrete(self, ob):
        """Choose action using current policy (discrete action space).

        Args:

        Returns:

        """
        # choose action using current policy pi (discrete action space)
        probs = self.pi(torch.tensor(ob, dtype=torch.float32).to(self.device)).detach()
        probs = probs.cpu().numpy()
        a = np.random.choice(range(len(probs)), p=probs)
        return a

    def choose_action_continuous(self, ob):
        """Choose action using current policy pi (continuous action space).

        Policy network only estimates the mean of a Gaussian, while the std \sigma = 1.0

        Args:

        Returns:

        """
        mean_a = self.pi(torch.tensor(ob, dtype=torch.float32).to(self.device)).detach()
        mean_a = mean_a.cpu().numpy()
        a = np.random.normal(loc=mean_a, scale=1.0)
        return np.array(a)

    def reward_to_go(self, rewards, gamma=1.0):
        """Reward to go (discounted and undiscounted)

        Args:

        Returns:

        """
        # get returns of states from a trajectory
        # R_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
        returns = rewards.copy()
        for i in reversed(range(len(returns)-1)):
            r = returns[i] + gamma * returns[i+1]
            returns[i] = r
        return returns 