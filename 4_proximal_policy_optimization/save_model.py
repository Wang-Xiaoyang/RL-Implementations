"""Save model, load and render.
"""
import torch
import gym
import numpy as np
import matplotlib.pyplot as plt

class SaveModel():
    def __init__(self):
        return

    def save_model(self, path, networks, networks_name, optims, optims_name):
        """Save model.

        Args:
            path: the path to save model, 'x.pt'
        """
        save_dict = {}
        for i in range(len(networks_name)):
            save_dict[networks_name[i]] = networks[i].state_dict()
        for j in range(len(optims_name)):
            save_dict[optims_name[j]] = optims[j].state_dict()
        torch.save(save_dict, path)

    def load_model(self, path, networks, networks_name, optims, optims_name):
        # load model
        checkpoint = torch.load(path)
        for i in range(len(networks)):
            networks[i].load_state_dict(checkpoint[networks_name[i]])
        for j in range(len(optims)):
            optims[j].load_state_dict(checkpoint[optims_name[j]])