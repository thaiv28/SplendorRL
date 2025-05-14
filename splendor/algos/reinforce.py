import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

from collections import defaultdict

from splendor.sample import BatchInfo
from splendor.algos.algorithm import Algorithm

class Reinforce(Algorithm):
    def __init__(self, policy_network, optimizer: str = "adam", lr: float = 1e-2):
        """Set policy network and optimizer for algorithm"""

        super().__init__(policy_network, optimizer, lr)

    def update(self, batch: BatchInfo):
        """Update parameters of network for a sampled batch using set optimizer"""

        self.optimizer.zero_grad()

        batch_losses = {}
        for agent in batch.obs:
            batch_loss = self.compute_loss(
                observations=torch.as_tensor(batch.obs[agent], dtype=torch.float32),
                actions=torch.as_tensor(batch.actions[agent], dtype=torch.int32),
                masks=torch.as_tensor(batch.masks[agent], dtype=torch.bool),
                rets=torch.as_tensor(batch.weights[agent], dtype=torch.float32)
            )
            batch_losses[agent] = torch.tensor(batch_loss, requires_grad=True)

        total_loss = torch.sum(torch.stack(list(batch_losses.values())))
        total_loss.backward()

        self.optimizer.step()

        return batch_losses

    def compute_loss(self, observations, masks, actions, rets):
        """Compute loss for a batch of observations, actions, and returns"""

        logprobs = self.policy_network._get_distribution(observations, masks).log_prob(actions)
        return -(logprobs * rets).mean()

