from typing import Mapping
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

from collections import defaultdict

from splendor.sample import BatchInfo
from splendor.algos.algorithm import Algorithm

class Reinforce(Algorithm):
    def __init__(self, policy_network, optimizer: str = "adam", lr: float = 1e-2,
                 gamma: float = 0.99):
        """Set policy network and optimizer for algorithm"""

        super().__init__(policy_network, optimizer, lr, gamma)

    def update(self, batch: BatchInfo):
        """Update parameters of network for a sampled batch using set optimizer"""

        self.optimizer.zero_grad()

        batch_losses = {}
        rewards = Reinforce.batchify_rewards(batch.rewards, self.gamma)

        for agent in batch.obs:
            batch_loss = self.compute_loss(
                observations=torch.as_tensor(batch.obs[agent], dtype=torch.float32),
                actions=torch.as_tensor(batch.actions[agent], dtype=torch.int32),
                masks=torch.as_tensor(batch.masks[agent], dtype=torch.bool),
                rets=torch.as_tensor(rewards[agent], dtype=torch.float32)
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

    @staticmethod
    def batchify_rewards(rewards: Mapping[str, list[np.ndarray]], gamma) \
        -> dict[str, np.ndarray]:
        """
        Applies RTG and transforms rewards into advantages from lists of rewards,
        then returns as one array per player.

        Returns:
            dict[str, np.ndarray of shape (number rewards,)]
        """

        rtgs = Reinforce._compute_rewards_to_go(rewards, gamma)
        adv_rewards = Reinforce._compute_mean_baseline(rtgs)
        
        ret = {}
        for agent, traj_list in adv_rewards.items():
            ret[agent] = np.concatenate(traj_list)

        return ret

    @staticmethod
    def _compute_mean_baseline(rewards: Mapping[str, list[np.ndarray]]) \
        -> dict[str, list[np.ndarray]]:
        """
        Compute baseline as the mean of the remaining rewards in that trajectory.

        Returns:
            Mapping[str, list[np.ndarray]]: original rewards mapping with mean subtracted

        This baseline is constant over a trajectory, making it independent of state.
        However, it allows some rewards to be negative and others to be positive, as opposed
        to the alternative where all actions are always encouraged.
        """

        updated_rewards = {}

        for agent, traj_list in rewards.items():
            assert isinstance(traj_list, list)

            updated_traj_list = []
            for traj in traj_list:
                updated_traj_list.append(traj - np.mean(traj))
            updated_rewards[agent] = updated_traj_list

        return updated_rewards

    @staticmethod
    def _compute_rewards_to_go(rewards: Mapping[str, list[np.ndarray]], gamma) \
        -> dict[str, list[np.ndarray]]:
        """
        Compute "rewards to-go" for all agents.

        Rewards to-go is the remaining rewards in that episode.
        RTG example: [0, 5, 2, 6] -> [13, 13, 8, 6]

        Returns:
            dict[str, list[ndarray of shape (num_rewards,)]: original rewards mapping with RTG
        """

        ret = {}
        for agent, traj_list in rewards.items():
            rtgs_list = []
            for reward_arr in traj_list:
                assert reward_arr.ndim == 1

                n = len(reward_arr)
                rtgs = np.zeros(n)
                for i in reversed(range(n)):
                    rtgs[i] = reward_arr[i] + (gamma * rtgs[i+1] if i+1 < n else 0)
               
                rtgs_list.append(rtgs)


            ret[agent] = rtgs_list

        return ret
                
