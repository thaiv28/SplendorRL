from collections import defaultdict
from typing import Mapping

import torch
from pettingzoo import AECEnv
import numpy as np

from splendor.environment import SplendorEnv
from splendor.policy import MLP

class BatchInfo:
    def __init__(self, batch_obs, batch_masks, batch_actions, batch_rewards, info=None):
        """
        Convert batch information to numpy array if applicable.

        Args:
        batch_obs - dict[agent, list of 1d arrays of size observation space]
        batch_masks - dict[agent, list of 1d arrays of size action space]
        batch_actions - dict[agent, list of scalars representing actions]
        batch_rewards - dict[agent, list of 1d arrays of size episode_length]
                Outer list: List of episodes 
                Inner list: List of scalar rewards for that episode

        Batch_rewards are nested lists because the inner lists represent an episode.
        We keep it this way so that the algorithm can decide how to calculate returns,
        and we give it an easy way to determine where episodes start and end.

        Returns:
        batch_obs - dict[agent, 2d array of shape (num_observations, observation_space)
        batch_masks - dict[agent, 2d array of shape (num_actions, action_space)
        batch_actions - dict[agent, 1d array of shape (action_space,)]
        batch_rewards - dict[agent, list of 1d arrays of size episode length)]

        Return value will be a dictionary of type [agent_name (str), np.ndarray]. If the
        input is a nested list of ints (as it should be for observations and masks), 
        the return values will be a 2d numpy array.
        """

        self.obs: dict[str, np.ndarray] = BatchInfo.convert_to_numpy(batch_obs)
        self.masks: dict[str, np.ndarray] = BatchInfo.convert_to_numpy(batch_masks, dtype=bool)
        self.actions: dict[str, np.ndarray] = BatchInfo.convert_to_numpy(batch_actions)
        self.rewards: Mapping[str, list[np.ndarray]] = defaultdict(lambda: [])
        for agent, rewards_list in batch_rewards.items():
            for ep_rewards in rewards_list:
                self.rewards[agent].append(np.array(ep_rewards))                                 

        if info is None:
            self.info = {}
        else:
            self.info = info

        for agent in self.obs:
            n = len(self.obs[agent])

            if (len(self.masks[agent]) != n or
                len(self.actions[agent]) != n or
                sum(len(l) for l in self.rewards[agent]) != n):
                raise ValueError("Observations, masks, actions, and weights are differing sizes")

            assert self.obs[agent].ndim == 2
            assert self.masks[agent].ndim == 2
            assert self.actions[agent].ndim == 1


    @staticmethod
    def convert_to_numpy(batch_nums, dtype=None):
        """Convert dictionary of lists to dictionary of np arrays"""

        if isinstance(batch_nums, dict):
            if dtype:
                return {k: np.array(v, dtype=dtype) for k, v in batch_nums.items()}
            else:
                return {k: np.array(v) for k, v in batch_nums.items()}

        raise ValueError("Type of batch information is unsupported", type(batch_nums))



def sample_one_epoch(env: AECEnv, policy: MLP, batch_size: int, render=False) -> BatchInfo:
    """
    Sample one epoch from the given environment.

    Returns batch observations, masks, actions, weights, and info dict.
    """

    batch_obs = defaultdict(lambda: [])
    batch_masks = defaultdict(lambda: [])
    batch_actions = defaultdict(lambda: [])
    batch_rewards = defaultdict(lambda: [])
    batch_returns = defaultdict(lambda: [])
    batch_lens = defaultdict(lambda: [])

    env.reset()
    ep_rewards = defaultdict(lambda: [])

    for agent in env.agent_iter():
        if render:
            env.render()

        while True:
            # get observation for current agent
            obs, _, term, trunc, info = env.last()

            if term or trunc:
                break

            mask = info["action_mask"]
            action = policy.get_action(torch.as_tensor(obs, dtype=torch.float32),
                                       mask=torch.from_numpy(mask))
            env.step(action)

            # get reward for action just taken
            _, reward, _, _, _ = env.last()

            batch_obs[agent].append(obs)
            batch_masks[agent].append(mask)
            batch_actions[agent].append(action)
            ep_rewards[agent].append(reward)

            # If the reward is negative, the agent has taken a move that allows them to go again.
            # Therefore, we continue without switching to the next agent.
            # TODO: implement better signal for repeating moves that doesn't 
            # involve reward. i.e. info dict
            if reward < 0:
                continue
            else:
                env.next_agent()
                break

        if term or trunc:
            agent_list = ep_rewards.keys()

            # The "win" reward (+100) must be manually added to our rewards 
            # because it is never returned by env.last(). We purposely keep
            # it in the rewards dict (not cumulative rewards) so it only gets
            # accounted for here.
            for agent in env.rewards:
                ep_rewards[agent][-1] += env.rewards[agent]
   
            ep_returns = {a: sum(rews) for a, rews in ep_rewards.items()}
            ep_lengths = {a: len(rews) for a, rews in ep_rewards.items()}

            for agent in agent_list:
                # Update list of returns and lengths for debugging
                batch_rewards[agent].append(ep_rewards[agent])
                batch_returns[agent].append(ep_returns[agent])
                batch_lens[agent].append(ep_lengths[agent])

                # Update the batch rewards that will be used for training

            env.reset()
            ep_rewards = defaultdict(lambda: [])

            if len(batch_obs[agent]) > batch_size:
                break

    info = {
        "returns": batch_returns,
        "lengths": batch_lens
    }

    return BatchInfo(
        batch_obs,
        batch_masks,
        batch_actions, 
        batch_rewards,
        info
    )







