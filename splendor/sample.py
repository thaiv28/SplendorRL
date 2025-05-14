from collections import defaultdict

import torch
from pettingzoo import AECEnv
import numpy as np

from splendor.environment import SplendorEnv
from splendor.policy import MLP

class BatchInfo:
    def __init__(self, batch_obs, batch_masks, batch_actions, batch_weights, info=None):
        """Convert batch information to numpy array if applicable."""

        self.obs: dict[str, np.ndarray] = BatchInfo.convert_to_numpy(batch_obs)
        self.masks: dict[str, np.ndarray] = BatchInfo.convert_to_numpy(batch_masks)
        self.actions: dict[str, np.ndarray] = BatchInfo.convert_to_numpy(batch_actions)
        self.weights: dict[str, np.ndarray] = BatchInfo.convert_to_numpy(batch_weights)
        if info is None:
            self.info = {}
        else:
            self.info = info

    @staticmethod
    def convert_to_numpy(batch_nums):
        """Convert dictionary of lists to dictionary of np arrays"""

        if isinstance(batch_nums, dict):
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
    batch_weights = defaultdict(lambda: [])
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
                batch_returns[agent].append(ep_returns[agent])
                batch_lens[agent].append(ep_lengths[agent])

                # Update the batch weights that will be used for training
                batch_weights[agent] += [ep_returns[agent]] * ep_lengths[agent]

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
        batch_weights,
        info
    )







