from pettingzoo import AECEnv
import numpy as np

import wandb

from splendor.environment.env import (
    SplendorEnv,
    FlattenObservationWrapper,
    FlattenActionWrapper
)
from splendor.policy import MLP
from splendor.sample import sample_one_epoch, BatchInfo
from splendor.algos import Reinforce

def train(
        env: AECEnv,
        algo: str,
        policy: str,
        lr: float = 1e-2,
        epochs: int = 50,
        batch_size: int = 5000,
        hidden_sizes: list[int] = [32],
        render: bool = True,
        algo_args: list = [],
        algo_kwargs: dict = {},
):
    """
    Trains an agent given an environment, algorithm, and policy architecture.
    """

    env = SplendorEnv()
    wrappers = [FlattenObservationWrapper, FlattenActionWrapper]
    for wrapper in wrappers:
        env = wrapper(env)

    obs_dim = env.observation_space(None).shape[0]
    action_dim = env.action_space(None).n

    # initialize policy
    match policy.lower():
        case "mlp":
            policy_network = MLP(sizes=[obs_dim] + hidden_sizes + [action_dim])
        case _:
            raise NotImplementedError("Policy not supported", policy)

    match algo.lower():
        case "reinforce":
            algorithm = Reinforce(policy_network, optimizer="adam", lr=lr)
        case _:
            raise NotImplementedError("Algorithm not supported", algo)

    # sample observations given policy 
    for i in range(epochs):
        batch_info = sample_one_epoch(env, policy_network, batch_size, render=render)
        batch_info.info["losses"] = algorithm.update(batch_info)
        

        print('epoch: %3d: '%i)
        for agent in batch_info.info["losses"]:
            batch_info.info["returns"][agent] = np.mean(batch_info.info["returns"][agent])
            batch_info.info["lengths"][agent] = np.mean(batch_info.info["lengths"][agent])


            print('agent %s \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                  (agent, batch_info.info["losses"][agent], 
                   batch_info.info["returns"][agent], 
                   batch_info.info["lengths"][agent]))

        if wandb.run:
            wandb.log(batch_info.info)








