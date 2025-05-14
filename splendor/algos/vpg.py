import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete

from collections import defaultdict

from splendor.environment.env import SplendorEnv, FlattenActionWrapper, FlattenObservationWrapper

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        if j < len(sizes) - 2:
            act = activation
        else:
            act = output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1], act())]
    return nn.Sequential(*layers)

# TODO : find out why +100 not added to winner, and how to deal with multiple agents same policy
# look into pettingzoo to figure out the reward accumulation
def train(lr=1e-2, epochs=50, batch_size=5000, hidden_sizes=[32], render=True):
    env = SplendorEnv()
    env = FlattenActionWrapper(env)
    env = FlattenObservationWrapper(env)
   
    # Dimension of observation space: 532
    obs_dim = env.observation_space("").shape[0]
    # Dimension of action space: 67
    num_actions = env.action_space("").n
    
    policy_network = mlp(sizes=[obs_dim] + hidden_sizes + [num_actions])
    
    def get_policy(obs, mask=None):
        logits = policy_network(obs)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e10)
        return Categorical(logits=logits)
    
    def get_action(obs, mask=None):
        return get_policy(obs, mask=mask).sample().item()
    
    # TODO: fix error if action=None
    def compute_loss(obs, action, ret):
        logp = get_policy(obs).log_prob(action)
        return -(logp * ret).mean()
    
    optimizer = Adam(policy_network.parameters(), lr=lr)
    
    def train_one_epoch():
        batch_obs = defaultdict(lambda: [])
        batch_acts = defaultdict(lambda: [])
        batch_weights = defaultdict(lambda: [])
        batch_rets = defaultdict(lambda: [])
        batch_lens = defaultdict(lambda: [])

        
        env.reset()
        ep_rews = defaultdict(lambda: [])
        batch_complete = False
       
        # problem: agent 1 is chosen by env.last(). agent 1 takes action. agent 1 wins.
        # agent 0 is called by last(). agent 0 gets termination signal. game ends, 
        # but agent 1 never got reward.
        
        # assume agent 0 returned by env.last(). agent 0 takes action nad wins.
        # termination is not sent bc need everyone to finish. agent 1 is returned
        # by env.last(). agent 1 takes action. agent 1 discovers that agent 0 won. 
        # agent0 is returned by env.last(). agent0 gets reward and termination.
        # end game and all is well!
        
        # agent0 is returned by env.last(). agent0 gets reward and termination.
        # end game and all is well!
        for agent in env.agent_iter():
            #agent = env.agent_selection 
            if render:
                env.render()

            while True:
                # get observation
                # get action from model
                # get reward
                # if reward is -1, loop again. else, break from loop.
                obs, _, term, trunc, info = env.last()

                if term or trunc:
                    break

                action = get_action(torch.as_tensor(obs, dtype=torch.float32),
                                    mask=torch.from_numpy(info["action_mask"]))
                env.step(action)

                _, rew, _, _, _ = env.last()

                batch_obs[agent].append(obs.copy())
                batch_acts[agent].append(action)
                ep_rews[agent].append(rew)

                if rew < 0:
                    continue
                else:
                    env.next_agent()
                    break


            if term or trunc:
                for agent in env._cumulative_rewards:
                    ep_rews[agent][-1] += env._cumulative_rewards[agent]

                ep_returns = {a: sum(rews) for a, rews in ep_rews.items()}
                ep_length = {a: len(rews) for a, rews in ep_rews.items()}

                for agent in ep_returns:
                    batch_rets[agent].append(ep_returns[agent])
                    batch_lens[agent].append(ep_length[agent])
                    batch_weights[agent] += [ep_returns[agent]] * ep_length[agent]

                env.reset()
                ep_rews = defaultdict(lambda: [])
                
                if len(batch_obs[agent]) > batch_size:
                    break

        optimizer.zero_grad()
        batch_losses = {}
        batch_losses_tensor = torch.zeros(len(batch_obs))
        for i, agent in enumerate(batch_obs):
            # efficiency reasons
            np_batch_obs = np.array(batch_obs[agent])
            batch_loss = compute_loss(obs=torch.as_tensor(np_batch_obs, dtype=torch.float32),
                                  action=torch.as_tensor(batch_acts[agent], dtype=torch.int32),
                                  ret=torch.as_tensor(batch_weights[agent], dtype=torch.float32))
            batch_losses[agent] = batch_loss
            batch_losses_tensor[i] = batch_loss

        total_loss = torch.sum(batch_losses_tensor)
        total_loss.backward()
        optimizer.step()
        return batch_losses, batch_rets, batch_lens
    
    for i in range(epochs):
        batch_losses, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d: '%i)
        for agent in batch_losses:
            print('agent %s \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                  (agent, batch_losses[agent], np.mean(batch_rets[agent]), np.mean(batch_lens[agent])))
    
if __name__=="__main__":
    train(render=False)
