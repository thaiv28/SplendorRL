import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete

from collections import defaultdict

from splendor.env import SplendorEnv, FlattenActionWrapper, FlattenObservationWrapper

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
        batch_obs = {}
        batch_acts = {}
        batch_weights = {}
        batch_rets = {}
        batch_lens = {}
        
        env.reset()
        ep_rews = {}
        batch_complete = False
       
        # problem: agent 1 is chosen by env.last(). agent 1 takes action. agent 1 wins.
        # agent 0 is called by last(). agent 0 gets termination signal. game ends, 
        # but agent 1 never got reward.
        
        # assume agent 0 returned by env.last(). agent 0 takes action nad wins.
        # termination is not sent bc need everyone to finish. agent 1 is returned
        # by env.last(). agent 1 takes action. agent 1 discovers that agent 0 won. 
        # agent0 is returned by env.last(). agent0 gets reward and termination.
        # end game and all is well!
        
        for agent in env.agent_iter():
            #agent = env.agent_selection 
            if render:
                env.render()
               
            # print(f"New loop. Getting reward for {agent}") 
            obs, rew, term, trunc, info = env.last()
            # if rew != 0:
            #     print(f"{rew=} for {agent=}")
            
            
            if term or trunc:
                action = None
            else:
                action = get_action(torch.as_tensor(obs, dtype=torch.float32),
                                mask=torch.from_numpy(info["action_mask"]))
                
            env.step(action)
            #print(f"Finished action for {agent}")
           
            if action is not None:
                batch_obs[agent] = batch_obs.get(agent, []) + [(obs.copy())]
                batch_acts[agent] = batch_acts.get(agent, []) + [action]
                
            ep_rews[agent] = ep_rews.get(agent, []) + [rew]
            
            if term or trunc:
                # problematic cause rewards tensor is different size than action tensor.
                # we are addinga reward without a corresponding action
                for agent in env._cumulative_rewards:
                    ep_rews[agent].append(env._cumulative_rewards[agent])
                    
                for a in ep_rews:
                    ep_rews[a].pop(0)
                    
                ep_ret = {a: sum(ret) for a, ret in ep_rews.items()} 
                ep_len = len(ep_rews[agent])
                
                for a in ep_ret:
                    batch_rets[a] = batch_rets.get(a, []) + [ep_ret[a]]
                    batch_lens[a] = batch_lens.get(a, []) + [ep_len]
                    batch_weights[a] = batch_weights.get(a, []) + [ep_ret[a]] * ep_len
                
                assert(len(batch_acts[agent]) == len(batch_weights.get(agent, []))) 
                env.reset()
                ep_rews = {}

                if len(batch_obs[agent]) > batch_size:
                    break
        
        breakpoint()
        optimizer.zero_grad()
        batch_losses = torch.zeros(len(batch_obs))
        for i, agent in enumerate(batch_obs):
            # efficiency reasons
            batch_obs[agent] = np.array(batch_obs[agent])
            batch_losses[i] = compute_loss(obs=torch.as_tensor(batch_obs[agent], dtype=torch.float32),
                                  action=torch.as_tensor(batch_acts[agent], dtype=torch.int32),
                                  ret=torch.as_tensor(batch_weights[agent], dtype=torch.float32))
            breakpoint()
             
        batch_loss.backward()
        optimizer.step()
        return batch_loss, batch_rets, batch_lens
    
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
    
if __name__=="__main__":
    train(render=False)