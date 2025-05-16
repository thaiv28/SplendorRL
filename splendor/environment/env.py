import functools   
import os
import pickle
import random

import gymnasium as gym
from gymnasium.spaces import OneOf, Discrete, Dict, MultiDiscrete, Box, Tuple
from gymnasium.spaces.utils import flatten, flatten_space
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, BaseWrapper
import numpy as np
from splendor.environment.player import Player
from splendor.environment.cards import Development, Noble, Token
from splendor.environment.tokens import COMBINATIONS
from splendor.environment.game import Splendor

NUM_ITERS = 200

def prewrapped_env():
    return FlattenObservationWrapper(FlattenActionWrapper(SplendorEnv()))

class SplendorEnv(AECEnv):
   
    metadata = {
        "name": "splendor_v0"
    } 
    
    def __init__(self, num_players=2):
        self.possible_agents = [f"player_{i}" for i in range(4)]
        self.generator = None
        self.game = Splendor()
        self.num_players=num_players
        
    def reset(self, seed=None, options={}):
        self.game.reset()
       
        self.num_steps = 0 
        self.agents = self.possible_agents[:self.num_players]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.infos = {agent: {"action_mask": self.game.get_action_mask(agent)} for agent in self.agents}
       
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()  
        
    # TODO : add functionality for reserving random card
    # TODO: add ability to return 1 token when reserving and gaining gold
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return OneOf(
            (
                Discrete(10),       # Choose 3 tokens
                Discrete(5),        # Choose 2 same tokens
                Discrete(12),       # Reserve cards
                Discrete(12),       # Purchase card on board
                Discrete(3),        # Purchase reserved card
                Discrete(10),       # Choose 2 different tokens (capped)
                Discrete(5),        # Choose 1 token (capped)
                Discrete(10),       # Return 3 tokens 
            )
        )
         
    @functools.lru_cache(maxsize=None) 
    def observation_space(self, agent): 
        return Dict(
            {
                "available_tokens": MultiDiscrete([6] * 6),
                "available_nobles": Box(low=0, high=4, shape=(3, 6), dtype=np.int_),
                    "available_cards": Box(low=0, high=10, shape=(12, 12), dtype=np.int_),
                "player":
                    Dict({
                            "prestige": Box(low=0, high=40, dtype=np.int_),
                            "tokens": MultiDiscrete([6] * 6),
                            "developments": MultiDiscrete([20] * 5),
                            "reserved_cards": Box(low=0, high=10, shape=(3, 12)),
                    }),
                "opponents":
                    Tuple((Dict({
                        "prestige": Box(low=0, high=40, dtype=np.int_),
                        "tokens": MultiDiscrete([6] * 6),
                        "developments": MultiDiscrete([20] * 5),
                        "reserved_cards": Box(low=0, high=10, shape=(3, 12)),
                    }) for _ in range(self.num_players - 1)))
            }
        )
        
        
    def observe(self, agent):
        obs = {}
        obs["available_tokens"] = np.array([q for _, q in self.game.tokens.items()], dtype=np.int_)
        
        obs["available_nobles"] = np.array([noble.obs_repr() for noble in self.game.nobles])
        if len(self.game.nobles) < 3:
            for _ in range(len(self.game.nobles, 3)):
                obs["available_nobles"] = np.concatenate([obs["available_nobles"], np.zeros(6)])
                
        obs["available_cards"] = np.array([development.obs_repr() for stack in self.game.developments.values() for development in stack])
        if len(obs["available_cards"]) < 12:
            for _ in range(len(obs["available_cards"]), 12):
                obs["available_cards"] = np.append(obs["available_cards"], np.zeros(12))
                
        EXPECTED_SIZES = {
            "available_tokens": 6,
            "available_nobles": 3 * 6,
            "available_cards":  12 * 12
        }
        
        for key, size in EXPECTED_SIZES.items():
            if obs[key].size != size:
                raise ValueError(f"Size of {key} should be {size}, but is {obs[key].size}")
                
        obs["player"] = self.game.players[agent].obs_repr()
        obs["opponents"] = np.array([self.game.players[a].obs_repr() for a in self.agents if a != agent])
        if obs["opponents"].size != self.num_players - 1:
            for _ in range(obs["opponents"].size, self.num_players - 1):
                obs["opponents"] = np.append(obs["opponents"], np.array(Player.empty_obs_repr()))
       
        return obs
    
    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return
        
        agent = self.agent_selection
        self._cumulative_rewards[agent] = 0
        # handle game logic depending on the action. assume all actions are valid (invalid would be masked)
        main_action = action[0]
        subaction = action[1]
        match main_action:
            case 0:
                reward = self.game.take_three_tokens(agent, subaction)
            case 1:
                reward = self.game.take_two_same_tokens(agent, subaction)
            case 2:
                reward = self.game.reserve_card(agent, subaction)
            case 3:
                reward = self.game.purchase_development(agent, subaction)
            case 4:
                reward = self.game.purchase_reserved(agent, subaction)
            case 5:
                reward = self.game.take_two_different_tokens(agent, subaction)
            case 6:
                reward = self.game.take_one_token(agent, subaction)
            case 7:
                # TODO: return 3 tokens should be combined with take_two_dffierent or take_one token. all moves should be self contained to one action. k
                reward = self.game.return_three_tokens(agent, subaction)
            case _:
                raise ValueError("Main action %s not supported", main_action)
      
        # print(f"{agent} should recieve reward of {reward}") 
        self.truncations = {
            agent: self.num_steps >= NUM_ITERS for agent in self.agents
        }
           
        self.infos = {agent: {
            "action_mask": self.game.get_action_mask(agent)
            } for agent in self.agents
        }
       
        self.rewards[agent] = reward

        # Update cumulative rewards dictionary. Update BEFORE adding win reward, because win
        # reward is accounted for in algorithm loop. This is because winner is only determined
        # once everyone has had the same number of turns.

        # Ex: player 0 gets 15 pp. player 0 does not "win" until player 1 has gone, because
        # player 1 could get 16 pp in their turn. We cannot assume player 0 won just because
        # they reached 15 pp.
        self._accumulate_rewards()
        self._clear_rewards()      

        if self._agent_selector.is_last():
            # print(f"{agent=}")
            winner = self.game.get_winner()
            if winner:
                self.terminations = {agent: True for agent in self.agents}
                self.rewards[winner] += 100
                
            self.num_steps += 1


        # print(f"__________________") 
        # print(f"Current Agent: {agent}")
        # print(f"Current rewards: {self.rewards}")
        # print(f"Pre accumulation cumulative rewards: {self._cumulative_rewards}")
        # print(f"Post accumulation cumulative rewards: {self._cumulative_rewards}")

        # if returned three tokens, that player gets to go again 
       
    def next_agent(self):
        self.agent_selection = self._agent_selector.next()
                       
    def render(self):
        print("____________________________________")
        print(f"Move {self.num_steps} of {NUM_ITERS}")
        print(f"AGENT'S TURN: {self.agent_selection}")
        print(self.game.game_state())
        
    def sample(self, mask):
        valid_actions = np.flatnonzero(mask[0])
        if len(valid_actions) == 0:
            raise ValueError("No valid actions in Discrete space mask.")
        return int(np.random.choice(valid_actions))
        
# Converts OneOf space into Discrete space
class FlattenActionWrapper(BaseWrapper):
    def __init__(self, env: SplendorEnv):
        super().__init__(env)
        self.original_space = env.action_space(self.possible_agents[0])
        self.mapping = []
        offset = 0

        for i, space in enumerate(self.original_space.spaces):
            assert(isinstance(space, Discrete))
            for j in range(space.n):
                self.mapping.append((i, j))

    def discrete_to_composite(self, action):
        # Convert flattened discrete action to (which_space, action_within_space)
        space_index, sub_action = self.mapping[action]
        return (space_index, sub_action)
    
    def step(self, action):
        if action is None:
            super().step(None)
        else:
            super().step(self.discrete_to_composite(action))
        
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(len(self.mapping))
    

class FlattenObservationWrapper(BaseWrapper):
    def __init__(self, env):
        super().__init__(env)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return flatten_space(self.env.observation_space(agent))

    def observe(self, agent):
        original_obs = self.env.observe(agent)
        return flatten(self.env.observation_space(agent), original_obs)
    
