import functools

import gymnasium as gym
from gymnasium.spaces import OneOf, Discrete, Dict, MultiDiscrete, Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import numpy as np

from splendor.player import Player
from splendor.cards import Development, Noble, Token

class Game(AECEnv):
   
    metadata = {
        "name": "splendor_v0"
    } 
    
    def __init__(self):
        self.possible_agents = [f"player_{i}" for i in range(4)]
         
        
    def reset(self, num_players=2):
        self.num_players = num_players
        
        self.agents = self.possible_agents[:num_players]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
       
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()  

        self.players = {agent: Player() for agent in self.agents}
        
        self.nobles = []
        self.developments = []
        
        self.tokens = {token: 4 for token in Token}
        self.tokens[Token.GOLD] = 5
        
        return self._get_obs(), self._get_info()
        
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return OneOf(
            (
                Discrete(10),       # Choose 3 tokens
                Discrete(5),        # Choose 2 tokens
                Discrete(15),       # Reserve cards
                Discrete(12),       # Purchase card on board
                Discrete(3)         # Purchase reserved card
            )
        )
         
    @functools.lru_cache(maxsize=None) 
    def observation_space(self, agent): 
        return Dict(
            {
                "available_tokens": MultiDiscrete([6] * 6),
                "available_nobles": Box(low=0, high=10, shape=(3, 6), dtype=np.int_),
                "available_cards": Box(low=0, high=10, shape=(12, 12), dtype=np.int_),
                "player":
                    {
                            "prestige": Box(low=0, high=40, dtype=np.int_),
                            "tokens": MultiDiscrete([5] * 6),
                            "developments": MultiDiscrete([5] * 5),
                            "reserved_cards": Box(low=0, high=10, shape=(3, 12)),
                    },
                "opponents":
                    {
                        [{
                            "prestige": Box(low=0, high=40, dtype=np.int_),
                            "tokens": MultiDiscrete([5] * 6),
                            "developments": MultiDiscrete([5] * 5),
                            "reserved_cards": Box(low=0, high=10, shape=(3, 12)),
                        } * (self.num_players - 1)]
                    }
            }
        )
        
        
    def observe(self, agent):
        obs = {}
        obs["available_tokens"] = np.array([q for _, q in self.tokens.items()], dtype=np.int_)
        obs["available_nobles"] = np.array([noble.obs_repr() for noble in self.nobles])
        obs["available_cards"] = np.array([development.obs_repr() for development in self.developments])
        obs["player"] = self.players[agent].obs_repr()
        obs["opponents"] = [self.players[a].obs_repr() for a in self.agents if a != agent]
        
        return obs
    
    def step(self, action):
        pass
        