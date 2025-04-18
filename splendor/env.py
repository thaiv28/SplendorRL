import functools
import os
import pickle
import random

import gymnasium as gym
from gymnasium.spaces import OneOf, Discrete, Dict, MultiDiscrete, Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
import numpy as np

from splendor.player import Player
from splendor.cards import Development, Noble, Token
from splendor.tokens import COMBINATIONS
from splendor.game import Splendor

class Game(AECEnv):
   
    metadata = {
        "name": "splendor_v0"
    } 
    
    def __init__(self):
        self.possible_agents = [f"player_{i}" for i in range(4)]
        self.generator = None
        self.game = Splendor()
         
        
    def reset(self, seed=None, num_players=2):
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

        self.game.reset()
        
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
                    [{
                        "prestige": Box(low=0, high=40, dtype=np.int_),
                        "tokens": MultiDiscrete([5] * 6),
                        "developments": MultiDiscrete([5] * 5),
                        "reserved_cards": Box(low=0, high=10, shape=(3, 12)),
                    } * (self.num_players - 1)]
            }
        )
        
        
    def observe(self, agent):
        obs = {}
        obs["available_tokens"] = np.array([q for _, q in self.game.tokens.items()], dtype=np.int_)
        obs["available_nobles"] = np.array([noble.obs_repr() for noble in self.game.nobles])
        obs["available_cards"] = np.array([development.obs_repr() for stack in self.game.developments for development in stack])
        obs["player"] = self.game.players[agent].obs_repr()
        obs["opponents"] = [self.game.players[a].obs_repr() for a in self.agents if a != agent]
        
        return obs
    
    def step(self, action):
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
        # handle game logic depending on the action. assume all actions are valid (invalid would be masked)
        main_action = action[0]
        subaction = action[1]
        match main_action:
            case 0:
                self.game.take_three_tokens(agent, subaction)
            case 1:
                self.game.take_two_tokens(agent, subaction)
            case 2:
                self.game.reserve_card(agent, subaction)
            case 3:
                self.game.purchase_development(agent, subaction)
            case 4:
                self.game.purchase_reserved(agent, subaction)
            case _:
                raise ValueError("Main action %s not supported", main_action)







