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

class Game(AECEnv):
   
    metadata = {
        "name": "splendor_v0"
    } 
    
    def __init__(self):
        self.possible_agents = [f"player_{i}" for i in range(4)]
        self.generator = None
         
        
    def reset(self, seed=None, num_players=2):
        if self.generator is None or seed is not None:
            self.seed = seed
            self.generator = np.random.default_rng(seed=seed) 
            
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
      
        d_pickle = os.path.abspath(__file__) / "../files/developments.pickle"
        with open(d_pickle, 'rb') as file:
            all_developments: list[Development] = self.generator.shuffle(pickle.load(file))
            
        self.development_stack = {
            i: [c for c in all_developments if c.level == i]
            for i in range(1, 4)
        }
        self.developments = {
            i: [stack.pop() for _ in range(4)] 
            for i, stack in self.development_stack.items()
        }
        self.developments = [self.development_stack.pop() for _ in range(12)]
            
        n_pickle = os.path.abspath(__file__) / "../files/nobles.pickle"
        with open(n_pickle, 'rb') as file:
            self.noble_stack = self.generator.shuffle(pickle.load(file))
        self.nobles = [self.noble_stack.pop() for _ in range(3)]
        
        self.tokens = {token: 4 for token in Token}
        self.tokens[Token.GOLD] = 5
        
        
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
        obs["available_tokens"] = np.array([q for _, q in self.tokens.items()], dtype=np.int_)
        obs["available_nobles"] = np.array([noble.obs_repr() for noble in self.nobles])
        obs["available_cards"] = np.array([development.obs_repr() for stack in self.developments for development in stack])
        obs["player"] = self.players[agent].obs_repr()
        obs["opponents"] = [self.players[a].obs_repr() for a in self.agents if a != agent]
        
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
                self.take_three_tokens(agent, subaction)
            case 1:
                self.take_two_tokens(agent, subaction)
            case 2:
                self.reserve_card(agent, subaction)
            case 3:
                self.purchase_development(agent, subaction)
            case 4:
                self.reserve_card(agent, subaction)
            case _:
                raise ValueError("Main action %s not supported", main_action)
        
    def take_three_tokens(self, agent, action):
        tokens = COMBINATIONS[action]
        for token in tokens:
            self.tokens[token] -= 1
            self.players[agent].tokens[token] += 1
    
    def take_two_tokens(self, agent, action):
        self.tokens[action] -= 2
        self.players[agent].tokens[action] += 2
        
    def reserve_card(self, agent, action):
        if action < 12:
            if action < 4:
                level = 1
            elif action < 8:
                level = 2
            else:
                level = 3
                
            index = action % 4
            card = self.developments[level][index]
            self.developments[level][index] = self.development_stack[level].pop()
        else:
            level = action - 11
            card = self.development_stack[level].pop()
        
        self.players[agent].reserved_cards.append(card)
      
    def purchase_development(self, agent, action):
        if action < 4:
            level = 1
        if action < 8:
            level = 2
        else:
            level = 3
        index = action % 4
        
        card = self.developments[level][index]
        self.purchase_development_helper(self, agent, card)
        self.developments[level][index] = self.development_stack[level].pop()
     
    def purchase_reserved(self, agent, action):
        card = self.players[agent].reserved_cards.pop(action)
        self.purchase_development_helper(agent, card)
    
    def purchase_development_helper(self, agent, development: Development):
        p = self.players[agent]
        for token, cost in development.cost:
            tokens_spent = cost - p.developments[token]
            p.tokens[token] -= tokens_spent
            self.tokens[token] += tokens_spent
            
        p.prestige += development.prestige
        p.developments[development.bonus] += 1
            
         
        
        
    
        

        