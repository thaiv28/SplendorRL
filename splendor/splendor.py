import os
import pickle
import numpy as np
from splendor.player import Player
from splendor.cards import Development, Noble, Token
from splendor.tokens import COMBINATIONS

class Splendor:
    def __init__(self):
        pass

    def reset(self, seed=None, num_players=2):
        if self.generator is None or seed is not None:
            self.seed = seed
            self.generator = np.random.default_rng(seed=seed) 
       
        self.players = {f"player_{i}": Player() for i in range(num_players)}
        
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

    def take_three_tokens(self, agent, action: int):
        tokens = COMBINATIONS[action]
        for token in tokens:
            self.tokens[token] -= 1
            self.players[agent].tokens[token] += 1
    
    def take_two_tokens(self, agent, action: int):
        self.tokens[action] -= 2
        self.players[agent].tokens[action] += 2
        
    def reserve_card(self, agent, action: int):
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
        self.purchase_development_helper(self.players[agent], card)
        self.developments[level][index] = self.development_stack[level].pop()
     
    def purchase_reserved(self, agent, action):
        card = self.players[agent].reserved_cards.pop(action)
        self.purchase_development_helper(self.players[agent], card)
    
    def purchase_development_helper(self, agent, development: Development):
        for token, cost in development.cost:
            tokens_spent = cost - self.players[agent].developments[token]
            self.players[agent].tokens[token] -= tokens_spent
            self.tokens[token] += tokens_spent
            
        self.players[agent].prestige += development.prestige
        self.players[agent].developments[development.bonus] += 1