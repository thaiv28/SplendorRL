import os
import pickle
import numpy as np
from splendor.player import Player
from splendor.cards import Development, Noble, Token
from splendor.tokens import COMBINATIONS

class Splendor:
    def __init__(self):
        self.generator = None
        self.ongoing = False

    def reset(self, seed=None, num_players=2):
        self.ongoing = True
        if self.generator is None or seed is not None:
            self.seed = seed
            self.generator = np.random.default_rng(seed=seed) 
       
        self.players = {f"player_{i}": Player() for i in range(num_players)}
        
        d_pickle = os.path.dirname(__file__) + "/../files/developments.pickle"
        with open(d_pickle, 'rb') as file:
            all_developments = pickle.load(file)
            self.generator.shuffle(all_developments)
           
        self.development_stack = {
            i: [c for c in all_developments if c.level == i]
            for i in range(1, 4)
        }
        self.developments = {
            i: [stack.pop(0) for _ in range(4)] 
            for i, stack in self.development_stack.items()
        }
            
        n_pickle = os.path.dirname(__file__) + "/../files/nobles.pickle"
        with open(n_pickle, 'rb') as file:
            noble_stack = pickle.load(file)
            self.generator.shuffle(noble_stack)
        self.nobles = [noble_stack.pop(0) for _ in range(3)]
        
        self.tokens = {token: 4 for token in Token}
        self.tokens[Token.GOLD] = 5

    def take_three_tokens(self, agent, action: int):
        tokens = COMBINATIONS[action]
        for token in tokens:
            self.tokens[token] -= 1
            self.players[agent].tokens[token] += 1
    
    def take_two_tokens(self, agent, action: int):
        self.tokens[Token(action)] -= 2
        self.players[agent].tokens[Token(action)] += 2
        
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
        
        if self.tokens[Token.GOLD] > 0:
            self.tokens[Token.GOLD] -= 1
            self.players[agent].tokens[Token.GOLD] += 1
      
    def purchase_development(self, agent, action):
        if action < 4:
            level = 1
        elif action < 8:
            level = 2
        else:
            level = 3
        index = action % 4
        
        card = self.developments[level][index % 4]
        self._purchase_development_helper(agent, card)
        self.developments[level][index] = self.development_stack[level].pop()
        
        self.ongoing = self.is_ongoing()
     
    def purchase_reserved(self, agent, action):
        card = self.players[agent].reserved_cards.pop(action)
        self._purchase_development_helper(agent, card)
        
        self.ongoing = self.is_ongoing()
    
    def _purchase_development_helper(self, agent, development: Development):
        for token, cost in development.cost.items():
            tokens_spent = cost - self.players[agent].developments[token]
            self.players[agent].tokens[token] -= tokens_spent
            self.tokens[token] += tokens_spent
            
        self.players[agent].prestige += development.prestige
        self.players[agent].developments[development.bonus] += 1
   
    def game_state(self) -> str:
        s = "\nNOBLES:\n" + "\n".join(str(noble) for noble in self.nobles)
        s += "\n--------------"
        s += "\nLEVEL 3: " + " | ".join(str(c) + f" ({8 + i})" for i, c in enumerate(self.developments[3]))
        s += "\nLEVEL 2: " + " | ".join(str(c) + f" ({4 + i})" for i, c in enumerate(self.developments[2]))
        s += "\nLEVEL 1: " + " | ".join(str(c) + f" ({i})" for i, c in enumerate(self.developments[1]))
        s += "\n--------------"
        s += "\nAVAILABLE TOKENS: " + str({t.name: v for t, v in self.tokens.items()})
        s += "\n--------------"
        s += "\n" + "\n".join([f"Player {i}: {str(p)}" for i, p in enumerate(self.players.values())])
        return s
    
    def is_ongoing(self) -> bool:
        for player in self.players.values():
            if player.prestige >= 15:
                return False
            
        return True
    
    def get_winner(self):
        for name, player in self.players.items():
            if player.prestige >= 15:
                return name
            
        return None