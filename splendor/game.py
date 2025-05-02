import os
import pickle
import numpy as np
from splendor.player import Player
from splendor.cards import Development, Noble, Token
from splendor.tokens import COMBINATIONS, DUO_COMBINATIONS

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
            
        return 0
    
    def take_two_same_tokens(self, agent, action: int):
        self.tokens[Token(action)] -= 2
        self.players[agent].tokens[Token(action)] += 2
        
        return 0
    
    def take_two_different_tokens(self, agent, action: int):
        tokens = DUO_COMBINATIONS[action]
        for token in tokens:
            self.tokens[token] -= 1
            self.players[agent].tokens[token] += 1
            
        return 0
    
    def take_one_token(self, agent, action: int):
        token = Token(action)
        self.tokens[token] -= 1
        self.players[agent].tokens[token] += 1
        
        return 0
    
    def return_three_tokens(self, agent, action: int):
        tokens = COMBINATIONS[action]
        for token in tokens:
            self.tokens[token] += 1
            self.players[agent].tokens[token] -= 1
            
        return 0
        
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
            if len(self.development_stack[level]) != 0:
                self.developments[level][index] = self.development_stack[level].pop()
            else:
                self.developments[level].pop(index)
        else:
            level = action - 11
            card = self.development_stack[level].pop()
        
        self.players[agent].reserved_cards.append(card)
        
        if self.tokens[Token.GOLD] > 0 and self.players[agent].total_tokens() < 10:
            self.tokens[Token.GOLD] -= 1
            self.players[agent].tokens[Token.GOLD] += 1
        
        return 0
      
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
        if len(self.development_stack[level]) != 0:
            self.developments[level][index] = self.development_stack[level].pop()
        else:
            self.developments[level].pop(index)
        
        self.ongoing = self.is_ongoing()
        
        return card.prestige
     
    def purchase_reserved(self, agent, action):
        card = self.players[agent].reserved_cards.pop(action)
        self._purchase_development_helper(agent, card)
        
        self.ongoing = self.is_ongoing()
        return card.prestige
    
    def _purchase_development_helper(self, agent, development: Development):
        player = self.players[agent]
        for token, cost in development.cost.items():
            if cost == 0:
                continue
            
            tokens_needed = cost - player.developments[token]
            if tokens_needed < 0:
                tokens_needed = 0
                
            if tokens_needed > player.tokens[token]:
                gold_used = tokens_needed - player.tokens[token]
                player.tokens[Token.GOLD] -= gold_used
                self.tokens[Token.GOLD] += gold_used
                
                tokens_needed -= gold_used
                
            player.tokens[token] -= tokens_needed
            self.tokens[token] += tokens_needed
            
        self.players[agent].prestige += development.prestige
        self.players[agent].developments[development.bonus] += 1
        
        for token, count in player.tokens.items():
            if count < 0:
                raise Exception("Player has token count less than zero")
   
    # TODO: invalidate buying empty cards
    def get_action_mask(self, agent):
        player = self.players[agent]
        
        choose_3 = np.ones(10, dtype=np.int8)
        choose_2 = np.ones(5, dtype=np.int8)
        reserve = np.ones(12, dtype=np.int8)
        purchase = np.ones(12, dtype=np.int8)
        purchase_reserved = np.ones(3, dtype=np.int8)
        choose_2_different = np.ones(10, dtype=np.int8)
        choose_1 = np.ones(5, dtype=np.int8)
        return_3 = np.zeros(10, dtype=np.int8)
        
        if player.total_tokens() > 7:
            return_3 = np.ones(10, dtype=np.int8)

        for token, count in self.players[agent].tokens.items():
            if token == Token.GOLD:
                continue
            if count <= 0:
                for i, combo in enumerate(COMBINATIONS):
                    if token in combo:
                        return_3[i] = 0
                        
        for token, count in self.tokens.items():
            if token == Token.GOLD:
                continue
            if count <= 0:
                choose_1[token.value] = 0
                for i, c in enumerate(COMBINATIONS):
                    if token in c:
                        choose_3[i] = 0
                for i, c in enumerate(DUO_COMBINATIONS):
                    if token in c:
                        choose_2_different[i] = 0
            if count < 4:
                choose_2[token.value] = 0
                
        if len(player.reserved_cards) >= 3:
            reserve = np.zeros(12, dtype=np.int8)
        
        for level in self.developments:
            for i, d in enumerate(self.developments[level]):
                if not player.is_purchasable(d):
                    purchase[4*(level - 1) + i] = 0
            
            # for when we run out of developments to purchase and board becomes empty
            for i in range(len(self.developments[level]), 4):
                purchase[4*(level - 1) + i] = 0
                reserve[4*(level - 1) + i] = 0
                    
        for i in range(3):
            if i >= len(player.reserved_cards):
                purchase_reserved[i] = 0
                continue
            if not player.is_purchasable(player.reserved_cards[i]):
                purchase_reserved[i] = 0
               
        if player.total_tokens() > 8:
            choose_2_different = np.zeros(10, dtype=np.int8)
            choose_2 = np.zeros(5, dtype=np.int8)
        if player.total_tokens() > 9:
            choose_1 = np.zeros(5, np.int8)
            
        if player.total_tokens() > 7:
            choose_3 = np.zeros(10, dtype=np.int8)
        
            
        mask = np.concat((choose_3, choose_2, reserve, purchase, purchase_reserved,
                          choose_2_different, choose_1, return_3))
        if not np.any(mask):
            raise ValueError("Mask allows for no actions.")
        return mask
                
        
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