from collections import defaultdict

import numpy as np

from splendor.cards import Development
from splendor.tokens import Token, encode_tokens

class Player:
    def __init__(self):
        self.tokens: dict[Token, int] = {
            t: 0 for t in Token
        }
        self.developments: dict[Token, int] = {
            name: 0 for name in Token if name != Token.GOLD
        }
        self.prestige = 0
        self.reserved_cards: list[Development] = []
        
    def total_tokens(self):
        return sum(self.tokens.values())
   
    # TODO : take into account gold when determining if card is purchasable 
    def is_purchasable(self, development: Development):
        for t, cost in development.cost.items():
            if cost > self.tokens[t]:
                return False
            
        return True
        
    def obs_repr(self):
        return {
            "prestige": np.int_(self.prestige),
            "tokens": np.array(encode_tokens(self.tokens)),
            "developments": np.array(encode_tokens(self.developments)),
            "reserved_cards": np.array([d.obs_repr() for d in self.reserved_cards])
        }
        
    def __str__(self):
        s = f"TOKENS ({list(self.tokens.values())}), \
            DEVELOPMENTS ({list(self.developments.values())}), \
            PRESTIGE ({self.prestige})"
            
        s += f"\nRESERVED: " + " | ".join(str(d) for d in self.reserved_cards)
        return s