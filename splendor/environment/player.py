from collections import defaultdict

import numpy as np

from splendor.environment.cards import Development
from splendor.environment.tokens import Token, encode_tokens

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
        count = sum(self.tokens.values())
        if count > 10:
            raise Exception("Player has total tokens > 10")
        
        return count
   
    def is_purchasable(self, development: Development):
        gold_required = 0
        for t, cost in development.cost.items():
            tokens_needed = cost - self.developments[t]
            if tokens_needed < 0:
                tokens_needed = 0
                
            if tokens_needed > self.tokens[t]:
                gold_required += tokens_needed - self.tokens[t]
                
            if gold_required > self.tokens[Token.GOLD]:
                return False
            
        return True

    @staticmethod
    def empty_obs_repr():
        return {
            "prestige": 0,
            "tokens": np.zeros(6, dtype=np.int_),
            "developments": np.zeros(5, dtype=np.int_),
            "reserved_cards": np.zeros(36, dtype=np.int_),
        }

    def obs_repr(self):
        d = {
            "prestige": np.int_(self.prestige),
            "tokens": np.array(encode_tokens(self.tokens)),
            "developments": np.array(encode_tokens(self.developments)),
            "reserved_cards": np.array([d.obs_repr() for d in self.reserved_cards])
        }
        
        if len(d["reserved_cards"]) < 3:
            for _ in range(len(d["reserved_cards"]), 3):
                d["reserved_cards"] = np.append(d["reserved_cards"], np.zeros(12))
        
        expected_sizes = {
            "prestige": 1,
            "tokens": 6,
            "developments": 5,
            "reserved_cards": 3 * 12
        }
        
        for key, size in expected_sizes.items():
            if d[key].size != size:
                raise ValueError(f"Size of {key} should be {size}, but is {d[key].size}")
           
        return d
        
        
        
    def __str__(self):
        s = f"TOKENS ({list(self.tokens.values())}), \
            DEVELOPMENTS ({list(self.developments.values())}), \
            PRESTIGE ({self.prestige})"
            
        s += f"\nRESERVED: " + " | ".join(str(d) for d in self.reserved_cards)
        return s
