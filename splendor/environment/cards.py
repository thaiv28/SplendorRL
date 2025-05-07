import numpy as np

from splendor.environment.tokens import Token, encode_tokens

class Development:
    def __init__(self, level: int, bonus: Token, prestige: int, cost: dict[Token, int]):
        self.level = level
        self.bonus = bonus
        self.prestige = prestige
        self.cost = cost
        
    def obs_repr(self):
        bonus_encoding = np.zeros(5)
        bonus_encoding[self.bonus.value] = 1
        return np.concatenate([np.array([self.level, self.prestige] + encode_tokens(self.cost),
                        dtype=np.int_), bonus_encoding])
        
    def __str__(self):
        return f"{self.bonus.name} ({self.prestige}) -- {list(self.cost.values())}"
        
class Noble:
    def __init__(self, name: str, prestige: int, cost: dict[Token, int]):
        self.name = name.replace("\n", "")
        self.prestige = prestige
        self.cost = cost
        
    def obs_repr(self):
        return np.array([self.prestige] + encode_tokens(self.cost), dtype=np.int_)
    
    def __str__(self):
        return f"{self.name} ({self.prestige}) -- {list(self.cost.values())}"
    