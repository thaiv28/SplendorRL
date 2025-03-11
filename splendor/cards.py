import numpy as np

from splendor.tokens import Token, encode_tokens

class Development:
    def __init__(self, level: int, bonus: Token, prestige: int, cost: dict[Token, int]):
        self.level = level
        self.bonus = bonus
        self.prestige = prestige
        self.cost = cost
        
    def obs_repr(self):
        return np.array([self.level, self.bonus.value, self.prestige] + encode_tokens(self.cost),
                        dtype=np.int_)
        
class Noble:
    def __init__(self, prestige: int, cost: dict[Token, int]):
        self.prestige = prestige
        self.cost = cost
        
    def obs_repr(self):
        return np.array([self.prestige] + encode_tokens(self.cost), dtype=np.int_)
    