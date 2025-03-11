import numpy as np

from splendor.cards import Development
from splendor.tokens import Token, encode_tokens

class Player:
    def __init__(self):
        self.tokens: dict[Token, int] = {}
        self.developments: dict[Token, int] = []
        self.prestige = 0
        self.reserved_cards: list[Development] = []
        
    def obs_repr(self):
        return {
            "prestige": np.int_(self.prestige),
            "tokens": np.array(encode_tokens(self.tokens)),
            "developments": np.array(encode_tokens(self.developments)),
            "reserved_cards": np.array([d.obs_repr() for d in self.reserved_cards])
        }