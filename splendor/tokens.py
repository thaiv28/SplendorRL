from enum import Enum

class Token(Enum):
    GREEN = 0
    RED = 1
    BLACK = 2
    WHITE = 3
    BLUE = 4
    GOLD = 5
    
def encode_tokens(tokens: dict[Token, int]):
    return [tokens[token] for token in Token]