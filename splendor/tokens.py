from enum import Enum

class Token(Enum):
    GREEN = 0
    RED = 1
    BLACK = 2
    WHITE = 3
    BLUE = 4
    GOLD = 5
    
COMBINATIONS = [
    (Token.WHITE, Token.BLUE, Token.GREEN),
    (Token.WHITE, Token.BLUE, Token.RED),
    (Token.WHITE, Token.BLUE, Token.BLACK),
    (Token.WHITE, Token.GREEN, Token.RED),
    (Token.WHITE, Token.GREEN, Token.BLACK),
    (Token.WHITE, Token.RED, Token.BLACK),
    (Token.BLUE, Token.GREEN, Token.RED),
    (Token.BLUE, Token.GREEN, Token.BLACK),
    (Token.BLUE, Token.RED, Token.BLACK),
    (Token.GREEN, Token.RED, Token.BLACK),
]

    
def encode_tokens(tokens: dict[Token, int]):
    return [tokens[token] for token in Token if token in tokens.keys()]