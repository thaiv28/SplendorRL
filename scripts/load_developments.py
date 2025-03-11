import pickle

import pandas as pd

from splendor.cards import Development
from splendor.tokens import Token

# Load the spreadsheet
file_path = '../files/raw_developments.csv'
df = pd.read_csv(file_path)

developments = []

for row in df.to_dict(orient="records"):
    level = int(row["level"])
    bonus = Token[row["bonus"].upper()]
    prestige = int(row["prestige"])
    cost = {
        Token[name.upper()]: value for name, value in row.items()
            if name in ["green", "red", "white", "blue", "black"]
    } 
    
    developments.append(Development(level, bonus, prestige, cost))
    
with open('../files/developments.pickle', 'wb') as file:
    pickle.dump(developments, file)
    
breakpoint()