import pickle

import pandas as pd
import numpy as np

from splendor.environment.cards import Development, Noble
from splendor.environment.tokens import Token

# Load the spreadsheet
file_path = '../files/raw_nobles.csv'
df = pd.read_csv(file_path)

nobles = []

for row in df.to_dict(orient="records"):
    name = row["Name"].replace("\n", "")
    prestige = 3
    cost = {
        Token[name.upper()]: value for name, value in row.items()
            if name.lower() in ["green", "red", "white", "blue", "black"]
    }
    
    for k,v in cost.items():
        if np.isnan(v):
            cost[k] = 0
        else:
            cost[k] = int(cost[k])
   
    nobles.append(Noble(name=name, prestige=prestige, cost=cost)) 
    
with open('../files/nobles.pickle', 'wb') as file:
    pickle.dump(nobles, file)
    
breakpoint()