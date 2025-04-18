import argparse

from splendor.game import Splendor

def main():
    parser = argparse.ArgumentParser("splendor")
    parser.add_argument('-p', "--players", default=2, help="Number of players", type=int)
    
    args = parser.parse_args()
    
    game = Splendor()
    game.reset(seed=None, num_players=args.players)
    i = 0
  
    while game.ongoing:
        print(game.game_state())
        player = f"player_{i}"
        action_message = f"""
        {player.upper()} CHOOSE AN ACTION:
        (0-11): Purchase development
        (12-23): Reserve development
        (24-34): Take 3 token combination
        (35-39): Take 2 token combination
        (40-42): Purchase reserved card
        """.replace("        ", "")
       
        while(True): 
            try:
                action = int(input(action_message))
                if action >= 0 and action <= 43:
                    break
            except NameError:
                continue
            
        if action < 12:
            game.purchase_development(player, action)
        elif action < 24:
            game.reserve_card(player, action - 12)
        elif action < 35:
            game.take_three_tokens(player, action - 24)
        elif action < 40:
            game.take_two_tokens(player, action - 35)
        elif action < 43:
            game.purchase_reserved(player, action - 40)
        else:
            assert(False)
        
            
        i = (i + 1) % args.players
        
    print(f"Game winner is {game.get_winner()}")
            
    

if __name__=="__main__":
    main()
