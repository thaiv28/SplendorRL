from splendor.env import SplendorEnv, FlattenActionWrapper

def main():
    env = SplendorEnv()
    env = FlattenActionWrapper(env)
    env.reset(seed=42)

    env.render()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            if "action_mask" in info:
                mask = info["action_mask"] 
            else:
                assert(False)
                    
            action = env.action_space(agent).sample(mask=mask)
            print(action)

        breakpoint()
        env.step(action)
        env.render() 
    env.close()
    
 
if __name__=="__main__":
    main()