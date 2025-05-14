import argparse

from splendor.train import train
from splendor.environment.env import SplendorEnv, FlattenActionWrapper, FlattenObservationWrapper

def main():
    parser = argparse.ArgumentParser("splendor")

    parser.add_argument('-e', "--environment", default="splendor_v0", 
                        help="Environment to run", type=str)
    parser.add_argument('-a', "--algorithm", default="reinforce", 
                        help="Algorithm to use", type=str,
                        choices=["reinforce"])
    parser.add_argument('-n', "--network", default="mlp", 
                        help="Policy network architecture", type=str,
                        choices=["mlp"])
    parser.add_argument('-l', "--learning-rate", default=1e-2,
                        help="Learning rate for optimizer", type=float)
    parser.add_argument('-p', "--epochs", default=50,
                        help="Number of training epochs", type=int)
    parser.add_argument('-b', "--batch-size", default=5000,
                        help="Number of timesteps per policy update", type=int)
    parser.add_argument('-r', "--render", action='store_true',
                        help="If set, renders the environment during training")

    args = parser.parse_args()

    #TODO: Add support for config
    
    env = SplendorEnv()
    env = FlattenObservationWrapper(FlattenActionWrapper(env))
    env.reset(seed=42)

    train(
        env=env,
        algo=args.algorithm,
        policy=args.network,
        lr=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        render=args.render
    )
 
if __name__=="__main__":
    main()
