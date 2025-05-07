from pettingzoo.test import api_test
from splendor.environment.env import *

def test_env_api():
    env = FlattenActionWrapper(FlattenObservationWrapper(SplendorEnv()))
    api_test(env, num_cycles= 1_000_000, verbose_progress=True)

if __name__ == "__main__":
    test_env_api()
