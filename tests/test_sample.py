import pytest

from splendor.sample import sample_one_epoch
from splendor.environment.env import AECEnv, prewrapped_env

def test_sample_basic():
    env = prewrapped_env()
