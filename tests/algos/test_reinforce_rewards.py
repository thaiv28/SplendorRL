import numpy as np

from splendor.algos import Reinforce

def test_single_agent():
    rewards = {
        "player_0": [np.array([1, 5, 1, 0]), np.array([4, 1, 0, 3])]
    }

    rewards = Reinforce.batchify_rewards(rewards, 0.9)

    expected = {
        "player_0": np.array([3.0075, 2.5975, -2.3025, -3.3025, 3.03275, -0.62425, -1.35425, -1.05425])
    }

    for player in rewards:
        assert np.allclose(rewards[player], expected[player])


def test_multi_agent():
    rewards = {
        "player_0": [np.array([1, 5, 1, 0]), np.array([4, 1, 0, 3])],
        "player_1": [np.array([2, 2, 2, 2]), np.array([0, 0, 10, 0])]
    }

    rewards = Reinforce.batchify_rewards(rewards, 0.9)
    expected = {
        "player_0": np.array([3.0075, 2.5975, -2.3025, -3.3025,
                              3.03275, -0.62425, -1.35425, -1.05425]),
        "player_1": np.array([2.3535, 0.8955, -0.7245, -2.5245,
                              1.325, 2.225, 3.225, -6.775])
    }

    for player in rewards:
        assert np.allclose(rewards[player], expected[player], rtol=1e-4)



