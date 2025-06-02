import numpy as np
import pytest

from splendor.algos import Reinforce

compute_baseline = Reinforce.compute_baseline

def test_single_agent_multiple_trajectories():
    rewards = {
        "agent_0": [np.array([1, 2, 3]), np.array([4, 5])]
    }
    expected = {
        "agent_0": [np.array([-1, 0, 1]), np.array([-0.5, 0.5])]
    }
    result = compute_baseline(rewards)
    assert all(np.allclose(r, e) for r, e in zip(result["agent_0"], expected["agent_0"]))


def test_single_agent_single_trajectory():
    rewards = {
        "agent_0": [np.array([10, 20, 30])]
    }
    expected = {
        "agent_0": [np.array([-10, 0, 10])]
    }
    result = compute_baseline(rewards)
    assert all(np.allclose(r, e) for r, e in zip(result["agent_0"], expected["agent_0"]))


def test_multiple_agents():
    rewards = {
        "agent_0": [np.array([1, 3])],
        "agent_1": [np.array([4, 4, 4])]
    }
    expected = {
        "agent_0": [np.array([-1, 1])],
        "agent_1": [np.array([0, 0, 0])]
    }
    result = compute_baseline(rewards)
    for agent in expected:
        assert all(np.allclose(r, e) for r, e in zip(result[agent], expected[agent]))


def test_empty_agent_list():
    rewards = {
        "agent_0": []
    }
    expected = {
        "agent_0": []
    }
    result = compute_baseline(rewards)
    assert result == expected


def test_empty_input_dict():
    rewards = {}
    expected = {}
    result = compute_baseline(rewards)
    assert result == expected

