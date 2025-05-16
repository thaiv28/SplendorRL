import pytest
import numpy as np

from splendor.sample import BatchInfo

def test_batch_info_basic_properties():
    b = BatchInfo({}, {}, {}, {})
    assert hasattr(b, "obs")
    assert hasattr(b, "masks")
    assert hasattr(b, "actions")
    assert hasattr(b, "weights")
    assert hasattr(b, "info")

    with pytest.raises(ValueError):
        _ = BatchInfo([], [], [], [])

def test_batch_info_valid_types():
    
    obs = {"agent_0": [[4, 3, 1],
                       [3, 2, 7],
                       [4, 5, 1]],
            "agent_1": [[5, 1, 9],
                        [5, 1, 6],
                        [1, 4, 1],
                        [5, 7, 2]]}
    
    masks = {
        "agent_0": [[1, 1, 0, 0, 1],
                    [1, 0, 1, 1, 1],
                    [0, 1, 1, 0, 1]],
        "agent_1": [[1, 0, 1, 1, 0],
                    [1, 1, 1, 0, 0],
                    [0, 1, 0, 1, 1],
                    [1, 1, 1, 1, 1]]
    }
    actions = {
    "agent_0": [2, 0, 4],
    "agent_1": [3, 1, 0, 2]
    }
    returns = {
    "agent_0": [0.0, 0.5, 1.0],
    "agent_1": [0.0, 0.0, 0.2, 1.5]
    }

    b = BatchInfo(obs, masks, actions, returns)

    assert isinstance(b.obs["agent_0"], np.ndarray)
    assert b.obs["agent_0"].shape == (3, 3)
    assert b.obs["agent_0"].dtype == np.dtype(np.int_)

    assert isinstance(b.obs["agent_1"], np.ndarray)
    assert b.obs["agent_1"].shape == (4, 3)
    assert b.obs["agent_1"].dtype == np.dtype(np.int_)


    assert isinstance(b.masks["agent_0"], np.ndarray)
    assert b.masks["agent_0"].shape == (3, 5)
    assert b.masks["agent_0"].dtype == np.dtype(np.bool_)

    assert isinstance(b.masks["agent_1"], np.ndarray)
    assert b.masks["agent_1"].shape == (4, 5)
    assert b.masks["agent_1"].dtype == np.dtype(np.bool_)


    assert isinstance(b.actions["agent_0"], np.ndarray)
    assert b.actions["agent_0"].shape == (3,)
    assert b.actions["agent_0"].dtype == np.dtype(np.int_)

    assert isinstance(b.actions["agent_1"], np.ndarray)
    assert b.actions["agent_1"].shape == (4,)
    assert b.actions["agent_1"].dtype == np.dtype(np.int_)


    assert isinstance(b.weights["agent_0"], np.ndarray)
    assert b.weights["agent_0"].shape == (3,)
    assert b.weights["agent_0"].dtype == np.dtype(np.float64)

    assert isinstance(b.weights["agent_1"], np.ndarray)
    assert b.weights["agent_1"].shape == (4,)
    assert b.weights["agent_1"].dtype == np.dtype(np.float64)



