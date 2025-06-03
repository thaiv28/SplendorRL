from abc import abstractmethod

from torch.optim import Adam
import numpy as np

from splendor.sample import BatchInfo

class Algorithm:
    def __init__(self, policy_network, optimizer = "adam", lr: float = 1e-2,
                 gamma: float = 0.99):
        """Set policy network and optimizer for algorithm"""

        self.policy_network = policy_network
        self.gamma = gamma

        match optimizer.lower():
            case "adam":
                self.optimizer = Adam(policy_network.parameters(), lr=lr)
            case _:
                raise NotImplementedError("Optimizer not supported", optimizer)

    @abstractmethod
    def update(self, batch: BatchInfo) -> dict[str, np.ndarray]:
        pass
