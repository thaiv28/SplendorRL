import torch.nn as nn
from torch.masked import masked_tensor
from torchrl.modules import MaskedCategorical

class MLP:
    def __init__(self, sizes, activation=nn.Tanh, output_activation=nn.Identity):
        """Create a multilayer perceptron with given sizes and activations."""

        self.nn = MLP.create_network(sizes, activation, output_activation)

    @staticmethod
    def create_network(sizes, activation, output_activation):
        layers = []
        for j in range(len(sizes) - 1):
            if j < len(sizes) - 2:
                act = activation
            else:
                act = output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        return nn.Sequential(*layers)

    def get_action(self, obs, mask=None):
        """Return action sampled from distribution given observation."""

        dist = self._get_distribution(obs, mask)
        return dist.sample().item()

    def _get_distribution(self, obs, mask):
        """Return categorical distribution of actions for given observation."""

        logits = self.nn(obs)
    
        return MaskedCategorical(logits=logits, mask=mask)

    def parameters(self):
        return self.nn.parameters()

