"""Feedforward network for flattened observation histories."""
from torch import nn


class MLP(nn.Sequential):
    flatten = True

    def __init__(self, input_size, output_size, config):
        sizes = (input_size, *config.hidden_sizes)
        layers = []
        for before, after in zip(sizes, sizes[1:]):
            layers.extend((nn.Linear(before, after), nn.ReLU()))
        super().__init__(*layers, nn.Linear(sizes[-1], output_size))
