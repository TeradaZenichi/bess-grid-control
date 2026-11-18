"""Recurrent network that skips padded history steps."""
import torch
from torch import nn


class GRU(nn.Module):
    flatten = False

    def __init__(self, input_size, output_size, config):
        super().__init__()
        self.cell = nn.GRUCell(input_size, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, output_size)

    def forward(self, history, mask=None):
        hidden = history.new_zeros(history.shape[0], self.cell.hidden_size)
        for step, values in enumerate(history.unbind(1)):
            candidate = self.cell(values, hidden)
            hidden = candidate if mask is None else torch.where(mask[:, step, None], candidate, hidden)
        return self.output(hidden)
