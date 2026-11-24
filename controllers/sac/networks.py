"""A squashed Gaussian actor and independent state-action critics."""
from math import log

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn.functional import softplus


def observation_batch(observations):
    if isinstance(observations[0], dict):
        return {key: torch.from_numpy(np.stack([item[key] for item in observations]))
                for key in observations[0]}
    return torch.from_numpy(np.stack(observations))


def encode(network, observations):
    return network(**observations) if isinstance(observations, dict) else network(observations)


class Actor(nn.Module):
    def __init__(self, architecture, input_size, action_size, config):
        super().__init__()
        self.network = architecture(input_size, 2 * action_size, config)
        self.power_actions = None

    def project(self, observations, action):
        return self.power_actions.project(observations, action) if self.power_actions is not None else action

    def deterministic_action(self, observations):
        """Return the squashed mean for callers that only need an action."""
        mean, _ = encode(self.network, observations).chunk(2, dim=-1)
        return self.project(observations, mean.tanh())

    def forward(self, observations, deterministic=False):
        mean, log_std = encode(self.network, observations).chunk(2, dim=-1)
        distribution = Normal(mean, log_std.clamp(-20, 2).exp())
        raw = mean if deterministic else distribution.rsample()
        log_prob = distribution.log_prob(raw) - 2 * (log(2) - raw - softplus(-2 * raw))
        # Entropy is measured on the latent squashed action; critics see normalized power.
        return self.project(observations, raw.tanh()), log_prob.sum(dim=-1, keepdim=True)


class Critic(nn.Module):
    def __init__(self, architecture, input_size, action_size, config):
        super().__init__()
        hidden = config.hidden_sizes[-1] if architecture.flatten else config.hidden_size
        self.encoder = architecture(input_size, hidden, config)
        self.value = nn.Sequential(nn.Linear(hidden + action_size, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, observations, actions):
        state = encode(self.encoder, observations)
        return self.value(torch.cat((state, actions), dim=-1))
