"""Local BESS policy; the actor starts untrained."""
from copy import deepcopy

import torch

from opendss_environment import ControllerActions
from opendss_environment.elements import BESSConfig

from ..base import BaseController
from . import Config
from .networks import Actor, observation_batch
from .observation import LocalObservation


class SACController(BaseController):
    """Encode local measurements and dispatch one actor's bounded request."""

    def __init__(self, bess, settings=None, architecture_settings=None, *, scaling):
        self.parameters = deepcopy(bess)
        self.scaling = scaling
        self.config = Config(settings, architecture_settings)
        self.id, self.bus = bess["id"], bess["bus"]
        self.architecture = self.config.network
        self.architecture_config = self.config
        self.extra_features = tuple(self.config.extra_features)
        self.observations = LocalObservation(self.parameters, scaling, self.architecture_config.history_steps,
                                            self.config.observation_timezone)
        self.phases = tuple(bess["phases"])
        self.actions = ControllerActions({"bess": [BESSConfig(**bess)]}, scaling.s_base_kva, seed=self.config.seed)
        self.action_space = self.actions.action_space
        self.action_low, self.action_high = self.actions.low, self.actions.high
        self.action_size = self.action_space.shape[0]
        self.input_size = self.observations.flat_size if self.config.flatten else self.observations.shape[1]
        self.actor = Actor(self.architecture, self.input_size, self.action_size, self.architecture_config)

    def reset(self):
        self.observations.reset()

    def observe(self, observation):
        return self.observations.encode(observation, flatten=self.config.flatten)

    def sample(self, state, deterministic=False):
        with torch.no_grad():
            action, log_prob = self.actor(observation_batch([state]), deterministic)
        return action[0].numpy(), log_prob.item()

    def command(self, action):
        return self.actions.command(action).bess[self.id]

    def act(self, observation, objective=None):
        action, _ = self.sample(self.observe(observation), deterministic=True)
        return self.command(action)
