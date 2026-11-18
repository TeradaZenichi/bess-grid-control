"""Local BESS policy; the actor starts untrained."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from opendss_environment import PhaseAction, TotalAction

from ..base import BaseController
from ..config import load_config
from .architectures import ARCHITECTURES
from .networks import Actor, observation_batch
from .observation import LocalObservation


class SACController(BaseController):
    """Encode local measurements and dispatch one actor's bounded request."""

    def __init__(self, bess, settings=None, architecture_settings=None, *, scaling):
        self.parameters = deepcopy(bess)
        self.scaling = scaling
        folder = Path(__file__).parent
        self.config = load_config(folder / "config.json", settings)
        self.id, self.bus = bess["id"], bess["bus"]
        self.architecture = ARCHITECTURES[self.config.architecture]
        self.architecture_config = load_config(
            folder / "architectures" / self.config.architecture / "config.json", architecture_settings)
        self.extra_features = tuple(self.config.extra_features)
        self.observations = LocalObservation(self.parameters, scaling, self.architecture_config.history_steps,
                                            self.config.observation_timezone)
        self.phases = tuple(bess["phases"])
        count = len(self.phases) if bess["dispatch_mode"] == "per_phase" else 1
        charge = min(bess["p_charge_max_kw"], bess["s_max_kva"]) / count
        discharge = min(bess["p_discharge_max_kw"], bess["s_max_kva"]) / count
        reactive = bess["s_max_kva"] / count if bess["reactive_control"] else 0
        self.action_low = np.array([-discharge] * count + [-reactive] * count) / scaling.s_base_kva
        self.action_high = np.array([charge] * count + [reactive] * count) / scaling.s_base_kva
        self.action_size = 2 * count
        self.input_size = self.observations.flat_size if self.architecture.flatten else self.observations.shape[1]
        self.actor = Actor(self.architecture, self.input_size, self.action_size, self.architecture_config)

    def reset(self):
        self.observations.reset()

    def observe(self, observation):
        return self.observations.encode(observation, flatten=self.architecture.flatten)

    def sample(self, state, deterministic=False):
        with torch.no_grad():
            action, log_prob = self.actor(observation_batch([state]), deterministic)
        return action[0].numpy(), log_prob.item()

    def command(self, action):
        power = self.action_low + (np.asarray(action) + 1) * (self.action_high - self.action_low) / 2
        p, q = np.split(power, 2)
        if self.parameters["dispatch_mode"] == "aggregate":
            return TotalAction(float(p[0]), float(q[0]))
        return PhaseAction(dict(zip(self.phases, p.tolist())), dict(zip(self.phases, q.tolist())))

    def act(self, observation, objective=None):
        action, _ = self.sample(self.observe(observation), deterministic=True)
        return self.command(action)
