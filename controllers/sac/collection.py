"""Local OpenDSS rollout state for SAC collection."""
from multiprocessing.util import Finalize

import torch

from opendss_environment import EnvironmentConfig, MicrogridEnv, Reward
from .actions import bess_actions
from .observation import LocalObservation


class Collector:
    def __init__(self, env, observations, actions, config):
        self.env, self.observations, self.actions, self.config = env, observations, actions, config

    def encode(self, observation):
        local = self.env.observe_controller(self.config.bus or self.observations.bus, {"bess": [self.config.bess_id]},
                                            observation=observation, extra_features=self.config.extra_features)
        return self.observations.encode(local, flatten=self.config.flatten)

    def reset(self, window):
        steps = window["episode_steps"]
        self.last = max(1, steps - 1)
        self.observation, _ = self.env.reset(seed=self.config.seed, options={"start_step": window["start_step"], "episode_steps": steps})
        self.observations.reset()
        return self.encode(self.observation)

    def step(self, action):
        command = self.env.compose_actions(self.actions.command(action), observation=self.observation)
        self.observation, _, terminated, truncated, info = self.env.step(command, full_info=False)
        truncated = truncated or (not terminated and self.env.idx == self.last)
        reward = Reward.get_local(info["rewards"], self.config.bess_id)
        return self.encode(self.observation) if self.observation is not None else None, reward, terminated, truncated

    @classmethod
    def create(cls, config, scaling, reward, data, first):
        options = EnvironmentConfig(config.case_path, first["episode_steps"], start_step=first["start_step"],
                                    observe_evse=config.observe_evse, reward_function=reward)
        env = MicrogridEnv(options, data=data)
        bess = config.bess(data)
        encoder = LocalObservation(bess, scaling, config.history_steps, config.observation_timezone)
        return cls(env, encoder, bess_actions(bess, scaling.s_base_kva, config, data["phases"]), config)

    @classmethod
    def initialize(cls, config, scaling, reward, data):
        torch.set_num_threads(1)
        cls.worker = cls.create(config, scaling, reward, data, {"episode_steps": 1, "start_step": 0})
        Finalize(cls.worker, cls.worker.env.close, exitpriority=10)

    @staticmethod
    def run(method, value):
        return getattr(Collector.worker, method)(value)
