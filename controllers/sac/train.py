"""SAC training/HPO contract and architecture construction."""
from datetime import datetime, timezone
from copy import deepcopy
import json

from torch import nn
import pandas as pd
import torch

from opendss_environment import EnvironmentConfig, MicrogridEnv

from ..normalization import ObservationScaling
from .observation import LocalObservation
from ..episodes import monthly_episodes
from .networks import Actor, Critic
from . import Config



class SACTrainer:
    def __init__(self, config=None):
        self.config = config if config is not None else Config()

    def architecture(self, input_size, action_size):
        self.actor = Actor(self.config.network, input_size, action_size, self.config)
        self.critic = nn.Module()
        self.critic.q1 = Critic(self.config.network, input_size, action_size, self.config)
        self.critic.q2 = Critic(self.config.network, input_size, action_size, self.config)
        return self.actor, self.critic

    def train(self, *, trial=None):
        """Plan all months and initialize the first episode; SAC updates remain pending."""
        config = self.config
        torch.manual_seed(config.seed)
        episodes, validation = monthly_episodes(config.case_path, config.episode_steps, config.seed)
        print(pd.DataFrame(episodes).to_string(index=False))
        first = episodes[0]
        env = MicrogridEnv(EnvironmentConfig(config.case_path, config.episode_steps))
        observation, _ = env.reset(seed=config.seed, options={
            "start_step": first["start_step"], "episode_steps": first["episode_steps"],
        })
        bess = next(device for device in env.data["devices"]["bess"] if config.bess_id is None or device["id"] == config.bess_id)
        self.scaling        = ObservationScaling.from_environment(env, bess["bus"])
        self.observations   = LocalObservation(bess, self.scaling, config.history_steps, config.observation_timezone)
        local               = env.observe_controller(config.bus or bess["bus"], {"bess": [bess["id"]]}, observation=observation, extra_features=config.extra_features)
        self.initial_state  = self.observations.encode(local, flatten=self.config.flatten)
        input_size          = self.observations.flat_size if self.config.flatten else self.observations.shape[1]
        action_size         = 2 * (len(bess["phases"]) if bess["dispatch_mode"] == "per_phase" else 1)

        self.architecture(input_size, action_size)
        for _ in range(self.config.num_training_episodes):
            # Perform training steps
            pass
        # Create target critics, replay and optimizers; collect transitions and apply SAC updates.
        # Iterate the plan for the training budget, then resample another monthly cycle.
        # Reset physical state/history per episode; never put held-out transitions in replay.
        # Reuse env.reset(options=...) for each window and reset self.observations.
        # Validation uses a separate environment and encoder; never add its transitions to replay.
        # Treat artificial window cutoffs explicitly before implementing SAC bootstrap targets.
        # Evaluate on validation periods/seeds; score is mean validation reward (higher is better).
        # Save weights, configs, scales and metrics.
        # Current initialization intentionally performs no env.step() or weight update.


        env.close()

        trial = trial or datetime.now(timezone.utc).strftime("trial_%Y%m%d_%H%M%S_%f")
        output = config.output_path(trial)
        result = {"status": "episodes_ready", "score": None, "episodes": episodes, "validation": validation,
                  "initial_timestamp": observation["timestamp"], "input_size": input_size, "action_size": action_size,
                  "settings": deepcopy(vars(config)), "flatten": config.flatten,
                  "trial": trial, "output_path": str(output)}
        output.mkdir(parents=True, exist_ok=False)
        (output / "initialization.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def hpo(self, *, trial=None):
        # Create the objective, then assign trial suggestions directly to self.config.
        # Set architecture first (loads its defaults), then history/layer/algorithm settings.
        # Each train() run must use fresh networks/replay and comparable budgets.
        result = self.train(trial=trial)
        # The study will maximize result["score"]; keep final test periods/seeds separate.
        return result


if __name__ == "__main__":
    SACTrainer().hpo()
