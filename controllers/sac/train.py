"""SAC training/HPO contract and architecture construction."""
from datetime import datetime, timezone
from types import SimpleNamespace
from copy import deepcopy
import json

from torch import nn
import pandas as pd
import numpy as np
import torch

from opendss_environment import EnvironmentConfig, MicrogridEnv

from ..normalization import ObservationScaling
from .observation import LocalObservation
from ..episodes import monthly_episodes
from .networks import Actor, Critic
from . import Config


class ReplayBuffer:
    """Uniform replay storing individual measurements, not overlapping windows."""

    def __init__(self, config, observation, action_size):
        self.capacity = config.replay_buffer_size
        self.history_steps = config.history_steps
        self.flatten = config.flatten
        self.batch_size = config.batch_size
        self.rng = np.random.default_rng(config.seed)
        self.total, self.episode = 0, 0
        # Keep the prefix needed by the oldest sampleable transition.
        self.slots = self.capacity + self.history_steps - 1
        features = self._latest(observation).size
        self.data = {
            "state": np.empty((self.slots, features), dtype=np.float32),
            "next_state": np.empty((self.slots, features), dtype=np.float32),
            "action": np.empty((self.slots, action_size), dtype=np.float32),
            "reward": np.empty((self.slots, 1), dtype=np.float32),
            "terminated": np.empty((self.slots, 1), dtype=bool),
            "truncated": np.empty((self.slots, 1), dtype=bool),
            "episode": np.full(self.slots, -1, dtype=np.int64),
        }

    def _latest(self, observation):
        if isinstance(observation, dict):
            return observation["history"][-1]
        return np.asarray(observation)[:-self.history_steps].reshape(self.history_steps, -1)[-1]

    def start_episode(self):
        """Call after an external reset, together with the observation encoder reset."""
        self.episode += 1

    def add(self, state, action, reward, next_state, terminated, truncated):
        slot = self.total % self.slots
        self.data["state"][slot] = self._latest(state)
        self.data["next_state"][slot] = 0 if next_state is None and terminated else self._latest(next_state)
        for name, value in (("action", action), ("reward", reward),
                            ("terminated", terminated), ("truncated", truncated), ("episode", self.episode)):
            self.data[name][slot] = value
        self.total += 1
        if terminated or truncated:
            self.start_episode()

    def _encode(self, history, mask):
        if self.flatten:
            return torch.from_numpy(np.concatenate((history.reshape(len(history), -1), mask), axis=1))
        return {"history": torch.from_numpy(history), "mask": torch.from_numpy(mask)}

    def sample(self, batch_size=None):
        indices = self.rng.integers(max(0, self.total - self.capacity), self.total,
                                    size=self.batch_size if batch_size is None else batch_size)
        slots = indices % self.slots
        history_indices = indices[:, None] - np.arange(self.history_steps - 1, -1, -1)
        history_slots = history_indices % self.slots
        mask = (history_indices >= max(0, self.total - self.slots)) & (
            self.data["episode"][history_slots] == self.data["episode"][slots, None])
        history = np.where(mask[..., None], self.data["state"][history_slots], 0)
        next_history = np.concatenate((history[:, 1:], self.data["next_state"][slots, None]), axis=1)
        next_mask = np.concatenate((mask[:, 1:], np.ones((len(slots), 1), dtype=bool)), axis=1)
        batch = {name: torch.from_numpy(self.data[name][slots])
                 for name in ("action", "reward", "terminated", "truncated")}
        return batch | {"state": self._encode(history, mask),
                        "next_state": self._encode(next_history, next_mask)}

    def __len__(self):
        return min(self.total, self.capacity)

    @property
    def memory_bytes(self):
        return sum(values.nbytes for values in self.data.values())


class SACTrainer:
    def __init__(self, config=None):
        self.config = config if config is not None else Config()

    def architecture(self, input_size, action_size):
        self.actor      = Actor(self.config.network, input_size, action_size, self.config)
        self.critic     = nn.Module()
        self.critic.q1  = Critic(self.config.network, input_size, action_size, self.config)
        self.critic.q2  = Critic(self.config.network, input_size, action_size, self.config)
        self.target     = nn.Module()
        self.target.actor = deepcopy(self.actor)
        self.target.q1    = deepcopy(self.critic.q1)
        self.target.q2    = deepcopy(self.critic.q2)
        self.target.requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(self.config.alpha, dtype=torch.float32).log())
        self.target_entropy = -float(action_size)
        self.optimizer = SimpleNamespace(
            actor=torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr),
            critic=torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr),
            alpha=torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr),
        )
        return self.actor, self.critic

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self, *, trial=None):
        """Plan all months and initialize the first episode; SAC updates remain pending."""
        torch.manual_seed(self.config.seed)
        episodes, validation = monthly_episodes(self.config.case_path, self.config.episode_steps, self.config.seed)
        self.validation = validation
        self.env = SimpleNamespace(
            train=MicrogridEnv(EnvironmentConfig(self.config.case_path, self.config.episode_steps)),
            eval=MicrogridEnv(EnvironmentConfig(self.config.case_path, validation[0]["episode_steps"],
                                               start_step=validation[0]["start_step"])),
        )
        start, steps = episodes[0]["start_step"], episodes[0]["episode_steps"]
        bess = self.config.bess(self.env.train.data)
        observation, _      = self.env.train.reset(seed=self.config.seed, options={"start_step": start, "episode_steps": steps})
        initial_timestamp = observation["timestamp"]
        self.scaling        = ObservationScaling.from_environment(self.env.train, bess["bus"])
        self.observations   = LocalObservation(bess, self.scaling, self.config.history_steps, self.config.observation_timezone)
        self.eval_observations = LocalObservation(bess, self.scaling, self.config.history_steps, self.config.observation_timezone)
        local               = self.env.train.observe_controller(self.config.bus or bess["bus"], {"bess": [self.config.bess_id]}, observation=observation, extra_features=self.config.extra_features)
        self.initial_state  = self.observations.encode(local, flatten=self.config.flatten)
        input_size          = self.observations.flat_size if self.config.flatten else self.observations.shape[1]
        action_size         = 2 * (len(bess["phases"]) if bess["dispatch_mode"] == "per_phase" else 1)

        self.architecture(input_size, action_size)

        self.replay         = ReplayBuffer(self.config, self.initial_state, action_size)

        for _ in range(self.config.num_training_episodes):
            # choice a random episode from episodes
            episode = episodes[torch.randint(len(episodes), (1,)).item()]
            start, steps = episode["start_step"], episode["episode_steps"]
            observation, _ = self.env.train.reset(seed=self.config.seed, options={"start_step": start, "episode_steps": steps})
            self.observations.reset()
            self.replay.start_episode()
            terminated = truncated = False
            while not (terminated or truncated):
                observation, reward, terminated, truncated, info = self.env.train.step()


            pass

        # Iterate the plan for the training budget, then resample another monthly cycle.
        # Reset physical state/history per episode; never put held-out transitions in replay.
        # Reuse self.env.train.reset(options=...) for each training window.
        # Reuse self.env.eval for self.validation windows and reset self.eval_observations.
        # Never add validation transitions to replay.
        # Treat artificial window cutoffs explicitly before implementing SAC bootstrap targets.
        # Evaluate on validation periods/seeds; score is mean validation reward (higher is better).
        # Save weights, configs, scales and metrics.
        # Collection still uses default device operation; no weight updates are implemented.


        self.env.train.close()
        self.env.eval.close()

        trial = trial or datetime.now(timezone.utc).strftime("trial_%Y%m%d_%H%M%S_%f")
        output = self.config.output_path(trial)
        result = {"status": "episodes_ready", "score": None, "episodes": episodes, "validation": validation,
                  "initial_timestamp": initial_timestamp, "input_size": input_size, "action_size": action_size,
                  "settings": deepcopy(vars(self.config)), "flatten": self.config.flatten,
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
