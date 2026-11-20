"""SAC training/HPO contract and architecture construction."""
from datetime import datetime, timezone
from types import SimpleNamespace
from copy import deepcopy
import json

from torch import nn
import numpy as np
import torch

from opendss_environment import EnvironmentConfig, MicrogridEnv, Reward
from opendss_environment.data import load_data

from ..normalization import ObservationScaling
from .observation import LocalObservation
from ..episodes import split_episodes
from .networks import Actor, Critic, observation_batch
from . import Config


class ReplayBuffer:
    """Uniform replay storing individual measurements, not overlapping windows."""

    def __init__(self, config, observation, action_size):
        self.capacity = config.replay_buffer_size
        self.nsteps = getattr(config, "nsteps", 1)
        if type(self.nsteps) is not int or not 1 <= self.nsteps <= self.capacity:
            raise ValueError("nsteps must be an integer between 1 and replay_buffer_size")
        self.gamma = getattr(config, "gamma", 0.99)
        self.history_steps = config.history_steps
        self.flatten = config.flatten
        self.batch_size = config.batch_size
        self.rng = np.random.default_rng(config.seed)
        self.total, self.episode = 0, 0
        self.episode_start = 0
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
        self.episode_start = self.total

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

    def _history(self, indices):
        slots = indices % self.slots
        history_indices = indices[:, None] - np.arange(self.history_steps - 1, -1, -1)
        history_slots = history_indices % self.slots
        mask = (history_indices >= max(0, self.total - self.slots)) & (
            self.data["episode"][history_slots] == self.data["episode"][slots, None])
        history = np.where(mask[..., None], self.data["state"][history_slots], 0)
        return history, mask

    def sample(self, batch_size=None):
        oldest = max(0, self.total - self.capacity)
        indices = self.rng.integers(oldest, oldest + len(self),
                                    size=self.batch_size if batch_size is None else batch_size)
        slots = indices % self.slots
        end = indices.copy()
        reward = self.data["reward"][slots]
        discount = np.full_like(reward, self.gamma)
        for _ in range(1, self.nsteps):
            current, following = end % self.slots, (end + 1) % self.slots
            advance = (end + 1 < self.total) & ~self.data["terminated"][current, 0] & ~self.data["truncated"][current, 0]
            advance &= self.data["episode"][following] == self.data["episode"][current]
            if not advance.any():
                break
            reward[advance] += discount[advance] * self.data["reward"][following[advance]]
            discount[advance] *= self.gamma
            end[advance] += 1

        last = end % self.slots
        history, mask = self._history(indices)
        final_history, final_mask = (history, mask) if self.nsteps == 1 else self._history(end)
        next_history = np.concatenate((final_history[:, 1:], self.data["next_state"][last, None]), axis=1)
        next_mask = np.concatenate((final_mask[:, 1:], np.ones((len(last), 1), dtype=bool)), axis=1)
        batch = {name: torch.from_numpy(self.data[name][last]) for name in ("terminated", "truncated")}
        discount[self.data["terminated"][last]] = 0
        batch.update(action=torch.from_numpy(self.data["action"][slots]),
                     reward=torch.from_numpy(reward), discount=torch.from_numpy(discount))
        return batch | {"state": self._encode(history, mask),
                        "next_state": self._encode(next_history, next_mask)}

    def __len__(self):
        # The newest starts in an open episode still need future rewards.
        pending = min(self.nsteps - 1, self.total - self.episode_start)
        return min(self.total, self.capacity) - pending

    @property
    def memory_bytes(self):
        return sum(values.nbytes for values in self.data.values())


class SACTrainer:
    def __init__(self, config=None):
        self.config = config if config is not None else Config()
        self.data   = load_data(self.config.case_path)
        self.episodes, self.validation = split_episodes(self.data)

    def architecture(self, input_size, action_size):
        self.actor        = Actor(self.config.network, input_size, action_size, self.config)
        self.critic       = nn.Module()
        self.critic.q1    = Critic(self.config.network, input_size, action_size, self.config)
        self.critic.q2    = Critic(self.config.network, input_size, action_size, self.config)
        self.target       = nn.Module()
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

    def encode_local(self, observation):
        local = self.env.train.observe_controller(
            self.config.bus or self.observations.bus, {"bess": [self.config.bess_id]},
            observation=observation, extra_features=self.config.extra_features,
        )
        return self.observations.encode(local, flatten=self.config.flatten)

    def train(self, *, trial=None):
        """Reuse cached chronological windows; SAC updates remain pending."""
        torch.manual_seed(self.config.seed)
        self.env = SimpleNamespace(
            train=MicrogridEnv(EnvironmentConfig(self.config.case_path, self.episodes[0]["episode_steps"]), data=self.data),
            eval=MicrogridEnv(EnvironmentConfig(self.config.case_path, self.validation[0]["episode_steps"],
                                               start_step=self.validation[0]["start_step"]), data=self.data),
        )
        start, steps = self.episodes[0]["start_step"], self.episodes[0]["episode_steps"]
        bess = self.config.bess(self.env.train.data)
        observation, _      = self.env.train.reset(seed=self.config.seed, options={"start_step": start, "episode_steps": steps})
        initial_timestamp = observation["timestamp"]
        self.scaling        = ObservationScaling.from_environment(self.env.train, bess["bus"])
        self.observations   = LocalObservation(bess, self.scaling, self.config.history_steps, self.config.observation_timezone)
        self.eval_observations = LocalObservation(bess, self.scaling, self.config.history_steps, self.config.observation_timezone)
        self.initial_state  = self.encode_local(observation)
        input_size          = self.observations.flat_size if self.config.flatten else self.observations.shape[1]
        self.actions = self.env.train.controller_actions({"bess": [self.config.bess_id]}, seed=self.config.seed)
        action_size = self.actions.action_space.shape[0]

        self.architecture(input_size, action_size)

        self.replay         = ReplayBuffer(self.config, self.initial_state, action_size)
        self.env.train.reward_function = Reward(controllers={self.config.bess_id: {"devices": {"bess": [self.config.bess_id]}, "buses": [bess["bus"]], "weights": {self.config.reward_term: 1.0}}})

        for index in range(self.config.num_training_episodes):
            episode         = self.episodes[index % len(self.episodes)]
            start, steps    = episode["start_step"], episode["episode_steps"]
            observation, _  = self.env.train.reset(seed=self.config.seed, options={"start_step": start, "episode_steps": steps})
            self.observations.reset()
            self.replay.start_episode()
            terminated = truncated = False
            state = self.encode_local(observation)
            while not (terminated or truncated):
                if self.replay.total < self.config.warmup_steps:
                    action = self.actions.action_space.sample()
                else:
                    with torch.no_grad():
                        action, _ = self.actor(observation_batch([state]))
                        action = action[0].cpu().numpy()
                command = self.env.train.compose_actions(self.actions.command(action))
                next_observation, _, terminated, truncated, info = self.env.train.step(command)
                reward = Reward.get_local(info["rewards"], self.config.bess_id)
                next_state = self.encode_local(next_observation) if next_observation is not None else None
                self.replay.add(state, action, reward, next_state, terminated, truncated)

                # Before enabling backward, resolve window truncation and its final next_state.
                if self.replay.total >= self.config.learning_starts and len(self.replay) >= self.config.batch_size:
                    self.update(self.replay.sample())

                state = next_state
                observation = next_observation

        self.env.train.close()
        self.env.eval.close()

        trial = trial or datetime.now(timezone.utc).strftime("trial_%Y%m%d_%H%M%S_%f")
        output = self.config.output_path(trial)
        result = {"status": "episodes_ready", "score": None, "episodes": self.episodes, "validation": self.validation,
                  "initial_timestamp": initial_timestamp, "input_size": input_size, "action_size": action_size,
                  "settings": deepcopy(vars(self.config)), "flatten": self.config.flatten,
                  "trial": trial, "output_path": str(output)}
        output.mkdir(parents=True, exist_ok=False)
        (output / "initialization.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    def update(self, batch):
        # 1. Under no_grad, sample next actions/log_probs from the current actor.
        with torch.no_grad():
            next_action, next_log_prob = self.actor(batch["next_state"])
            next_q1 = self.target.q1(batch["next_state"], next_action)
            next_q2 = self.target.q2(batch["next_state"], next_action)
            next_value = torch.minimum(next_q1, next_q2) - self.alpha * next_log_prob
            backup = batch["reward"] + batch["discount"] * next_value

        # 2. Sum the MSE losses of both online critics against the detached backup.
        #    self.optimizer.critic.zero_grad(); critic_loss.backward(); self.optimizer.critic.step().
        # 3. Freeze critic parameters, retaining gradients through sampled actor actions.
        #    actor_loss = mean(self.alpha.detach() * log_prob - min(Q1, Q2)).
        #    self.optimizer.actor.zero_grad(); actor_loss.backward(); self.optimizer.actor.step().
        #    Re-enable critic parameter gradients afterwards.
        # 4. alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean().
        #    self.optimizer.alpha.zero_grad(); alpha_loss.backward(); self.optimizer.alpha.step().
        # 5. Under no_grad, soft-update target critics: target = (1 - tau) * target + tau * online.
        #    Add tau to the SAC configuration. Standard SAC does not use target.actor.
        # Return detached loss values for logging, without retaining computation graphs.
        pass

    def hpo(self, *, trial=None):
        # Create the objective, then assign trial suggestions directly to self.config.
        # Set architecture first (loads its defaults), then history/layer/algorithm settings.
        # Each train() run must use fresh networks/replay and comparable budgets.
        result = self.train(trial=trial)
        # The study will maximize result["score"]; keep final test periods/seeds separate.
        return result


if __name__ == "__main__":
    SACTrainer().hpo()
