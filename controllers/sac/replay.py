"""Chronological replay for SAC training."""
import numpy as np
import torch


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
