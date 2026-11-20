from types import SimpleNamespace
import unittest

import numpy as np
import torch

from controllers.sac.train import ReplayBuffer


def settings(flatten=False, history=4, capacity=3):
    return SimpleNamespace(flatten=flatten, history_steps=history, replay_buffer_size=capacity,
                           batch_size=32, seed=42)


def observation(values, config):
    history = np.zeros((config.history_steps, 2), dtype=np.float32)
    values = values[-config.history_steps:]
    history[-len(values):] = [[value, value + 0.25] for value in values]
    mask = np.arange(config.history_steps) >= config.history_steps - len(values)
    return np.concatenate((history.ravel(), mask)) if config.flatten else {"history": history, "mask": mask}


class ReplayBufferTest(unittest.TestCase):
    def test_nsteps_stops_at_episode_end_and_preserves_wrapped_history(self):
        for terminated in (False, True):
            config = settings(capacity=3)
            config.nsteps, config.gamma = 3, 0.5
            replay = ReplayBuffer(config, observation([0], config), 1)
            for i in range(4):
                last = i == 3
                replay.add(observation(list(range(i + 1)), config), [i], i + 1,
                           observation(list(range(i + 2)), config),
                           last and terminated, last and not terminated)
                self.assertEqual(len(replay), (0, 0, 1, 3)[i])
                if i == 2:
                    full = replay.sample(1)
                    self.assertEqual(full["reward"].item(), 2.75)
                    self.assertEqual(full["discount"].item(), 0.125)
            replay.add(observation([99], config), [99], 99, observation([100], config), False, False)
            batch = replay.sample(32)
            actions = batch["action"][:, 0]
            self.assertEqual(set(actions.tolist()), {2, 3})
            for i, reward, discount in ((2, 5, 0.25), (3, 4, 0.5)):
                rows = actions == i
                torch.testing.assert_close(batch["reward"][rows], torch.full_like(batch["reward"][rows], reward))
                actual = batch["discount"][rows]
                torch.testing.assert_close(actual, torch.full_like(actual, 0 if terminated else discount))
                self.assertTrue((batch["terminated"][rows] == terminated).all())
                self.assertTrue((batch["truncated"][rows] != terminated).all())
                history = batch["next_state"]["history"][rows, :, 0]
                torch.testing.assert_close(history, torch.tensor([1., 2., 3., 4.]).expand_as(history))

    def test_wraparound_preserves_exact_histories_flags_and_episode_boundaries(self):
        for flatten in (False, True):
            config = settings(flatten)
            replay = ReplayBuffer(config, observation([0], config), 1)
            expected = {}
            counter = 0
            for episode, length in enumerate((2, 5, 3, 6)):
                values = [100 * episode]
                for step in range(length):
                    state = observation(values, config)
                    values.append(values[-1] + 1)
                    next_state = observation(values, config)
                    terminated = step == length - 1 and episode % 2 == 0
                    truncated = step == length - 1 and episode % 2 == 1
                    replay.add(state, [counter], counter / 10, next_state, terminated, truncated)
                    expected[counter] = (state, next_state, terminated, truncated)
                    counter += 1
                    if counter > 1:
                        batch = replay.sample(32)
                        for row, action in enumerate(batch["action"][:, 0].tolist()):
                            self.assertGreaterEqual(action, max(0, counter - config.replay_buffer_size))
                            current, following, terminal, cutoff = expected[int(action)]
                            for name, reference in (("state", current), ("next_state", following)):
                                if flatten:
                                    torch.testing.assert_close(batch[name][row], torch.from_numpy(reference))
                                else:
                                    for key in ("history", "mask"):
                                        torch.testing.assert_close(batch[name][key][row], torch.from_numpy(reference[key]))
                            self.assertAlmostEqual(batch["reward"][row].item(), action / 10, places=5)
                            self.assertEqual(batch["terminated"][row].item(), terminal)
                            self.assertEqual(batch["truncated"][row].item(), cutoff)
                            self.assertAlmostEqual(batch["discount"][row].item(), 0 if terminal else 0.99)
            self.assertEqual(len(replay), config.replay_buffer_size)

    def test_insertion_copies_values_and_external_reset_clears_history(self):
        config = settings(history=1)
        state, following = observation([1], config), observation([2], config)
        action = np.array([0.5], dtype=np.float32)
        replay = ReplayBuffer(config, state, 1)
        replay.add(state, action, 1, following, False, False)
        state["history"][:] = 99
        following["history"][:] = 99
        action[:] = 99
        batch = replay.sample(1)
        self.assertEqual(batch["state"]["history"][0, -1, 0].item(), 1)
        self.assertEqual(batch["next_state"]["history"][0, -1, 0].item(), 2)
        self.assertEqual(batch["action"].item(), 0.5)
        config = settings(capacity=1)
        replay = ReplayBuffer(config, observation([0], config), 1)
        replay.add(observation([1], config), [1], 0, observation([1, 2], config), False, False)
        replay.start_episode()
        replay.add(observation([7], config), [7], 0, observation([7, 8], config), False, False)
        self.assertEqual(replay.sample(1)["state"]["mask"].tolist(), [[False, False, False, True]])

    def test_terminal_none_is_supported_without_reusing_reset_observation(self):
        config = settings(capacity=1)
        replay = ReplayBuffer(config, observation([1], config), 1)
        replay.add(observation([1], config), [0], 2, None, True, False)
        batch = replay.sample(1)
        self.assertTrue(batch["terminated"].item())
        self.assertFalse(batch["truncated"].item())
        self.assertEqual(batch["next_state"]["history"][0, -1].tolist(), [0, 0])

    def test_storage_is_compact_for_week_long_history_and_sampling_is_seeded(self):
        config = settings(history=672, capacity=10000)
        initial = {"history": np.zeros((672, 32), dtype=np.float32), "mask": np.ones(672, dtype=bool)}
        replay = ReplayBuffer(config, initial, 6)
        self.assertLess(replay.memory_bytes, 4 * 1024**2)
        buffers = [ReplayBuffer(settings(), observation([0], settings()), 1) for _ in range(2)]
        for buffer in buffers:
            for i in range(3):
                buffer.add(observation([i], settings()), [i], i, observation([i + 1], settings()), True, False)
        torch.testing.assert_close(buffers[0].sample()["action"], buffers[1].sample()["action"])


if __name__ == "__main__":
    unittest.main()
