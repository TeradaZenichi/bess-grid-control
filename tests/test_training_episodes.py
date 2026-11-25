from pathlib import Path
import json
from tempfile import TemporaryDirectory
from threading import Event
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np
import torch

from controllers.episodes import split_episodes
from opendss_environment.data import load_data
from controllers.base import Config
from controllers.sac.networks import observation_batch
from controllers.sac.evaluation import Evaluator
from controllers.sac.train import SACTrainer
from controllers.sac.controller import SACController
from opendss_environment import MicrogridEnv
from tools.evaluate_controller import evaluate


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "data/switzerland/5bus/config.json"


class TrainingEpisodesTest(unittest.TestCase):
    def test_split_covers_every_training_year_row_once_in_chronological_order(self):
        data = load_data(CASE)
        plan, validation = split_episodes(data)
        self.assertEqual(len(plan), 13)
        coverage = np.zeros(data["steps"], dtype=int)
        for windows in (plan, validation):
            self.assertEqual([row["start_step"] for row in windows], sorted(row["start_step"] for row in windows))
            for row in windows:
                coverage[row["start_step"]:row["end_step_exclusive"]] += 1
        times = pd.DatetimeIndex(pd.to_datetime(data["timestamps"], utc=True)).tz_convert("Europe/Zurich")
        np.testing.assert_array_equal(coverage, (times.year == 2023).astype(int))
        self.assertEqual(sum(row["episode_steps"] for row in plan), 26976)
        self.assertEqual(sum(row["episode_steps"] for row in validation), 8064)
        for episode in plan:
            start = pd.Timestamp(episode["start"])
            end = pd.Timestamp(episode["end_exclusive"])
            dates = times[episode["start_step"]:episode["end_step_exclusive"]]
            self.assertTrue((dates.year == 2023).all())
            self.assertFalse(((dates.day >= 15) & (dates.day < 22)).any())
            self.assertEqual(start.hour, 0)
            self.assertEqual(end - start, pd.Timedelta(minutes=15 * episode["episode_steps"]))
            self.assertEqual(episode["end_step_exclusive"] - episode["start_step"], episode["episode_steps"])
        self.assertEqual(len(validation), 12)
        for block in validation:
            self.assertEqual(pd.Timestamp(block["start"]).day, 15)
            self.assertEqual(pd.Timestamp(block["end_exclusive"]).day, 22)
            self.assertEqual(block["end_step_exclusive"] - block["start_step"], block["episode_steps"])
            self.assertEqual(block["episode_steps"], 672)

    def test_train_initializes_selected_window_and_networks_without_stepping(self):
        with TemporaryDirectory() as directory, patch("controllers.sac.Config.output_path",
                lambda config, trial: Config.output_path(config, trial, root=directory)), \
                patch.object(MicrogridEnv, "step") as step:
            with patch.object(MicrogridEnv, "close", autospec=True, side_effect=MicrogridEnv.close) as close:
                for architecture in ("mlp", "gru"):
                    trainer = SACTrainer()
                    trainer.config.architecture = architecture
                    trainer.config.history_steps = 4
                    trainer.config.num_training_episodes = 0
                    result = trainer.train()
                    self.assertEqual(result["status"], "episodes_ready")
                    self.assertIsNone(result["score"])
                    self.assertEqual(pd.Timestamp(result["initial_timestamp"]), pd.Timestamp(result["episodes"][0]["start"]))
                    self.assertEqual(result["input_size"], 132 if architecture == "mlp" else 32)
                    self.assertEqual(result["action_size"], 6)
                    self.assertIsNot(trainer.env.train, trainer.env.eval)
                    self.assertIsNot(trainer.env.train.dss, trainer.env.eval.dss)
                    self.assertTrue(trainer.env.train.evse_list)
                    self.assertNotIn("evse", trainer.env.train.observe())
                    self.assertIsNot(trainer.observations, trainer.eval_observations)
                    self.assertEqual(len(trainer.eval_observations.history), 0)
                    self.assertEqual(trainer.validation, result["validation"])
                    self.assertEqual(len(trainer.validation), 12)
                    self.assertEqual(pd.Timestamp(trainer.env.eval.timestamps.iloc[0]),
                                     pd.Timestamp(trainer.validation[0]["start"]))
                    self.assertEqual(len(trainer.replay), 0)
                    self.assertLess(trainer.replay.memory_bytes, 4 * 1024**2)
                    output = Path(result["output_path"])
                    self.assertEqual(json.loads((output / "initialization.json").read_text()), result)
                    self.assertEqual(output.parts[-3:], (architecture, result["trial"], "seed_42"))
                    state = trainer._to_device(observation_batch([trainer.initial_state]))
                    action, _ = trainer.actor(state)
                    self.assertEqual(tuple(action.shape), (1, 6))
                    self.assertEqual(tuple(trainer.critic.q1(state, action).shape), (1, 1))
                    self.assertEqual(tuple(trainer.critic.q2(state, action).shape), (1, 1))
                    for name, online in (("actor", trainer.actor), ("q1", trainer.critic.q1), ("q2", trainer.critic.q2)):
                        target = getattr(trainer.target, name)
                        for source, copied in zip(online.parameters(), target.parameters()):
                            torch.testing.assert_close(source, copied)
                            self.assertNotEqual(source.data_ptr(), copied.data_ptr())
                            self.assertFalse(copied.requires_grad)
                        source, copied = next(online.parameters()), next(target.parameters())
                        before = copied.clone()
                        with torch.no_grad():
                            source.add_(1)
                        torch.testing.assert_close(copied, before)
                    mask = trainer.initial_state[-4:] if architecture == "mlp" else trainer.initial_state["mask"]
                    self.assertEqual(mask.sum(), 1)
                self.assertEqual(close.call_count, 4)
            step.assert_not_called()

    def test_collection_can_sample_local_warmup_or_actor_without_global_action_space(self):
        trainer = SACTrainer()
        trainer.config.num_training_episodes = 1
        first = trainer.episodes[0]
        trainer.episodes = [first | {"episode_steps": 3, "end_step_exclusive": first["start_step"] + 3,
                                   "end_exclusive": (pd.Timestamp(first["start"]) + pd.Timedelta(minutes=45)).isoformat()}]
        with TemporaryDirectory() as directory, patch("controllers.sac.Config.output_path",
                lambda config, trial: Config.output_path(config, trial, root=directory)):
            for warmup in (1000, 0):
                trainer.config.warmup_steps = warmup
                trainer.train()
                self.assertEqual(trainer.env.train.idx, 2)
                self.assertEqual(trainer.actions.action_space.shape, (6,))
                self.assertEqual(len(trainer.replay), 2)
                self.assertEqual(trainer.replay.data["terminated"][:2, 0].tolist(), [False, False])
                self.assertEqual(trainer.replay.data["truncated"][:2, 0].tolist(), [False, True])
                continuation = trainer.encode_local(trainer.env.train.observe())
                np.testing.assert_array_equal(trainer.replay.data["next_state"][1], trainer.replay._latest(continuation))
                torch.testing.assert_close(trainer.replay.sample(4)["discount"], torch.full((4, 1), trainer.config.gamma))
                self.assertTrue(np.isfinite(trainer.replay.data["reward"][:2]).all())
                self.assertTrue((np.abs(trainer.replay.data["action"][:2]) <= 1).all())

    def test_train_scores_held_out_windows_without_adding_them_to_replay(self):
        trainer = SACTrainer()
        trainer.config.num_training_episodes = 4
        trainer.config.learning_starts = trainer.config.batch_size = 2
        trainer.episodes = [window | {"episode_steps": 3} for window in trainer.episodes[:3]]
        trainer.validation = [window | {"episode_steps": steps}
                              for window, steps in zip(trainer.validation, (2, 3))]
        validation_started, training_resumed = Event(), Event()
        snapshots = {}
        evaluate_actor, update_models = trainer.evaluate, trainer.update

        def evaluate_snapshot(*, actor):
            if not validation_started.is_set():
                validation_started.set()
                self.assertTrue(training_resumed.wait(10), "Episode 2 did not update during validation")
            return evaluate_actor(actor=actor)

        def update_during_validation(batch):
            steps = trainer.replay.total
            if steps == 3:
                self.assertTrue(validation_started.wait(10), "Episode 1 did not start validation")
            metrics = update_models(batch)
            if steps % 2 == 0:
                snapshots[steps] = {
                    name: {key: value.cpu().clone() for key, value in model.state_dict().items()}
                    for name, model in (("actor", trainer.actor), ("critic", trainer.critic))
                }
            if steps == 3:
                training_resumed.set()
            return metrics

        with TemporaryDirectory() as directory, patch("controllers.sac.Config.output_path",
                lambda config, trial: Config.output_path(config, trial, root=directory)):
            with patch.object(trainer, "evaluate", evaluate_snapshot), patch.object(trainer, "update", update_during_validation):
                result = trainer.train()
            output = Path(result["output_path"])
            training = [json.loads(row) for row in (output / "training.jsonl").read_text().splitlines()]
            self.assertEqual([(row["steps"], row["updates"]) for row in training], [(2, 1)] + [(2, 2)] * 3)
            self.assertTrue(np.isfinite([row["mean_critic_loss"] for row in training]).all())
            checkpoint = torch.load(result["checkpoint"], weights_only=True, map_location="cpu")
            for name, expected in snapshots[checkpoint["evaluation"]["training_steps"]].items():
                torch.testing.assert_close(checkpoint[name], expected, rtol=0, atol=0)
            self.assertEqual(checkpoint["evaluation"], result["evaluation"])
            validations = [json.loads(row) for row in (output / "validation.jsonl").read_text().splitlines()]
            self.assertEqual([row["episode"] for row in validations], [1, 2, 3, 4])
            self.assertEqual(max(validations, key=lambda row: row["score"]), result["evaluation"])
            trainer.actor = SACController.from_checkpoint(trainer.config.bess(trainer.data), checkpoint).actor.to(trainer.device)
            self.assertAlmostEqual(trainer.evaluate()["score"], result["score"])
            annual = evaluate({"checkpoint": result["checkpoint"], "step_limit": 2, "output_root": directory})
            self.assertEqual(annual["year"].tolist(), [2024])
            report, = Path(directory).rglob("evaluation.json")
            metadata = json.loads(report.read_text())
            self.assertEqual(metadata["policy_status"], "trained")
            self.assertEqual(metadata["checkpoint_validation"]["score"], result["score"])
        monthly = result["evaluation"]["monthly"]
        self.assertEqual([(row["month"], row["steps"]) for row in monthly], [(1, 2), (2, 3)])
        self.assertAlmostEqual(result["score"], -sum(row["adjusted_cost"] for row in monthly) / 5)
        self.assertTrue(np.isfinite(result["score"]))
        self.assertEqual(trainer.env.train.idx, 2)
        self.assertEqual(trainer.replay.total, 8)

    def test_parallel_windows_with_soc_actions_and_spaced_updates(self):
        trainer = SACTrainer(eval_workers=2)
        trainer.config.action_mode = "p_soc"
        trainer.config.num_training_episodes = 1
        trainer.config.learning_starts = trainer.config.batch_size = trainer.config.update_every = 2
        trainer.episodes = [trainer.episodes[0] | {"episode_steps": 5}]
        trainer.validation = [window | {"episode_steps": 2} for window in trainer.validation[:2]]
        with TemporaryDirectory() as directory, patch("controllers.sac.Config.output_path",
                lambda config, trial: Config.output_path(config, trial, root=directory)):
            result = trainer.train()
            training = json.loads((Path(result["output_path"]) / "training.jsonl").read_text())
            self.assertEqual(training["updates"], 2)
            monthly = result["evaluation"]["monthly"]
            self.assertEqual([row["steps"] for row in monthly], [2, 2])
            self.assertAlmostEqual(result["score"], -sum(row["adjusted_cost"] for row in monthly) / 4)

    def test_terminal_energy_is_settled_at_closing_buy_and_sell_tariffs(self):
        battery = SimpleNamespace(id="b1", e_cap_kwh=100, soc_init_frac=0.5, eta_charge=0.8,
                                  eta_discharge=0.9, array_charge_kw=[4], array_discharge_kw=[2])
        env = SimpleNamespace(bess_list=[battery], dt=0.25, idx=2, results=SimpleNamespace(diagnostics=[]),
                              grid=SimpleNamespace(prices=[1, 0.2], sell_prices=[1, 0.05]))
        for soc, expected in ((0.4, 2.5), (0.5, 0.0), (0.6, -0.45)):
            battery.soc = soc
            metrics = Evaluator.energy_metrics(env, "b1", 10.0)
            self.assertAlmostEqual(metrics["terminal_energy_cost"], expected)
            self.assertAlmostEqual(metrics["adjusted_cost"], 10.0 + expected)

    def test_trials_reuse_loaded_data_and_plans_without_reading_csv_again(self):
        with patch("controllers.sac.train.load_data", wraps=load_data) as loader:
            trainer = SACTrainer()
            trainer.config.num_training_episodes = 0
            data, episodes, validation = trainer.data, trainer.episodes, trainer.validation
            with TemporaryDirectory() as directory, patch("controllers.sac.Config.output_path",
                    lambda config, trial: Config.output_path(config, trial, root=directory)), \
                    patch("pandas.read_csv", side_effect=AssertionError("CSV reloaded during trial")):
                for seed in (17, 42):
                    trainer.config.seed = seed
                    trainer.train()
                    self.assertIs(trainer.data, data)
                    self.assertIs(trainer.episodes, episodes)
                    self.assertIs(trainer.validation, validation)
                    self.assertIs(trainer.env.train.data["demand"], data["demand"])
                    self.assertIs(trainer.env.eval.data["demand"], data["demand"])
                    self.assertNotIn("start_step", data)
                    self.assertEqual(trainer.env.train.data["start_step"], 0)
                    self.assertEqual(trainer.env.eval.data["start_step"], validation[0]["start_step"])
            loader.assert_called_once()


if __name__ == "__main__":
    unittest.main()
