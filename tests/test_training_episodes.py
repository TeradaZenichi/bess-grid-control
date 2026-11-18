from pathlib import Path
import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from controllers.episodes import monthly_episodes
from controllers.base import Config
from controllers.sac.networks import observation_batch
from controllers.sac.train import SACTrainer
from opendss_environment import MicrogridEnv


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "data/switzerland/5bus/config.json"


class TrainingEpisodesTest(unittest.TestCase):
    def test_monthly_plan_is_seeded_and_excludes_validation_and_test(self):
        plan, validation = monthly_episodes(CASE, 672, 42)
        self.assertEqual((plan, validation), monthly_episodes(CASE, 672, 42))
        self.assertNotEqual(plan, monthly_episodes(CASE, 672, 43)[0])
        self.assertEqual(sorted(row["month"] for row in plan), list(range(1, 13)))
        for episode in plan:
            start = pd.Timestamp(episode["start"])
            end = pd.Timestamp(episode["end_exclusive"])
            dates = pd.date_range(start=start, periods=672, freq="15min").tz_convert("Europe/Zurich")
            self.assertTrue((dates.year == 2023).all())
            self.assertFalse(((dates.day >= 15) & (dates.day < 22)).any())
            self.assertEqual(start.hour, 0)
            self.assertEqual(end - start, pd.Timedelta(hours=168))
            self.assertEqual(episode["end_step_exclusive"] - episode["start_step"], 672)
        self.assertEqual(len(validation), 12)
        for block in validation:
            self.assertEqual(pd.Timestamp(block["start"]).day, 15)
            self.assertEqual(pd.Timestamp(block["end_exclusive"]).day, 22)

    def test_train_initializes_selected_window_and_networks_without_stepping(self):
        with TemporaryDirectory() as directory, patch("controllers.sac.Config.output_path",
                lambda config, trial: Config.output_path(config, trial, root=directory)), \
                patch.object(MicrogridEnv, "step") as step:
            with patch.object(MicrogridEnv, "close", autospec=True, side_effect=MicrogridEnv.close) as close:
                for architecture in ("mlp", "gru"):
                    trainer = SACTrainer()
                    trainer.config.architecture = architecture
                    trainer.config.episode_steps = 96
                    trainer.config.history_steps = 4
                    result = trainer.train()
                    self.assertEqual(result["status"], "episodes_ready")
                    self.assertIsNone(result["score"])
                    self.assertEqual(pd.Timestamp(result["initial_timestamp"]), pd.Timestamp(result["episodes"][0]["start"]))
                    self.assertEqual(result["input_size"], 132 if architecture == "mlp" else 32)
                    self.assertEqual(result["action_size"], 6)
                    output = Path(result["output_path"])
                    self.assertEqual(json.loads((output / "initialization.json").read_text()), result)
                    self.assertEqual(output.parts[-3:], (architecture, result["trial"], "seed_42"))
                    state = observation_batch([trainer.initial_state])
                    action, _ = trainer.actor(state)
                    self.assertEqual(tuple(action.shape), (1, 6))
                    self.assertEqual(tuple(trainer.critic.q1(state, action).shape), (1, 1))
                    self.assertEqual(tuple(trainer.critic.q2(state, action).shape), (1, 1))
                    mask = trainer.initial_state[-4:] if architecture == "mlp" else trainer.initial_state["mask"]
                    self.assertEqual(mask.sum(), 1)
                self.assertEqual(close.call_count, 2)
            step.assert_not_called()


if __name__ == "__main__":
    unittest.main()
