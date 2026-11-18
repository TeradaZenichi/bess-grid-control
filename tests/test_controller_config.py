from pathlib import Path
import unittest

from controllers.base import Config, ROOT
from controllers.sac.train import SACTrainer
from controllers.sac import Config as SACConfig


class ControllerConfigTest(unittest.TestCase):
    def test_controller_defaults_and_trial_overrides_are_independent(self):
        rule = Config.controller("rule_based")
        self.assertEqual(rule.case_path, str(ROOT / "data/switzerland/5bus/config.json"))
        config = SACConfig(
            {"architecture": "gru", "reward_term": "feeder_cost", "seed": 17,
             "actor_lr": 0.001, "extra_features": ["branches"]}, {"history_steps": 4})
        trainer = SACTrainer(config)
        self.assertIs(trainer.config, config)
        self.assertEqual(rule.case_path, trainer.config.case_path)
        self.assertEqual(rule.mode, trainer.config.mode)
        self.assertEqual((config.controller_type, config.architecture, config.history_steps), ("sac", "gru", 4))
        self.assertEqual((config.reward_term, config.seed, config.actor_lr), ("feeder_cost", 17, 0.001))
        config.extra_features.append("grid")
        self.assertEqual(SACTrainer().config.architecture, "mlp")
        self.assertEqual(SACTrainer().config.reward_term, "node_cost")
        self.assertEqual(Config.controller().extra_features, ["grid", "branches"])

    def test_architecture_changes_refresh_defaults_and_flatten(self):
        config = SACConfig()
        self.assertTrue(config.flatten)
        config.architecture = "gru"
        config.history_steps = 4
        self.assertFalse(config.flatten)
        self.assertEqual(config.hidden_size, 64)
        self.assertEqual(config.history_steps, 4)
        config.architecture = "mlp"
        self.assertTrue(config.flatten)
        self.assertEqual(config.history_steps, 8)

    def test_output_distinguishes_controller_objective_architecture_trial_and_seed(self):
        config = Config.controller("sac")
        expected = ROOT / "results/controllers/switzerland_5bus/sac/economic/node_cost/mlp/trial_003/seed_42"
        self.assertEqual(config.output_path("trial_003"), expected)
        for settings in ({"mode": "voltage"}, {"reward_term": "feeder_cost"},
                         {"architecture": "gru"}, {"seed": 17}, {"case_path": "data/other/config.json"}):
            self.assertNotEqual(Config.controller("sac", settings).output_path("trial_003"), expected)
        self.assertNotEqual(config.output_path("trial_004"), expected)
        rule = Config.controller("rule_based")
        self.assertEqual(rule.output_path("simulation", root="custom"),
                         Path("custom/switzerland_5bus/rule_based/economic/node_cost/simulation/seed_42"))


if __name__ == "__main__":
    unittest.main()
