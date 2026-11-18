import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from controllers.config import load_config
from controllers.rule_based.controller import RuleBasedController
from tools.evaluate_controller import CONFIG, evaluate, year_windows


class ControllerEvaluationTest(unittest.TestCase):
    def test_local_year_windows_include_leap_year_and_have_no_overlap(self):
        config = load_config(CONFIG)
        self.assertEqual(year_windows(config), [
            ("train_eval", 2023, 0, 35040, 35040),
            ("test", 2024, 35040, 35136, 35136),
        ])
        config.step_limit = 4
        self.assertEqual(year_windows(config, groups=("test",)), [("test", 2024, 35040, 4, 35136)])

    def test_evaluation_calls_controller_and_saves_separate_period_results(self):
        with TemporaryDirectory() as directory:
            with patch.object(RuleBasedController, "act", autospec=True, side_effect=RuleBasedController.act) as act:
                frame = evaluate({"output_path": directory, "step_limit": 4})
                self.assertEqual(act.call_count, 8)
            self.assertEqual(frame.group.tolist(), ["train_eval", "test"])
            self.assertEqual(frame.steps.tolist(), [4, 4])
            self.assertFalse(frame.complete_year.any())
            reports = list(Path(directory).rglob("evaluation.json"))
            self.assertEqual(len(reports), 2)
            for report in reports:
                metadata = json.loads(report.read_text())
                folder = report.parent
                year = metadata["metrics"]["year"]
                buses = pd.read_csv(folder / "sim_buses_timeseries.csv.gz")
                devices = pd.read_csv(folder / "sim_devices_timeseries.csv.gz")
                nodes = pd.read_csv(folder / "node_costs.csv.gz")
                branches = pd.read_csv(folder / "branches.csv.gz")
                self.assertTrue(metadata["reset_each_year"])
                self.assertEqual(metadata["policy_status"], "rule_based")
                self.assertEqual(len(buses), 4)
                pd.testing.assert_series_equal(buses.timestamp, devices.timestamp)
                pd.testing.assert_series_equal(buses.timestamp, nodes.timestamp)
                local_start = pd.Timestamp(buses.timestamp.iloc[0]).tz_convert("Europe/Zurich")
                self.assertEqual(local_start, pd.Timestamp(f"{year}-01-01", tz="Europe/Zurich"))
                self.assertAlmostEqual(buses.grid_cost.sum(), metadata["metrics"]["feeder_cost"])
                self.assertAlmostEqual(nodes.b1.sum(), metadata["metrics"]["node_cost_b1"])
                self.assertEqual(branches.timestamp.nunique(), 4)
                self.assertIn("bess_b1_soc_frac", devices.columns)
                self.assertIn("bess_b1_a_p_net_kw", devices.columns)


if __name__ == "__main__":
    unittest.main()
