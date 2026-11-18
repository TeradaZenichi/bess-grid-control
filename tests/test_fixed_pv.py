from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import main
from opendss_environment import Action, EnvironmentConfig, MicrogridEnv, TotalAction
from tools.build_pv_profile import SOURCE, build_profile
from tools.compare_bess import CASE, prepare_case


MIDDAY_STEP = 15988  # June 16, 2023 at 12:00 UTC.


class FixedPVTest(unittest.TestCase):
    def test_profile_matches_demand_and_interpolates_leap_day(self):
        timestamps = pd.read_csv(CASE.parent / "unbalanced_demand.csv", usecols=["timestamp"]).timestamp
        frame = pd.read_csv(CASE.parent / "pv.csv")
        self.assertEqual(len(frame), 70176)
        pd.testing.assert_series_equal(frame.timestamp, timestamps)
        self.assertTrue(frame.pv1_kw.between(0, 200).all())
        expected = build_profile(pd.read_csv(SOURCE), timestamps, 200)
        np.testing.assert_allclose(frame.pv1_kw, expected, rtol=0, atol=1e-12)
        values = frame.set_index(pd.to_datetime(frame.timestamp, utc=True)).pv1_kw
        np.testing.assert_allclose(values.loc["2024-02-29"],
                                   (values.loc["2023-02-28"].to_numpy() + values.loc["2023-03-01"].to_numpy()) / 2,
                                   rtol=0, atol=1e-12)

    def test_fixed_pv_ignores_curtailment_and_reactive_requests(self):
        original = (CASE.parent / "devices.json").read_bytes()
        with TemporaryDirectory() as directory:
            case = prepare_case(CASE, Path(directory) / "case", with_bess=True)
            env = MicrogridEnv(EnvironmentConfig(case, 2, start_step=MIDDAY_STEP))
            self.addCleanup(env.close)
            env.reset()
            pv = env.pv_list[0]
            self.assertEqual(pv.bus, "bus_005")
            self.assertGreater(pv.profile[0], 0)
            env.step(Action(bess={"b1": TotalAction(0)}, pv={"pv1": TotalAction(0, 0.1)}))
            self.assertAlmostEqual(pv.array_kw[0], pv.profile[0])
            self.assertEqual(pv.array_kvar[0], 0)
        self.assertEqual((CASE.parent / "devices.json").read_bytes(), original)

    def test_main_supports_fixed_pv_with_and_without_bess(self):
        with TemporaryDirectory() as directory:
            for with_bess in (False, True):
                case = prepare_case(CASE, Path(directory) / str(with_bess), with_bess)
                with patch.object(main, "CASE", case), patch.object(main, "START_STEP", MIDDAY_STEP), \
                     patch.object(main, "EPISODE_STEPS", 2), patch.object(main, "save_results"):
                    results = main.main()
                pv = results["pv_list"][0]
                np.testing.assert_allclose(pv.array_kw, pv.profile, rtol=0, atol=1e-12)
                np.testing.assert_array_equal(pv.array_kvar, [0, 0])
                self.assertTrue(all(not device.controlled for device in results["evse_list"]))


if __name__ == "__main__":
    unittest.main()
