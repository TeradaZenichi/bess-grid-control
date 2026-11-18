from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from opendss_environment import Action, EnvironmentConfig, EVSEAction, MicrogridEnv, PhaseAction
from tools.benchmark_environment import FullCopyEnv


ROOT = Path(__file__).resolve().parents[1] / "opendss_environment"
FIXTURES = ROOT / "tests/fixtures"


class StepEquivalenceTest(unittest.TestCase):
    def test_checkpoint_matches_full_copy_for_bess_pv_and_v2g(self):
        phase_action = Action(
            bess={"b1": PhaseAction({"a": 0.03, "b": -0.02, "c": 0.01},
                                    {"a": 0.01, "b": 0.02, "c": -0.01})},
            pv={"pv1": PhaseAction(dict.fromkeys("abc", 0.02), dict.fromkeys("abc", 0.01))},
        )
        cases = ((ROOT / "examples/case5", 24, None),
                 (FIXTURES / "three_phase_volt_control/config_per_phase.json", 2, phase_action),
                 (FIXTURES / "three_phase_volt_control/config_delta.json", 2, phase_action),
                 (FIXTURES / "evse_case", 4, Action(evse={"c1": EVSEAction(-0.08, 0.06)})))
        for case, steps, action in cases:
            with self.subTest(case=case):
                traces = []
                for env_type in (FullCopyEnv, MicrogridEnv):
                    env = env_type(EnvironmentConfig(case, steps))
                    self.addCleanup(env.close)
                    initial = env.reset(seed=42)
                    trace = [deepcopy(initial)]
                    for _ in range(steps):
                        trace.append(deepcopy(env.step(action)))
                    trace.append(deepcopy(vars(env.results)))
                    traces.append(trace)
                np.testing.assert_equal(*traces)


if __name__ == "__main__":
    unittest.main()
