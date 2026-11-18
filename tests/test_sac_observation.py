from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from controllers.normalization import ObservationScaling
from controllers.sac.controller import SACController
from controllers.sac.observation import FEATURE_NAMES, LocalObservation
from opendss_environment import Action, EnvironmentConfig, MicrogridEnv, TotalAction


ROOT = Path(__file__).resolve().parents[1]
BESS = json.loads((ROOT / "data/switzerland/5bus/devices.json").read_text())["bess"][0]
SCALING = ObservationScaling(
    voltage_deviation_ref_pu=0.05, vuf_ref_pct=2.0, price_ref_per_kwh=0.2,
    s_base_kva=1000.0, bus=BESS["bus"], v_base_ln_kv=0.22,
    meter_element="Transformer.t4", meter_terminal=2, meter_s_kva=250.0, meter_phases=3,
)


def sample(step=0):
    return {
        "units": "pu", "controller": {"bus": BESS["bus"]}, "dt_h": 0.25,
        "timestamp": (datetime(2023, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * step)).isoformat(),
        "bus": {"v_before_pu": dict.fromkeys("abc", 1),
                "angle_before_deg": dict(zip("abc", (0, -120, 120)))},
        "components": {"bess": {"b1": {"bus": BESS["bus"], "phases": list("abc"),
            "soc_before_frac": 0.5, "previous_p_pu": dict.fromkeys("abc", 0.01),
            "previous_q_pu": dict.fromkeys("abc", 0.005)}}},
        "branches": [{"bus": BESS["bus"], "element": "Transformer.t4", "terminal": 2,
                      "phase": p, "p_flow_pu": -0.02, "q_flow_pu": -0.01,
                      "current_a": 90, "loading_pct": 24} for p in "cba"],
        "grid": {"buy_price_per_kwh": 0.2, "sell_price_per_kwh": 0.04},
    }


class ObservationSAC(SACController):
    def act(self, observation, objective=None):
        raise NotImplementedError("Observation-only test policy")


class SACObservationTest(unittest.TestCase):
    def make_encoder(self, **kwargs):
        return LocalObservation(BESS, SCALING, **kwargs)

    def test_named_features_meter_signs_units_and_padding(self):
        encoder = self.make_encoder()
        result = encoder.encode(sample())
        self.assertEqual(len(FEATURE_NAMES), 32)
        self.assertEqual(result["history"].shape, (8, 32))
        self.assertEqual(result["mask"].tolist(), [False] * 7 + [True])
        np.testing.assert_array_equal(result["history"][:-1], 0)
        values = dict(zip(FEATURE_NAMES, result["history"][-1]))
        self.assertAlmostEqual(values["p_import_a_scaled"], 0.24)
        self.assertAlmostEqual(values["q_import_b_scaled"], 0.12)
        self.assertAlmostEqual(values["previous_q_c_scaled"], 0.3)
        self.assertAlmostEqual(values["v_a_scaled"], 0)
        self.assertAlmostEqual(values["buy_price_scaled"], 1)
        self.assertAlmostEqual(values["sell_price_scaled"], 0.2)
        self.assertAlmostEqual(values["loading_c_frac"], 0.24)
        self.assertAlmostEqual(values["vuf_scaled"], 0, places=6)
        self.assertAlmostEqual(values["angle_b_minus_a_sin"], -np.sqrt(3) / 2)
        flat = encoder.encode(sample(), flatten=True)
        self.assertEqual(flat.shape, (264,))
        self.assertEqual(flat.dtype, np.float32)
        np.testing.assert_array_equal(flat[-8:], result["mask"])

    def test_history_is_bounded_copied_and_repeated_calls_do_not_append(self):
        encoder = self.make_encoder(history_steps=3)
        for step in range(5):
            frame = sample(step)
            frame["components"]["bess"]["b1"]["soc_before_frac"] = step / 10
            result = encoder.encode(frame)
        col = FEATURE_NAMES.index("soc_frac")
        np.testing.assert_allclose(result["history"][:, col], [0.2, 0.3, 0.4])
        result["history"][:] = 999
        repeated = encoder.encode(frame)
        np.testing.assert_allclose(repeated["history"][:, col], [0.2, 0.3, 0.4])
        encoder.reset()
        self.assertEqual(encoder.encode(sample())["mask"].sum(), 1)

    def test_time_discontinuities_reset_history(self):
        encoder = self.make_encoder()
        encoder.encode(sample())
        self.assertEqual(encoder.encode(sample(3))["mask"].sum(), 1)
        self.assertEqual(encoder.encode(sample(1))["mask"].sum(), 1)
        self.assertEqual(encoder.encode(sample(2))["mask"].sum(), 2)

    def test_scaling_preserves_signs_violations_and_physical_observations(self):
        frame = sample()
        frame["bus"]["v_before_pu"] = dict(zip("abc", (0.94, 1, 1.08)))
        frame["branches"][0].update(p_flow_pu=0.02, loading_pct=150)
        frame["grid"]["buy_price_per_kwh"] = -0.1
        original = deepcopy(frame)
        result = self.make_encoder().encode(frame)
        values = dict(zip(FEATURE_NAMES, result["history"][-1]))
        np.testing.assert_allclose([values[f"v_{p}_scaled"] for p in "abc"], [-1.2, 0, 1.6])
        self.assertAlmostEqual(values["p_import_c_scaled"], -0.24)
        self.assertEqual(values["loading_c_frac"], 1.5)
        self.assertEqual(values["buy_price_scaled"], -0.5)
        self.assertEqual(frame, original)
        np.testing.assert_array_equal(result["history"][:-1], 0)

    def test_scaling_is_independent_of_network_base_and_json_restorable(self):
        restored = ObservationScaling(**json.loads(json.dumps(vars(SCALING))))
        restored.s_base_kva = 2000
        frame = sample()
        for flow in frame["branches"]:
            flow["p_flow_pu"] /= 2
            flow["q_flow_pu"] /= 2
        state = frame["components"]["bess"]["b1"]
        for field in ("previous_p_pu", "previous_q_pu"):
            state[field] = {p: value / 2 for p, value in state[field].items()}
        expected = self.make_encoder().encode(sample(), flatten=True)
        actual = LocalObservation(BESS, restored).encode(frame, flatten=True)
        np.testing.assert_array_equal(actual, expected)

    def test_network_references_read_ratings_without_solving_or_advancing(self):
        env = MicrogridEnv(EnvironmentConfig(ROOT / "data/switzerland/5bus/config.json", 2))
        self.addCleanup(env.close)
        env.reset(seed=42)
        dss = env.dss._dss
        active = (dss.CktElement.Name(), dss.Bus.Name(), dss.Transformers.Name(), dss.Transformers.Wdg())
        original = env.observe()
        with patch.object(env.dss.solution, "solve") as solve:
            t2 = ObservationScaling.from_environment(env, "bus_002_lv")
            t4 = ObservationScaling.from_environment(env, "bus_004_lv")
            line = ObservationScaling.from_environment(env, "bus_004", "Line.line_003_004")
            solve.assert_not_called()
        self.assertEqual((t2.meter_s_kva, t4.meter_s_kva), (400, 250))
        self.assertAlmostEqual(line.meter_s_kva, 600)
        self.assertEqual(t4.meter_terminal, 2)
        self.assertEqual(t4.v_base_ln_kv, 0.22)
        self.assertEqual(t4.voltage_deviation_ref_pu, 0.05)
        self.assertEqual(t4.vuf_ref_pct, 2)
        self.assertEqual(t4.price_ref_per_kwh, 0.2)
        self.assertEqual(env.idx, 0)
        self.assertEqual(env.observe(), original)
        self.assertEqual((dss.CktElement.Name(), dss.Bus.Name(), dss.Transformers.Name(), dss.Transformers.Wdg()), active)

    def test_relative_angles_and_vuf_ignore_reference_rotation_and_remote_data(self):
        frame = sample()
        a = np.exp(2j * np.pi / 3)
        voltages = [1.02, a**2 + 0.02 * a, a + 0.02 * a**2]
        frame["bus"]["v_before_pu"] = dict(zip("abc", np.abs(voltages)))
        frame["bus"]["angle_before_deg"] = dict(zip("abc", np.rad2deg(np.angle(voltages))))
        encoder = self.make_encoder()
        expected = encoder.encode(frame)["history"][-1].copy()
        self.assertAlmostEqual(expected[FEATURE_NAMES.index("vuf_scaled")], 1, places=5)
        changed = deepcopy(frame)
        changed["bus"]["angle_before_deg"] = {p: v + 87 for p, v in frame["bus"]["angle_before_deg"].items()}
        changed["pv"] = {"remote": {"available_pu": 1e6}}
        changed["bus"]["p_load_pu"] = dict.fromkeys("abc", 1e6)
        changed["grid"]["global_power_pu"] = 1e6
        np.testing.assert_allclose(encoder.encode(changed)["history"][-1], expected, atol=1e-6)

    def test_calendar_uses_local_time_without_creating_a_dst_gap(self):
        encoder = self.make_encoder()
        frame = sample()
        frame["timestamp"] = "2023-03-26T00:45:00+00:00"
        encoder.encode(frame)
        frame["timestamp"] = "2023-03-26T01:00:00+00:00"  # 03:00 in Zurich.
        result = encoder.encode(frame)
        self.assertEqual(result["mask"].sum(), 2)
        self.assertAlmostEqual(result["history"][-1, FEATURE_NAMES.index("day_sin")], np.sin(2 * np.pi * 3 / 24))

    def test_controller_architectures_share_inputs_and_reset_history(self):
        mlp = ObservationSAC(BESS, scaling=SCALING)
        gru = ObservationSAC(BESS, {"architecture": "gru"}, scaling=SCALING)
        self.assertEqual(mlp.extra_features, ("grid", "branches"))
        flat, sequence = mlp.observe(sample()), gru.observe(sample())
        np.testing.assert_array_equal(flat[:256], sequence["history"].ravel())
        np.testing.assert_array_equal(flat[256:], sequence["mask"])
        mlp.observe(sample(1))
        mlp.reset()
        self.assertEqual(mlp.observe(sample())[-8:].sum(), 1)

    def test_live_encoder_sees_executed_action_and_lv_meter_not_requested_action(self):
        env = MicrogridEnv(EnvironmentConfig(ROOT / "data/switzerland/5bus/config.json", 2))
        self.addCleanup(env.close)
        before, _ = env.reset(seed=42)
        scaling = ObservationScaling.from_environment(env, BESS["bus"])
        policy = ObservationSAC(BESS, scaling=scaling)
        local = env.observe_controller(policy.bus, {"bess": [policy.id]}, observation=before,
                                       extra_features=policy.extra_features)
        policy.observe(local)
        after, _, _, _, _ = env.step(Action(bess={"b1": TotalAction(0.5)}, pv={"pv1": TotalAction(0)}))
        local = env.observe_controller(policy.bus, {"bess": [policy.id]}, observation=after,
                                       extra_features=policy.extra_features)
        vector = policy.observe(local)[:256].reshape(8, 32)[-1]
        values = dict(zip(FEATURE_NAMES, vector))
        self.assertAlmostEqual(sum(values[f"previous_p_{p}_scaled"] for p in "abc"), 2.4, places=6)
        self.assertAlmostEqual(values["p_import_a_scaled"], -next(f["p_flow_pu"] for f in local["branches"] if f["phase"] == "a") * 12, places=6)
        env.dss.circuit.set_active_bus(policy.bus)
        sequence = env.dss._dss.Bus.SeqVoltages()
        self.assertAlmostEqual(values["vuf_scaled"], 50 * sequence[2] / sequence[1], places=5)


if __name__ == "__main__":
    unittest.main()
