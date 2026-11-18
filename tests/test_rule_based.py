from copy import deepcopy
import cmath
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from controllers import Controller
from controllers.rule_based.controller import RuleBasedController
from opendss_environment import Action, EnvironmentConfig, MicrogridEnv, PhaseAction, TotalAction
from opendss_environment.elements import BESS as Battery, BESSConfig


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "opendss_environment/tests/fixtures/three_phase_volt_control"
BESS = json.loads((FIXTURE / "devices_per_phase.json").read_text())["bess"][0]


def observation(soc=0.5, price=0.2, voltages=(1.0, 1.0, 1.0), imported=0.1, previous=0.0):
    phases = ("a", "b", "c")
    return {
        "units": "pu", "base": {"s_base_kva": 100.0}, "dt_h": 0.25,
        "phase_order": list(phases), "controller": {"bus": "load"},
        "bus": {"v_before_pu": dict(zip(phases, voltages)),
                "angle_before_deg": dict(zip(phases, (0, -120, 120)))},
        "components": {"bess": {"b1": {
            "bus": "load", "phases": list(phases), "soc_before_frac": soc,
            "previous_p_pu": dict.fromkeys(phases, previous / 3),
            "previous_q_pu": dict.fromkeys(phases, 0.0),
        }}},
        "grid": {"buy_price_per_kwh": price, "sell_price_per_kwh": 0.2 * price},
        "branches": [{"bus": "load", "p_flow_pu": -imported}],
    }


class RuleBasedTest(unittest.TestCase):
    def test_public_controller_is_concrete_and_uses_its_own_configuration(self):
        controller = Controller(BESS, {"mode": "voltage"})
        self.assertIsInstance(controller, RuleBasedController)
        self.assertEqual(controller.config.mode, "voltage")
        self.assertEqual(RuleBasedController(BESS).config.mode, "economic")

    def test_economic_charging_idle_and_local_discharge(self):
        controller = Controller(BESS)
        self.assertAlmostEqual(sum(controller.act(observation(price=0.15)).p_pu.values()), 0.15)
        self.assertAlmostEqual(sum(controller.act(observation(price=0.2)).p_pu.values()), 0)
        self.assertAlmostEqual(sum(controller.act(observation(price=0.25)).p_pu.values()), -0.1)
        self.assertEqual(sum(controller.act(observation(price=0.25, imported=-0.02)).p_pu.values()), 0)

    def test_economic_charge_stops_when_any_phase_has_low_voltage(self):
        controller = Controller(BESS, {"max_ramp_frac": 0.01})
        action = controller.act(observation(price=0.15, voltages=(0.99, 0.94, 0.93), imported=0.2, previous=0.1))
        self.assertTrue(all(value <= 0 for value in action.p_pu.values()))
        self.assertLess(sum(action.p_pu.values()), 0)
        self.assertEqual(controller.last_decision["charge_limit_pu"], 0)

    def test_price_bands_use_only_observed_prices_and_cover_conversion_losses(self):
        controller = Controller(BESS, {"price_window_hours": 1})
        for price in (0.15, 0.15, 0.19):
            controller.act(observation(price=price))
        self.assertEqual(controller.last_decision["price_high"], 0.23)
        action = controller.act(observation(price=0.19))
        self.assertEqual(controller.last_decision["price_low"], 0.15)
        self.assertEqual(controller.last_decision["price_high"], 0.19)
        self.assertLess(sum(action.p_pu.values()), 0)
        expensive = Controller(BESS | {"eta_charge": 0.8, "eta_discharge": 0.8}, {"price_window_hours": 1})
        for price in (0.15, 0.15, 0.19, 0.19):
            action = expensive.act(observation(price=price))
        self.assertEqual(sum(action.p_pu.values()), 0)
        controller.reset()
        self.assertFalse(controller.price_history)

    def test_price_history_warms_up_again_after_an_observation_gap(self):
        controller = Controller(BESS, {"price_window_hours": 1})
        for index, price in enumerate((0.15, 0.15, 0.19, 0.19)):
            state = observation(price=price)
            state["timestamp"] = f"2023-01-01T00:{15 * index:02d}:00+00:00"
            controller.act(state)
        self.assertEqual(controller.last_decision["price_high"], 0.19)
        state["timestamp"] = "2023-01-03T00:00:00+00:00"
        controller.act(state)
        self.assertEqual(controller.last_decision["price_high"], 0.23)

    def test_recharge_has_hysteresis_and_waits_for_voltage_headroom(self):
        for mode in ("voltage", "balance"):
            controller = Controller(BESS, {"mode": mode})
            self.assertGreater(sum(controller.act(observation(soc=0.39)).p_pu.values()), 0)
            self.assertGreater(sum(controller.act(observation(soc=0.5)).p_pu.values()), 0)
            self.assertEqual(sum(controller.act(observation(soc=0.6)).p_pu.values()), 0)
            action = controller.act(observation(soc=0.2, voltages=(0.96, 0.96, 0.96)))
            self.assertEqual(sum(action.p_pu.values()), 0)
            self.assertTrue(controller.last_decision["recharge_active"])
            self.assertGreater(sum(controller.act(observation(soc=0.2)).p_pu.values()), 0)

    def test_recharge_reaches_target_and_unlatches_despite_reactive_losses(self):
        parameters = BESS | {"q_loss_rated_kw": 0.5}
        controller = Controller(parameters, {"mode": "voltage"})
        controller.act(observation(soc=0.39))
        battery = Battery(BESSConfig(**parameters))
        battery.soc = 0.599999
        state = observation(soc=battery.soc, voltages=(0.99, 0.98, 0.99))
        action = controller.act(state)
        battery.operate_phases({p: value * 100 for p, value in action.p_pu.items()},
                               {p: value * 100 for p, value in action.q_pu.items()}, 0.25)
        self.assertAlmostEqual(battery.soc, 0.6)
        controller.act(observation(soc=battery.soc))
        self.assertFalse(controller.recharging)

    def test_voltage_soft_band_uses_reactive_power_without_active_discharge(self):
        controller = Controller(BESS, {"mode": "voltage"})
        action = controller.act(observation(soc=0.5, voltages=(0.96, 0.96, 0.96)))
        self.assertEqual(sum(action.p_pu.values()), 0)
        self.assertGreater(sum(action.q_pu.values()), 0)

    def test_support_modes_remain_available_over_repeated_daily_events(self):
        parameters = BESS | {"q_loss_rated_kw": 0.5}
        for mode in ("voltage", "balance"):
            controller = Controller(parameters, {"mode": mode})
            battery = Battery(BESSConfig(**parameters))
            minimum = battery.soc
            for step in range(30 * 96):
                voltage = 0.94 if 72 <= step % 96 < 84 else 1.0
                action = controller.act(observation(soc=battery.soc, voltages=(voltage,) * 3))
                battery.operate_phases({phase: value * 100 for phase, value in action.p_pu.items()},
                                       {phase: value * 100 for phase, value in action.q_pu.items()}, 0.25)
                minimum = min(minimum, battery.soc)
            self.assertGreater(minimum, 0.25)
            self.assertGreater(sum(battery.array_charge_kw) * 0.25, 80)
            self.assertLessEqual(max(battery.array_soc), 0.6 + 1e-10)

    def test_export_cap_removes_the_previous_bess_exchange_and_overrides_a_ramp(self):
        controller = Controller(BESS, {"max_ramp_frac": 0.1})
        action = controller.act(observation(price=0.25, imported=0.02, previous=-0.03))
        self.assertAlmostEqual(sum(action.p_pu.values()), -0.05)
        action = controller.act(observation(price=0.25, imported=-0.06, previous=-0.06))
        self.assertAlmostEqual(sum(action.p_pu.values()), 0)

    def test_reserve_and_charge_target_limit_energy_without_changing_observation(self):
        controller = Controller(BESS)
        state = observation(soc=0.799, price=0.15)
        before = deepcopy(state)
        action = controller.act(state)
        self.assertAlmostEqual(sum(action.p_pu.values()), 0.004)
        self.assertEqual(state, before)
        action = controller.act(observation(soc=0.401, price=0.25))
        self.assertAlmostEqual(sum(action.p_pu.values()), -0.004)
        self.assertEqual(sum(controller.act(observation(soc=0.3, price=0.25)).p_pu.values()), 0)
        support = controller.act(observation(soc=0.3, price=0.25, voltages=(0.94, 0.94, 0.94)))
        self.assertLess(sum(support.p_pu.values()), 0)
        action = controller.act(observation(soc=0.15), {"mode": "tracking", "p_reference_pu": -0.1, "q_reference_pu": 0})
        self.assertEqual(sum(action.p_pu.values()), 0)

    def test_voltage_deadband_signs_and_active_assistance(self):
        controller = Controller(BESS, {"mode": "voltage"})
        action = controller.act(observation(voltages=(0.97, 1.0, 1.03)))
        self.assertGreater(action.q_pu["a"], 0)
        self.assertEqual(action.q_pu["b"], 0)
        self.assertLess(action.q_pu["c"], 0)
        self.assertAlmostEqual(sum(abs(value) for value in action.p_pu.values()), 0)
        action = controller.act(observation(voltages=(0.94, 1.0, 1.06)))
        self.assertLess(action.p_pu["a"], 0)
        self.assertGreater(action.p_pu["c"], 0)

    def test_balance_uses_relative_phase_voltage_and_requires_independent_phases(self):
        controller = Controller(BESS)
        action = controller.act(observation(voltages=(0.98, 1.0, 1.02)), {"mode": "balance"})
        self.assertAlmostEqual(sum(action.q_pu.values()), 0)
        self.assertGreater(action.q_pu["a"], 0)
        self.assertLess(action.q_pu["c"], 0)
        self.assertEqual(sum(action.p_pu.values()), 0)
        aggregate = Controller(BESS | {"dispatch_mode": "aggregate"})
        with self.assertRaisesRegex(ValueError, "per-phase"):
            aggregate.act(observation(), {"mode": "balance"})

    def test_balance_responds_to_angles_when_voltage_magnitudes_are_equal(self):
        state = observation()
        state["bus"]["angle_before_deg"]["b"] = -123
        controller = Controller(BESS, {"mode": "balance"})
        action = controller.act(state)
        self.assertGreater(max(abs(q) for q in action.q_pu.values()), 1e-6)
        self.assertAlmostEqual(sum(action.q_pu.values()), 0)

    def test_balance_reduces_vuf_in_opendss_without_voltage_guard_intervention(self):
        env = MicrogridEnv(EnvironmentConfig(FIXTURE / "config_per_phase.json", 2))
        self.addCleanup(env.close)
        values = []
        for controlled in (False, True):
            global_state, _ = env.reset()
            command = TotalAction(0)
            if controlled:
                controller = Controller(BESS, {"mode": "balance", "v_min_pu": 0.9,
                                               "v_low_pu": 0.9, "v_high_pu": 1.15, "v_max_pu": 1.15})
                local = env.observe_controller("load", {"bess": ["b1"]}, observation=global_state)
                command = controller.act(local)
            _, _, _, _, info = env.step(Action(bess={"b1": command},
                pv={"pv1": PhaseAction(global_state["pv"]["pv1"]["available_pu"])}))
            v = {p: cmath.rect(info["bus_voltages_pu"]["load"][p],
                               math.radians(info["bus_angles_deg"]["load"][p])) for p in "abc"}
            a = cmath.exp(2j * math.pi / 3)
            values.append(abs((v["a"] + a*a*v["b"] + a*v["c"]) / (v["a"] + a*v["b"] + a*a*v["c"])))
        self.assertLess(values[1], values[0])

    def test_tracking_references_are_total_pu_or_explicit_phase_commands(self):
        controller = Controller(BESS)
        action = controller.act(observation(), {"mode": "tracking", "p_reference_pu": 0.06, "q_reference_pu": -0.03})
        self.assertAlmostEqual(sum(action.p_pu.values()), 0.06)
        self.assertAlmostEqual(sum(action.q_pu.values()), -0.03)
        reference = {"a": 0.01, "b": -0.01, "c": 0.0}
        action = controller.act(observation(), {"mode": "tracking", "p_reference_pu": reference, "q_reference_pu": 0})
        self.assertEqual(action.p_pu, reference)
        with self.assertRaisesRegex(ValueError, "explicit"):
            controller.act(observation(), {"mode": "tracking"})

    def test_capability_and_reactive_loss_budget_are_enforced(self):
        controller = Controller(BESS | {"q_loss_rated_kw": 0.5})
        action = controller.act(observation(soc=0.200001), {
            "mode": "tracking", "p_reference_pu": -1, "q_reference_pu": 1,
        })
        energy_used = 0.0
        for phase in "abc":
            self.assertLessEqual(action.p_pu[phase] ** 2 + action.q_pu[phase] ** 2, 0.1 ** 2 + 1e-12)
            energy_used += max(-action.p_pu[phase], 0) * 100 * 0.25
            energy_used += 0.5 / 3 * (action.q_pu[phase] / 0.1) ** 2 * 0.25
        self.assertLessEqual(energy_used, 0.0001 + 1e-10)

    def test_voltage_override_precedes_tracking_and_ramp_but_not_physical_limits(self):
        controller = Controller(BESS, {"max_ramp_frac": 0.01})
        action = controller.act(observation(voltages=(0.94, 1.0, 1.0)), {
            "mode": "tracking", "p_reference_pu": 0.1, "q_reference_pu": -0.03,
        })
        self.assertLess(action.p_pu["a"], 0)
        self.assertGreater(action.q_pu["a"], 0)
        self.assertGreater(controller.last_requested_action.p_pu["a"], 0)
        controller.reset()
        self.assertIsNone(controller.last_requested_action)
        self.assertIsNone(controller.last_action)

    def test_aggregate_commands_and_conflicting_voltage_extremes(self):
        controller = Controller(BESS | {"dispatch_mode": "aggregate"})
        action = controller.act(observation(price=0.15))
        self.assertIsInstance(action, TotalAction)
        self.assertAlmostEqual(action.p_pu, 0.15)
        action = controller.act(observation(voltages=(0.94, 1.0, 1.06)), {
            "mode": "tracking", "p_reference_pu": 0.03, "q_reference_pu": 0,
        })
        self.assertEqual((action.p_pu, action.q_pu), (0, 0))
        with self.assertRaisesRegex(ValueError, "Unequal"):
            controller.act(observation(), {"mode": "tracking", "p_reference_pu": {"a": 0.01, "b": 0, "c": 0}, "q_reference_pu": 0})

    def test_invalid_settings_and_nonfinite_observations_are_rejected(self):
        for settings in ({"mode": "unknown"}, {"q_gain": float("nan")}, {"price_low": 1}, {"soc_reserve": 0.9}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                Controller(BESS, settings)
        state = observation()
        state["bus"]["v_before_pu"]["a"] = float("nan")
        with self.assertRaises(ValueError):
            Controller(BESS).act(state)

    def test_all_modes_execute_in_opendss_without_extra_device_clipping(self):
        env = MicrogridEnv(EnvironmentConfig(FIXTURE / "config_per_phase.json", episode_steps=2))
        self.addCleanup(env.close)
        for mode in ("economic", "voltage", "balance", "tracking"):
            global_state, _ = env.reset()
            controller = Controller(BESS, {"mode": mode})
            local = env.observe_controller("load", {"bess": ["b1"]}, extra_features=controller.extra_features, observation=global_state)
            command = controller.act(local, {"mode": mode, "p_reference_pu": 0.06, "q_reference_pu": 0.03})
            _, _, _, _, info = env.step(Action(bess={"b1": command}, pv={"pv1": PhaseAction(global_state["pv"]["pv1"]["available_pu"])}))
            measured = info["device_measurements"]["bess"]["b1"]
            for phase in "abc":
                self.assertAlmostEqual(command.p_pu[phase], measured["p_net_pu"][phase], places=7)
                self.assertAlmostEqual(command.q_pu[phase], measured["q_injection_pu"][phase], places=7)

    def test_main_runs_a_bess_with_asap_evs_without_changing_source_data(self):
        import main

        ev_case = ROOT / "opendss_environment/tests/fixtures/evse_case"
        config = json.loads((ev_case / "config.json").read_text())
        devices = json.loads((ev_case / "devices.json").read_text())
        devices["bess"] = [deepcopy(BESS)]
        for device in devices["evse"]:
            device.update(control_mode="asap", v2g=False)
        config["network"]["master"] = str((ev_case / config["network"]["master"]).resolve())
        for key, filename in config["files"].items():
            config["files"][key] = str((ev_case / filename).resolve())
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            config["files"]["devices"] = str(folder / "devices.json")
            (folder / "devices.json").write_text(json.dumps(devices))
            case = folder / "config.json"
            case.write_text(json.dumps(config))
            with patch.object(main, "CASE", case), patch.object(main, "EPISODE_STEPS", 2), patch.object(main, "save_results"):
                results = main.main()
        self.assertEqual(len(results["bess_list"][0].array_kw), 2)
        self.assertTrue(all(not device.controlled for device in results["evse_list"]))
        self.assertAlmostEqual(sum(device.delivered_kwh for device in results["evse_list"]), 3.3)


if __name__ == "__main__":
    unittest.main()
