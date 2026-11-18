import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from opendss_environment import Action, EnvironmentConfig, MicrogridEnv, Reward, TotalAction
from opendss_environment.data import episode_data, load_data


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "data/switzerland/5bus/config.json"


class MVLVNetworkTest(unittest.TestCase):
    def test_lv_bases_placement_and_executed_power_balance_with_idle_bess(self):
        env = MicrogridEnv(EnvironmentConfig(CASE, 1, start_step=32230))
        self.addCleanup(env.close)
        env.reset(seed=42)
        _, _, _, _, info = env.step(None)
        self.assertEqual(len(env.current_bus_voltages_pu), 9)
        self.assertTrue(all(load.bus.endswith("_lv") for load in env.load_list))
        self.assertTrue(all(device.bus.endswith("_lv") for device in env.evse_list))
        self.assertEqual(env.bess_list[0].bus, "bus_004_lv")
        self.assertEqual(env.pv_list[0].bus, "bus_005")
        self.assertEqual(env.bess_list[0].array_kw, [0])
        self.assertEqual(env.bess_list[0].array_kvar, [0])
        self.assertEqual(env.bess_list[0].soc, 0.5)
        for bus in env.current_bus_voltages_pu:
            env.dss.circuit.set_active_bus(bus)
            nominal = 220 if bus.endswith("_lv") else 11000 / np.sqrt(3)
            self.assertAlmostEqual(env.dss.bus.kv_base * 1000, nominal)
            for phase, pu in env.current_bus_voltages_pu[bus].items():
                self.assertAlmostEqual(pu * nominal, env.current_bus_snapshot.volts[bus][phase], places=6)
        for load in env.load_list:
            self.assertEqual(load.id.removeprefix("Load_"), load.bus.removesuffix("_lv") + "_" + "abc"[load.phase_node - 1])
            env.dss.circuit.set_active_element("Load." + load.id)
            self.assertAlmostEqual(sum(env.dss.cktelement.powers[::2]), load.array_kw[0], places=5)
        losses = sum(flow["p_flow_pu"] for flow in info["branch_measurements"]) * env.data["base_kva"]
        self.assertGreater(losses, 0)
        demand = sum(load.array_kw[0] for load in env.load_list)
        ev = sum(device.array_kw[0] for device in env.evse_list)
        pv = sum(device.array_kw[0] for device in env.pv_list)
        self.assertAlmostEqual(-env.current_grid_kw + pv, demand + ev + losses, places=4)

    def test_local_bill_includes_loads_evs_and_bess_but_excludes_remote_pv(self):
        reward = Reward(controllers={
            "aggregator": {"buses": ["bus_004_lv"], "devices": {"bess": ["b1"]},
                           "weights": {"node_cost": 1}},
            "device": {"devices": {"bess": ["b1"]}, "weights": {"device_cost": 1}},
        })
        env = MicrogridEnv(EnvironmentConfig(CASE, 1, start_step=15988, reward_function=reward))
        self.addCleanup(env.close)
        env.reset(seed=42)
        _, value, _, _, info = env.step(Action(bess={"b1": TotalAction(0.02)}, pv={"pv1": TotalAction(0)}))
        power = sum(d.array_kw[0] for d in env.load_list if d.bus == "bus_004_lv")
        power += sum(d.array_kw[0] for d in env.evse_list if d.bus == "bus_004_lv")
        power += env.bess_list[0].array_kw[0]
        self.assertGreater(env.pv_list[0].array_kw[0], 0)
        self.assertAlmostEqual(reward.get_costs(info["rewards"], "aggregator")["node"],
                               power * env.grid.prices[0] * env.dt, places=5)
        self.assertAlmostEqual(reward.get_costs(info["rewards"], "device")["device"],
                               20 * env.grid.prices[0] * env.dt, places=5)
        self.assertEqual(value, -info["cost"])

    def test_optional_load_map_preserves_source_ids_phases_and_profiles(self):
        fixture = ROOT / "opendss_environment/tests/fixtures/three_phase_volt_control"
        with TemporaryDirectory() as directory:
            case = Path(directory) / "case"
            shutil.copytree(fixture, case)
            path = case / "config_per_phase.json"
            config = json.loads(path.read_text())
            original = episode_data(load_data(path), 0, 2)["load_list"]
            config["network"]["load_bus_map"] = {"load": "load_lv"}
            path.write_text(json.dumps(config))
            mapped = episode_data(load_data(path), 0, 2)["load_list"]
            for before, after in zip(original, mapped):
                self.assertEqual(after.bus, "load_lv" if before.bus == "load" else before.bus)
                self.assertEqual(before.id, after.id)
                self.assertEqual(before.phase_node, after.phase_node)
                np.testing.assert_array_equal(before.array_kw, after.array_kw)
                np.testing.assert_array_equal(before.array_kvar, after.array_kvar)
            config["network"]["load_bus_map"] = {"load": 123}
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "load_bus_map"):
                load_data(path)


if __name__ == "__main__":
    unittest.main()
