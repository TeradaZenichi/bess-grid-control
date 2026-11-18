"""Evaluate fixed controllers by local calendar year and save executed results."""
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import pandas as pd

from controllers import CONTROLLER_TYPE, Controller
from controllers.base import Config
from controllers.normalization import ObservationScaling
from opendss_environment import Action, EnvironmentConfig, MicrogridEnv, PhaseAction, Reward, TotalAction
from opendss_environment.results_scripts.summary import save_summary


ROOT = Path(__file__).resolve().parents[1]
PROGRESS_STEPS = 3504


def year_windows(config, groups=None):
    case = Path(config.case_path)
    source = json.loads(case.read_text(encoding="utf-8"))
    timestamps = pd.to_datetime(pd.read_csv(case.parent / source["files"]["demand"],
                                          usecols=["timestamp"])["timestamp"], utc=True)
    years = timestamps.dt.tz_convert(config.observation_timezone).dt.year.to_numpy()
    windows = []
    for group, selected_years in config.evaluation["years"].items():
        if groups is not None and group not in groups:
            continue
        for year in selected_years:
            indices = np.flatnonzero(years == year)
            steps = min(len(indices), config.evaluation["step_limit"] or len(indices))
            windows.append((group, year, int(indices[0]), steps, len(indices)))
    return windows


def run_year(config, group, year, start, steps, expected_steps, folder):
    started = perf_counter()
    folder.mkdir(parents=True)
    env = MicrogridEnv(EnvironmentConfig(config.case_path, steps, start_step=start))
    try:
        observation, _ = env.reset(seed=config.seed)
        if CONTROLLER_TYPE == "sac":
            import torch
            torch.manual_seed(config.seed)  # Identical untrained weights across periods.
        policies = [Controller(device, config.evaluation["controller_settings"],
                               scaling=ObservationScaling.from_environment(env, device["bus"]))
                    for device in env.data["devices"].get("bess", [])]
        env.reward_function = Reward(controllers={policy.id: {
            "devices": {"bess": [policy.id]}, "buses": [policy.bus], "weights": {"node_cost": 1.0},
        } for policy in policies})
        node_costs = []
        with gzip.open(folder / "branches.csv.gz", "wt", encoding="utf-8", newline="", compresslevel=1) as stream:
            writer = None
            for step in range(steps):
                commands = {}
                for policy in policies:
                    local = env.observe_controller(policy.bus, {"bess": [policy.id]},
                                                   extra_features=policy.extra_features, observation=observation)
                    commands[policy.id] = policy.act(local, {})
                pv_actions = {
                    pv.id: PhaseAction(observation["pv"][pv.id]["available_pu"])
                    if pv.dispatch_mode == "per_phase"
                    else TotalAction(sum(observation["pv"][pv.id]["available_pu"].values()))
                    for pv in env.pv_list
                }
                observation, _, terminated, truncated, info = env.step(Action(bess=commands, pv=pv_actions))
                flows = info["branch_measurements"]
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=["timestamp", *flows[0]])
                    writer.writeheader()
                writer.writerows({"timestamp": info["timestamp"], **flow} for flow in flows)
                node_costs.append({"timestamp": info["timestamp"], **{
                    policy.id: Reward.get_costs(info["rewards"], policy.id)["node"] for policy in policies
                }})
                if (step + 1) % PROGRESS_STEPS == 0:
                    print(f"[{group}/{year}] {step + 1}/{steps}; {perf_counter() - started:.1f}s", flush=True)
                if terminated or truncated:
                    break
        results = env.get_episode_results()
        summary = save_summary(results, folder)
        costs = pd.DataFrame(node_costs)
        costs.to_csv(folder / "node_costs.csv.gz", index=False)
        for path in folder.glob("sim_*_timeseries.csv"):
            with path.open("rb") as source, gzip.open(path.with_suffix(".csv.gz"), "wb", compresslevel=1) as target:
                shutil.copyfileobj(source, target)
            path.unlink()
        diagnostics = env.results.diagnostics
        buses = env.results.phase_voltages_pu
        voltage = np.asarray([[phases[p] for p in "abc"] for phases in buses.values()])
        angles = np.asarray([[env.results.phase_angles_deg[bus][p] for p in "abc"] for bus in buses])
        phasors, a = voltage * np.exp(1j * np.deg2rad(angles)), np.exp(2j * np.pi / 3)
        vuf = 100 * np.abs(phasors[:, 0] + a**2 * phasors[:, 1] + a * phasors[:, 2]) / np.abs(
            phasors[:, 0] + a * phasors[:, 1] + a**2 * phasors[:, 2])
        metrics = {
            "group": group, "year": year, "steps": env.idx, "expected_steps": expected_steps,
            "complete_year": env.idx == expected_steps, "feeder_cost": summary["cost"]["total_energy_cost"],
            **summary["energy"], **summary["voltage_pu"], "maximum_vuf_pct": float(vuf.max()),
            "voltage_violation_hours": sum(row["voltage_violations"] > 0 for row in diagnostics) * env.dt,
            "branch_overload_hours": sum(row["branch_overloads"] > 0 for row in diagnostics) * env.dt,
            **{f"node_cost_{policy.id}": float(costs[policy.id].sum()) for policy in policies},
            **{f"final_soc_{device.id}": device.soc for device in env.bess_list},
            "elapsed_seconds": perf_counter() - started,
        }
        metadata = {
            "controller_type": CONTROLLER_TYPE, "policy_status": "untrained" if CONTROLLER_TYPE == "sac" else "rule_based",
            "config": vars(config), "start_step": start, "start": results["timestamps"].iloc[0].isoformat(),
            "end_exclusive": (results["timestamps"].iloc[-1] + pd.Timedelta(hours=env.dt)).isoformat(),
            "reset_each_year": True, "devices": env.data["devices"],
            "controller_settings": {policy.id: vars(policy.config) for policy in policies},
            "scaling": {policy.id: vars(policy.scaling) for policy in policies}, "metrics": metrics,
        }
        (folder / "evaluation.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        print(f"[{group}/{year}] saved {env.idx} steps in {metrics['elapsed_seconds']:.1f}s", flush=True)
        return metrics
    finally:
        env.close()


def evaluate(settings=None, groups=None):
    config = Config.controller()
    config.evaluation.update(settings or {})
    policy_config = Config.controller(settings=config.evaluation["controller_settings"] | {
        "case_path": config.case_path, "seed": config.seed,
    })
    run = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = policy_config.output_path(f"evaluation_{run}", root=ROOT / config.evaluation["output_root"])
    rows = [run_year(config, group, year, start, steps, expected, output / group / str(year))
            for group, year, start, steps, expected in year_windows(config, groups)]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "comparison.csv", index=False)
    print(frame[["group", "year", "steps", "feeder_cost", "minimum_voltage", "maximum_vuf_pct"]].to_string(index=False))
    print(f"Evaluation: {output}")
    return frame


if __name__ == "__main__":
    evaluate()
