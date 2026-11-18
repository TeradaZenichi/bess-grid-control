"""Run annual rule-based baselines: python -m tools.run_baselines."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import subprocess
from time import perf_counter
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from controllers import Controller
from controllers.base import Config
from controllers import CONTROLLER_TYPE
from opendss_environment import Action, EnvironmentConfig, MicrogridEnv, PhaseAction, TotalAction
from opendss_environment.results_scripts.summary import save_summary
from tools.compare_bess import BESS, prepare_case, summarize


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "data/switzerland/5bus/config.json"
OUTPUT = ROOT / "results/baselines/annual_2023_pv_v2"
MODES = ("without_bess", "economic", "voltage", "balance")
EPISODE_STEPS = 365 * 96
START_STEP = 0
SEED = 42
MAX_WORKERS = 2
LOCAL_TIMEZONE = "Europe/Zurich"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")


def run_mode(mode, output, steps):
    started = perf_counter()
    folder = output / mode
    case = prepare_case(CASE, folder, mode != "without_bess")
    env = MicrogridEnv(EnvironmentConfig(case, steps, start_step=START_STEP))
    try:
        observation, _ = env.reset(seed=SEED)
        policy = Controller(BESS, {"mode": mode}) if mode != "without_bess" else None
        settings = vars(policy.config) if policy else None
        commands = []
        loop_started = perf_counter()
        with gzip.open(folder / "branch_timeseries.csv.gz", "wt", encoding="utf-8", newline="", compresslevel=1) as stream:
            writer = None
            for index in range(steps):
                action = None
                if policy:
                    local = env.observe_controller(policy.bus, {"bess": [policy.id]},
                                                   extra_features=policy.extra_features, observation=observation)
                    command = policy.act(local, {})
                    action = Action(bess={policy.id: command}, pv={
                        pv.id: PhaseAction(observation["pv"][pv.id]["available_pu"])
                        if pv.dispatch_mode == "per_phase"
                        else TotalAction(sum(observation["pv"][pv.id]["available_pu"].values()))
                        for pv in env.pv_list
                    })
                    row = {"timestamp": observation["timestamp"]}
                    row.update(policy.last_decision)
                    for name, requested in (("mode_request", policy.last_requested_action), ("command", command)):
                        for field, values in (("p_net_pu", requested.p_pu), ("q_injection_pu", requested.q_pu)):
                            row.update({f"{name}_{phase}_{field}": value for phase, value in values.items()})
                    commands.append(row)
                observation, _, terminated, truncated, info = env.step(action)
                flows = info["branch_measurements"]
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=["timestamp", *flows[0]])
                    writer.writeheader()
                writer.writerows({"timestamp": info["timestamp"], **flow} for flow in flows)
                if (index + 1) % 3504 == 0:
                    print(f"[{mode}] {index + 1}/{steps}; {perf_counter() - loop_started:.1f} s", flush=True)
                if terminated or truncated:
                    break
        simulation_seconds = perf_counter() - loop_started
        if env.idx != steps or truncated:
            raise RuntimeError(f"Incomplete baseline: {mode}, {env.idx}/{steps}")
        results = env.get_episode_results()
        save_summary(results, folder)
        metrics, trace = summarize(results)
        voltage = np.vstack([values for phases in env.results.phase_voltages_pu.values() for values in phases.values()])
        metrics.update({
            "steps": env.idx, "dt_h": env.dt, "simulation_seconds": simulation_seconds,
            "mean_absolute_voltage_deviation_pu": float(np.abs(voltage - 1).mean()),
            "mean_maximum_vuf_pct": float(trace.maximum_vuf_pct.mean()),
            "p95_maximum_vuf_pct": float(trace.maximum_vuf_pct.quantile(0.95)),
            "voltage_violation_hours": metrics["voltage_violation_steps"] * env.dt,
            "bess_inverter_loss_kwh": float(sum(sum(d.array_inverter_loss_kw) * env.dt for d in env.bess_list)),
            "episode_reward": float(env.episode_reward),
        })
        trace["energy_cost"] = env.results.costs
        trace["grid_q_kvar"] = env.grid.array_kvar
        trace.to_csv(folder / "evaluation_timeseries.csv.gz", index=False)
        if commands:
            pd.DataFrame(commands).to_csv(folder / "actions.csv.gz", index=False)
        for path in folder.glob("sim_*_timeseries.csv"):
            with path.open("rb") as source, gzip.open(path.with_suffix(".csv.gz"), "wb", compresslevel=1) as target:
                shutil.copyfileobj(source, target)
            path.unlink()
        metrics["total_seconds"] = perf_counter() - started
        write_json(folder / "metrics.json", {"mode": mode, "settings": settings, "metrics": metrics})
        print(f"[{mode}] saved: cost={metrics['energy_cost']:.2f}; Vmin={metrics['minimum_voltage_pu']:.5f}; "
              f"VUFmax={metrics['maximum_vuf_pct']:.4f}%; {metrics['total_seconds']:.1f} s", flush=True)
        return mode, metrics
    finally:
        env.close()


def manifest():
    config = json.loads(CASE.read_text(encoding="utf-8"))
    timestamps = pd.to_datetime(pd.read_csv(CASE.parent / config["files"]["demand"],
                                          usecols=["timestamp"])["timestamp"], utc=True)
    timestamps = timestamps.iloc[START_STEP:START_STEP + EPISODE_STEPS]
    if len(timestamps) != EPISODE_STEPS or not timestamps.diff().dropna().eq(pd.Timedelta(minutes=15)).all():
        raise ValueError("Baselines require the requested number of consecutive 15-minute intervals")
    inputs = [CASE, *(CASE.parent / path for path in config["files"].values()),
              *(CASE.parent / "dss").rglob("*.dss")]
    hashes = {}
    for path in inputs:
        with path.open("rb") as stream:
            hashes[str(path.resolve().relative_to(ROOT))] = hashlib.file_digest(stream, "sha256").hexdigest()
    with ZipFile(OUTPUT / "source.zip", "w", ZIP_DEFLATED) as archive:
        for path in [ROOT / "main.py", ROOT / "requirements.txt", ROOT / "opendss_environment/pyproject.toml",
                     ROOT / "opendss_environment/requirements.txt", *ROOT.glob("controllers/**/*.py"),
                     *ROOT.glob("controllers/**/*.json"), *ROOT.glob("tools/**/*.py"), *ROOT.glob("opendss_environment/opendss_environment/**/*.py"),
                     *(CASE.parent / "dss").rglob("*.dss")]:
            archive.write(path, str(path.relative_to(ROOT)))
    git = {}
    for name, path in (("root", ROOT), ("environment", ROOT / "opendss_environment")):
        git[name] = {"head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip(),
                     "status": subprocess.check_output(["git", "status", "--porcelain"], cwd=path, text=True).splitlines()}
    return {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "case": str(CASE.relative_to(ROOT)), "steps": EPISODE_STEPS, "start_step": START_STEP,
        "interval_start": timestamps.iloc[0].isoformat(),
        "interval_end_exclusive": (timestamps.iloc[-1] + pd.Timedelta(minutes=15)).isoformat(),
        "seed": SEED, "modes": list(MODES), "max_workers": MAX_WORKERS, "bess": BESS,
        "pv": json.loads((CASE.parent / config["files"]["devices"]).read_text())["pv"],
        "local_timezone": LOCAL_TIMEZONE, "input_sha256": hashes, "git": git,
        "python": platform.python_version(),
        "packages": {name: version(name) for name in ("numpy", "pandas", "gymnasium", "OpenDSSDirect.py", "pu-normalizer")},
        "rule_settings": vars(Config.controller("rule_based")),
        "notes": ["EVSEs are ASAP and PV follows the available profile at fixed power factor in every mode.",
                  "Each run resets once; SoC remains continuous for the complete episode.",
                  "No terminal SoC equality is imposed. Compare initial/final SoC along with energy cost.",
                  "Terminal recharge cost is only an estimate; it excludes network losses and scheduling feasibility.",
                  "Tracking is excluded until a supervisor provides references. No RL policy is implemented yet.",
                  "Source archive captures the actual working code, including uncommitted changes.",
                  "Branch flows and action requests use network pu; device/bus CSVs use units in column names.",
                  "Simulation times include controllers and branch logging; runs may execute concurrently."],
    }


def compare(metrics):
    traces = {mode: pd.read_csv(OUTPUT / mode / "evaluation_timeseries.csv.gz") for mode in MODES}
    reference = traces["without_bess"]
    for trace in traces.values():
        pd.testing.assert_series_equal(trace.timestamp, reference.timestamp)
        np.testing.assert_array_equal(trace.ev_power_kw, reference.ev_power_kw)
        np.testing.assert_array_equal(trace.pv_power_kw, reference.pv_power_kw)
    table = pd.DataFrame.from_dict(metrics, orient="index").reindex(MODES)
    table.index.name = "mode"
    table["cost_change_pct"] = 100 * (table.energy_cost / table.loc["without_bess", "energy_cost"] - 1)
    table.to_csv(OUTPUT / "comparison.csv")
    fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True)
    fields = ("energy_cost", "grid_import_kw", "minimum_voltage_pu", "maximum_vuf_pct", "bess_soc")
    labels = ("Daily cost [currency]", "Daily peak import [kW]", "Daily minimum V [pu]", "Daily maximum VUF [%]", "SoC [fraction]")
    for mode, trace in traces.items():
        trace.index = pd.to_datetime(trace.timestamp, utc=True).dt.tz_convert(LOCAL_TIMEZONE)
        for axis, field in zip(axes, fields):
            if field in trace:
                reducer = {"energy_cost": "sum", "grid_import_kw": "max", "minimum_voltage_pu": "min",
                           "maximum_vuf_pct": "max", "bess_soc": "last"}[field]
                trace[field].resample("D").agg(reducer).plot(ax=axis, label=mode, linewidth=0.8,
                                                          color=f"C{MODES.index(mode)}", drawstyle="steps-post")
    axes[2].axhline(0.95, color="red", linestyle="--", linewidth=0.8)
    for axis, label in zip(axes, labels):
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", ncol=4)
    fig.suptitle(f"Rule-based baselines: {EPISODE_STEPS} intervals; fixed PV and ASAP EV sessions")
    fig.tight_layout()
    fig.savefig(OUTPUT / "comparison.png", dpi=160)
    plt.close(fig)
    return table


def main():
    if CONTROLLER_TYPE != "rule_based":
        raise ValueError("Annual baselines require CONTROLLER_TYPE=rule_based")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    metadata = manifest()
    write_json(OUTPUT / "manifest.json", metadata)
    metrics = {}
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        pending = [pool.submit(run_mode, mode, OUTPUT, EPISODE_STEPS) for mode in MODES]
        for future in as_completed(pending):
            mode, metrics[mode] = future.result()
    table = compare(metrics)
    metadata.update(status="complete", finished_at=datetime.now(timezone.utc).isoformat(),
                    validation="All timestamps, ASAP EV aggregate powers and fixed PV generation match exactly across runs.")
    write_json(OUTPUT / "manifest.json", metadata)
    print(table[["energy_cost", "minimum_voltage_pu", "maximum_vuf_pct", "voltage_violation_steps", "bess_soc_final"]])
    print(f"Baselines: {OUTPUT}")


if __name__ == "__main__":
    main()
