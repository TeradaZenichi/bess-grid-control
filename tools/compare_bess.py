"""Compare identical ASAP-EV scenarios with and without a local BESS controller."""
from copy import deepcopy
import json
from os.path import relpath
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import main as runner
from controllers import CONTROLLER_TYPE
from controllers.config import load_config


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "data/switzerland/5bus/config.json"
OUTPUT = ROOT / "results/bess_comparison_pv"
EPISODE_STEPS = 96
START_STEP = 0
SEED = 42
MODE = "economic"
BESS = {
    "id": "b1", "bus": "bus_004_lv", "phases": ["a", "b", "c"],
    "connection": "wye", "dispatch_mode": "per_phase",
    "e_cap_kwh": 100.0, "p_charge_max_kw": 40.0, "p_discharge_max_kw": 40.0,
    "s_max_kva": 50.0, "reactive_control": True, "q_loss_rated_kw": 0.5,
    "eta_charge": 0.95, "eta_discharge": 0.95,
    "soc_init_frac": 0.5, "soc_min_frac": 0.1, "soc_max_frac": 1.0,
    "cyclic_soc": False,
}


def prepare_case(case, folder, with_bess, *, exist_ok=False):
    folder.mkdir(parents=True, exist_ok=exist_ok)
    config = json.loads(case.read_text(encoding="utf-8"))
    devices = json.loads((case.parent / config["files"]["devices"]).read_text(encoding="utf-8"))
    if devices.get("bess", []) not in ([], [BESS]) or any(
            device["control_mode"] != "asap" for device in devices.get("evse", [])):
        raise ValueError("Comparison expects the reference BESS or none, optional fixed PV and ASAP EVSEs")
    if any(pv["control"] != "fixed_pf" or pv["curtailable"] for pv in devices.get("pv", [])):
        raise ValueError("Comparison PV must use fixed power factor without curtailment")

    def profile_reference(reference):
        filename, column = reference.rsplit(":", 1)
        return f"{Path(relpath((case.parent / filename).resolve(), folder.resolve())).as_posix()}:{column}"

    for pv in devices.get("pv", []):
        pv["profile"] = profile_reference(pv["profile"])
        if pv.get("phase_profiles"):
            pv["phase_profiles"] = {phase: profile_reference(ref) for phase, ref in pv["phase_profiles"].items()}
    devices["bess"] = [deepcopy(BESS)] if with_bess else []
    for key, value in config["files"].items():
        config["files"][key] = str((case.parent / value).resolve())
    config["files"]["devices"] = "devices.json"
    config["network"]["master"] = str((case.parent / config["network"]["master"]).resolve())
    (folder / "devices.json").write_text(json.dumps(devices, indent=2), encoding="utf-8")
    (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return folder / "config.json"


def run_case(name, with_bess):
    folder = OUTPUT / name
    runner.CASE, runner.OUTPUT = prepare_case(CASE, folder, with_bess, exist_ok=True), folder
    runner.EPISODE_STEPS, runner.START_STEP, runner.SEED = EPISODE_STEPS, START_STEP, SEED
    return runner.main({"mode": MODE})


def summarize(results):
    dt = results["dt"]
    grid = -np.asarray(results["grid"].array_kw)
    voltages = results["results"].phase_voltages_pu
    angles = results["results"].phase_angles_deg
    minimum = np.min(np.vstack([v for phases in voltages.values() for v in phases.values()]), axis=0)
    maximum = max(float(v.max()) for phases in voltages.values() for v in phases.values())
    a = np.exp(2j * np.pi / 3)
    vuf = []
    for bus, phases in voltages.items():
        if set(phases) == {"a", "b", "c"}:
            va, vb, vc = (phases[p] * np.exp(1j * np.deg2rad(angles[bus][p])) for p in "abc")
            vuf.append(100 * np.abs(va + a**2 * vb + a * vc) / np.abs(va + a * vb + a**2 * vc))
    maximum_vuf = np.max(vuf, axis=0)
    ev_power = sum((np.asarray(d.array_kw) for d in results["evse_list"]), np.zeros(len(grid)))
    pv_power = sum((np.asarray(d.array_kw) for d in results["pv_list"]), np.zeros(len(grid)))
    diagnostics = results["results"].diagnostics
    cost = float(sum(results["results"].costs))
    metrics = {
        "energy_cost": cost,
        "grid_import_kwh": float(np.maximum(grid, 0).sum() * dt),
        "grid_export_kwh": float(np.maximum(-grid, 0).sum() * dt),
        "peak_import_kw": float(np.maximum(grid, 0).max()),
        "minimum_voltage_pu": float(minimum.min()), "maximum_voltage_pu": maximum,
        "maximum_vuf_pct": float(maximum_vuf.max()),
        "voltage_violation_steps": sum(d["voltage_violations"] > 0 for d in diagnostics),
        "branch_overload_steps": sum(d["branch_overloads"] > 0 for d in diagnostics),
        "ev_delivered_kwh": float(sum(d.delivered_kwh for d in results["evse_list"])),
        "ev_unmet_kwh": float(sum(d.unmet_kwh for d in results["evse_list"])),
        "pv_generation_kwh": float(pv_power.sum() * dt),
        "bess_soc_initial": None, "bess_soc_final": None,
        "bess_charge_kwh": 0.0, "bess_discharge_kwh": 0.0,
        "terminal_recharge_kwh_estimate": 0.0, "cost_plus_terminal_recharge_estimate": cost,
    }
    trace = pd.DataFrame({"timestamp": results["timestamps"],
                          "interval_end": pd.to_datetime(results["timestamps"], utc=True) + pd.to_timedelta(dt, unit="h"),
                          "minimum_voltage_pu": minimum,
                          "maximum_vuf_pct": maximum_vuf, "grid_import_kw": grid,
                          "ev_power_kw": ev_power, "pv_power_kw": pv_power})
    if results["bess_list"]:
        device = results["bess_list"][0]
        recharge = max(BESS["soc_init_frac"] - device.soc, 0) * device.e_cap_kwh / device.eta_charge
        metrics.update({
            "bess_soc_initial": BESS["soc_init_frac"], "bess_soc_final": float(device.soc),
            "bess_charge_kwh": float(sum(device.array_charge_kw) * dt),
            "bess_discharge_kwh": float(sum(device.array_discharge_kw) * dt),
            "terminal_recharge_kwh_estimate": recharge,
            "cost_plus_terminal_recharge_estimate": cost + recharge * float(results["grid"].prices[-1]),
        })
        trace["bess_p_kw"], trace["bess_q_kvar"], trace["bess_soc"] = device.array_kw, device.array_kvar, device.array_soc
    return metrics, trace


def plot_comparison(traces):
    fig, axes = plt.subplots(5, 1, figsize=(13, 13), sharex=True)
    for name, trace in traces.items():
        time = pd.to_datetime(trace.timestamp, utc=True)
        for axis, field in zip(axes[:3], ("minimum_voltage_pu", "maximum_vuf_pct", "grid_import_kw")):
            axis.step(time, trace[field], where="post", label=name)
    trace = traces["with_bess"]
    time = pd.to_datetime(trace.timestamp, utc=True)
    interval = pd.to_datetime(trace.interval_end, utc=True) - time
    axes[3].bar(time, trace.bess_p_kw, width=interval * 0.45, align="edge", label="P: positive charging")
    axes[3].bar(time + interval * 0.5, trace.bess_q_kvar, width=interval * 0.45, align="edge", label="Q: positive injection")
    soc_time = pd.DatetimeIndex([time.iloc[0], *pd.to_datetime(trace.interval_end, utc=True)])
    axes[4].step(soc_time, [BESS["soc_init_frac"], *trace.bess_soc], where="post", label="SoC")
    axes[4].axhline(BESS["soc_init_frac"], linestyle="--", color="black", label="Initial SoC")
    axes[0].axhline(0.95, linestyle="--", color="red", label="Lower limit")
    for axis, label in zip(axes, ("Minimum voltage [pu]", "Maximum VUF [%]", "Grid import [kW]", "BESS P/Q [kW/kvar]", "SoC [fraction]")):
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", ncol=2)
    axes[-1].set_xlabel("Time [UTC]")
    fig.suptitle(f"BESS comparison: {MODE}; identical loads and ASAP EV sessions")
    fig.tight_layout()
    fig.savefig(OUTPUT / "comparison.png", dpi=160)
    plt.close(fig)


def main():
    if CONTROLLER_TYPE != "rule_based":
        raise ValueError("This comparison requires CONTROLLER_TYPE=rule_based")
    metrics, traces = {}, {}
    for name, enabled in (("without_bess", False), ("with_bess", True)):
        results = run_case(name, enabled)
        metrics[name], traces[name] = summarize(results)
        traces[name].to_csv(OUTPUT / name / "comparison_timeseries.csv", index=False)
    pd.testing.assert_series_equal(traces["without_bess"].timestamp, traces["with_bess"].timestamp)
    np.testing.assert_allclose(traces["without_bess"].ev_power_kw, traces["with_bess"].ev_power_kw, atol=1e-10)
    np.testing.assert_array_equal(traces["without_bess"].pv_power_kw, traces["with_bess"].pv_power_kw)
    summary = pd.DataFrame(metrics)
    summary.to_csv(OUTPUT / "comparison.csv", index_label="metric")
    manifest = {
        "case": str(CASE.relative_to(ROOT)), "start_step": START_STEP,
        "episode_steps": EPISODE_STEPS, "seed": SEED, "bess": BESS,
        "rule_settings": vars(load_config(ROOT / "controllers/rule_based/config.json", {"mode": MODE})),
        "metrics": metrics,
        "notes": ["Identical exogenous loads, prices and ASAP EV profiles in both runs.",
                  "No terminal SoC equality is imposed. Raw cost includes use of initial battery inventory.",
                  "Terminal recharge estimate uses the final buy tariff and battery efficiency; excludes network losses and scheduling feasibility."],
    }
    (OUTPUT / "comparison.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    plot_comparison(traces)
    print(summary.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"Comparison: {OUTPUT}")


if __name__ == "__main__":
    main()
