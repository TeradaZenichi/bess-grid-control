"""Evaluate the MV/LV case with idle BESS: python -m tools.run_idle_bess."""
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opendss_environment import EnvironmentConfig, MicrogridEnv
from tools.reporting import PHASE_COLORS, save_results


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "data/switzerland/5bus/config.json"
OUTPUT = ROOT / "results/mv_lv_idle_bess_day"
START_STEP = 32160  # Local December 2, 2023: peak conventional apparent demand day.
EPISODE_STEPS = 96
SEED = 42
LV_NOMINAL_V = 220.0


def plot(results, flows):
    time = pd.DatetimeIndex(results["timestamps"])
    steps = results["steps"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    series = {"Grid import (negative export)": -np.asarray(results["grid"].array_kw)}
    for label, kind in (("Conventional loads", "load"), ("EVSEs ASAP", "evse"), ("PV generation", "pv")):
        series[label] = sum((np.asarray(d.array_kw) for d in results[f"{kind}_list"]), np.zeros(steps))
    for label, values in series.items():
        axes[0].step(time, values, where="post", label=label)
    axes[0].set_ylabel("Active power [kW]")
    for element, values in flows.items():
        axes[1].step(values.index, values, where="post", label=element)
    axes[1].axhline(100, color="red", linestyle="--", label="Rated current")
    axes[1].set_ylabel("Worst phase/terminal loading [%]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(loc="best", ncol=2)
    axes[-1].set_xlabel("Time [UTC]")
    fig.tight_layout()
    fig.savefig(OUTPUT / "power_flow.png", dpi=160)
    plt.close(fig)

    buses = [bus for bus in results["results"].phase_voltages if bus.endswith("_lv")]
    fig, axes = plt.subplots(len(buses), 1, figsize=(13, 9), sharex=True, sharey=True, squeeze=False)
    for axis, bus in zip(axes[:, 0], buses):
        for phase, values in results["results"].phase_voltages[bus].items():
            axis.step(time, values, where="post", color=PHASE_COLORS[phase], label=phase.upper())
        for limit in (0.95, 1.05):
            axis.axhline(LV_NOMINAL_V * limit, color="red", linestyle="--", linewidth=0.7)
        axis.axhline(LV_NOMINAL_V, color="black", linestyle=":", linewidth=0.6)
        axis.set_ylabel(f"{bus}\nPhase-neutral [V]")
        axis.legend(ncol=3)
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Time [UTC]")
    fig.tight_layout()
    fig.savefig(OUTPUT / "lv_voltages_v.png", dpi=160)
    plt.close(fig)


def main():
    started = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    env = MicrogridEnv(EnvironmentConfig(CASE, EPISODE_STEPS, start_step=START_STEP))
    try:
        env.reset(seed=SEED)
        rows = []
        for _ in range(EPISODE_STEPS):
            _, _, _, _, info = env.step(None)  # Idle BESS, available fixed PV and ASAP EVSEs.
            rows.extend({"timestamp": info["timestamp"], **flow} for flow in info["branch_measurements"])
        results = env.get_episode_results()
        for battery in results["bess_list"]:
            np.testing.assert_allclose(battery.array_kw, 0, atol=1e-10)
            np.testing.assert_allclose(battery.array_kvar, 0, atol=1e-10)
            np.testing.assert_allclose(battery.array_soc, battery.soc_init_frac, atol=1e-10)
        save_results(results, OUTPUT)
        branches = pd.DataFrame(rows)
        branches["p_kw"] = branches.p_flow_pu * env.data["base_kva"]
        branches.to_csv(OUTPUT / "branch_timeseries.csv", index=False)
        transformers = branches[branches.element.str.lower().str.startswith("transformer.")]
        loading = transformers.groupby(["timestamp", "element"]).loading_pct.max().unstack()
        plot(results, loading)
        voltage = results["results"].phase_voltages_pu
        report = {"start": env.timestamps.iloc[0].isoformat(), "steps": EPISODE_STEPS,
                  "start_step": START_STEP, "dt_h": env.dt, "seed": SEED,
                  "bess_p_kw": 0, "bess_q_kvar": 0, "bess_final_soc": env.bess_list[0].soc,
                  "voltage": {}, "transformer_peak_loading_pct": loading.max().to_dict(),
                  "network_loss_kwh": float(branches.p_kw.sum() * env.dt),
                  "voltage_violation_hours": sum(d["voltage_violations"] > 0 for d in env.results.diagnostics) * env.dt,
                  "branch_overload_hours": sum(d["branch_overloads"] > 0 for d in env.results.diagnostics) * env.dt,
                  "runtime_seconds": perf_counter() - started}
        for bus, phases in voltage.items():
            values = np.stack(list(phases.values()))
            report["voltage"][bus] = {"minimum_pu": float(values.min()), "maximum_pu": float(values.max())}
        series = {"grid": -np.asarray(env.grid.array_kw),
                  "load": sum((d.array_kw for d in env.load_list), np.zeros(env.steps)),
                  "ev": sum((np.asarray(d.array_kw) for d in env.evse_list), np.zeros(env.steps)),
                  "pv": sum((np.asarray(d.array_kw) for d in env.pv_list), np.zeros(env.steps))}
        pd.DataFrame(series, index=env.timestamps).to_csv(OUTPUT / "power_timeseries.csv", index_label="timestamp")
        np.testing.assert_allclose(series["grid"] + series["pv"],
                                   series["load"] + series["ev"] + branches.groupby("timestamp").p_kw.sum().to_numpy(), atol=1e-4)
        (OUTPUT / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return report
    finally:
        env.close()


if __name__ == "__main__":
    main()
