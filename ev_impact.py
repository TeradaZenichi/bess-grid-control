"""Compare the worst aggregate ASAP-charging day with the loads-only case."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from opendss_environment import EnvironmentConfig, MicrogridEnv


ROOT = Path(__file__).resolve().parent
LOADS_CASE = ROOT / "data" / "switzerland" / "5bus" / "config_unbalanced.json"
DEMAND = ROOT / "data" / "switzerland" / "5bus" / "unbalanced_demand.csv"
SESSIONS = ROOT / "data" / "EV" / "ev_sessions.csv"
SCENARIO = "realistic"
SCENARIOS = {
    "realistic": {
        "case": "config_unbalanced_ev_realistic.json",
        "devices": "devices_ev_realistic.json",
        "output": "ev_realistic_impact",
        "label": "24 diversified ASAP EVSEs",
    },
    "all": {
        "case": "config_unbalanced_ev_all_asap.json",
        "devices": "devices_ev_all_asap.json",
        "output": "ev_all_asap_impact",
        "label": "540 ASAP EVSEs",
    },
}
SETTINGS = SCENARIOS[SCENARIO]
CASE_ROOT = ROOT / "data" / "switzerland" / "5bus"
EV_CASE = CASE_ROOT / SETTINGS["case"]
EV_DEVICES = CASE_ROOT / SETTINGS["devices"]
OUTPUT = ROOT / "results" / SETTINGS["output"]
EPISODE_STEPS = 96
STEP_NS = 15 * 60 * 1_000_000_000


def aggregate_asap_power(evse_ids):
    timestamps = pd.to_datetime(pd.read_csv(DEMAND, usecols=["timestamp"])["timestamp"], utc=True)
    sessions = pd.read_csv(SESSIONS, usecols=["evse", "arr", "dep", "e_kwh", "p_kw"])
    sessions = sessions[sessions["evse"].isin(evse_ids)]
    arrival = pd.DatetimeIndex(pd.to_datetime(sessions["arr"], utc=True)).as_unit("ns").asi8
    departure = pd.DatetimeIndex(pd.to_datetime(sessions["dep"], utc=True)).as_unit("ns").asi8
    power = sessions["p_kw"].to_numpy(float)
    charge_end = np.minimum(
        departure,
        arrival + np.rint(sessions["e_kwh"].to_numpy(float) / power * 3.6e12).astype(np.int64),
    )
    grid_start = int(timestamps.iloc[0].value)
    grid_end = grid_start + len(timestamps) * STEP_NS
    start = np.maximum(arrival, grid_start)
    end = np.minimum(charge_end, grid_end)
    valid = end > start
    start, end, power = start[valid], end[valid], power[valid]
    first = (start - grid_start) // STEP_NS
    last = (end - grid_start - 1) // STEP_NS
    profile = np.zeros(len(timestamps))
    same = first == last
    np.add.at(profile, first[same], power[same] * (end[same] - start[same]) / STEP_NS)

    multiple = ~same
    np.add.at(
        profile, first[multiple],
        power[multiple] * (grid_start + (first[multiple] + 1) * STEP_NS - start[multiple]) / STEP_NS,
    )
    np.add.at(
        profile, last[multiple],
        power[multiple] * (end[multiple] - grid_start - last[multiple] * STEP_NS) / STEP_NS,
    )
    full = multiple & (last > first + 1)
    difference = np.zeros(len(timestamps) + 1)
    np.add.at(difference, first[full] + 1, power[full])
    np.add.at(difference, last[full], -power[full])
    profile += np.cumsum(difference[:-1])

    expected_kwh = float(np.sum(power * (end - start) / 3.6e12))
    if not np.isclose(profile.sum() * 0.25, expected_kwh, rtol=0.0, atol=1e-6):
        raise RuntimeError("ASAP profile does not conserve session energy")
    return timestamps, profile


def run_case(case, start_step):
    started = perf_counter()
    env = MicrogridEnv(EnvironmentConfig(case, EPISODE_STEPS, start_step=start_step))
    env.reset()
    branch_history = []
    for _ in range(EPISODE_STEPS):
        _, _, _, _, info = env.step(None)
        flows = {}
        for item in info["branch_measurements"]:
            if item["terminal"] == 1:
                flows[item["element"]] = flows.get(item["element"], 0.0) + item["p_flow_pu"] * env.data["base_kva"]
        branch_history.append(flows)
    voltages = {
        (bus, phase): values
        for bus, phases in env.results.phase_voltages_pu.items()
        for phase, values in phases.items()
    }
    worst = min(
        ((float(values.min()), bus, phase, int(values.argmin())) for (bus, phase), values in voltages.items()),
        key=lambda item: item[0],
    )
    grid_import = np.maximum(-np.asarray(env.grid.array_kw), 0.0)
    evse = env.evse_list
    ev_power = sum((np.asarray(device.array_kw) for device in evse), np.zeros(EPISODE_STEPS))
    ev_peak_step = int(ev_power.argmax())
    metrics = {
        "minimum_voltage_pu": worst[0],
        "minimum_voltage_bus": worst[1],
        "minimum_voltage_phase": worst[2],
        "minimum_voltage_timestamp": env.timestamps.iloc[worst[3]].isoformat(),
        "maximum_grid_import_kw": float(grid_import.max()),
        "maximum_ev_load_kw": float(ev_power[ev_peak_step]),
        "maximum_ev_load_timestamp": env.timestamps.iloc[ev_peak_step].isoformat(),
        "grid_import_kwh": float(grid_import.sum() * env.dt),
        "energy_cost": float(sum(env.results.costs)),
        "steps_with_violations": sum(not item["within_limits"] for item in env.results.diagnostics),
        "voltage_violation_events": sum(item["voltage_violations"] for item in env.results.diagnostics),
        "branch_overload_events": sum(item["branch_overloads"] for item in env.results.diagnostics),
        "grid_limit_events": sum(item["grid_excess_pu"] for item in env.results.diagnostics),
        "ev_delivered_kwh": float(sum(device.delivered_kwh for device in evse)),
        "ev_unmet_kwh": float(sum(device.unmet_kwh for device in evse)),
        "runtime_seconds": perf_counter() - started,
    }
    traces = {
        "timestamps": pd.DatetimeIndex(env.timestamps),
        "minimum_voltage_pu": np.min(np.vstack(tuple(voltages.values())), axis=0),
        "voltages": voltages,
        "grid_import_kw": grid_import,
        "ev_load_kw": ev_power,
        "branch_kw": {
            element: np.asarray([step.get(element, 0.0) for step in branch_history])
            for element in sorted({name for step in branch_history for name in step})
        },
    }
    return metrics, traces


def save_plots(loads, ev):
    timestamps = ev["timestamps"]
    worst_bus = min(ev["voltages"], key=lambda item: ev["voltages"][item].min())[0]
    label = SETTINGS["label"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(timestamps, loads["minimum_voltage_pu"], label="Loads only", linewidth=1.5)
    axes[0].plot(timestamps, ev["minimum_voltage_pu"], label=label, linewidth=1.5)
    axes[0].axhline(0.95, color="tab:red", linestyle="--", label="Lower limit 0.95 pu")
    axes[0].set_ylabel("Minimum voltage [pu]")
    axes[0].legend(ncol=3)
    for phase, color in zip("abc", ("tab:red", "tab:blue", "tab:green")):
        axes[1].plot(
            timestamps, ev["voltages"][(worst_bus, phase)],
            color=color, label=f"{worst_bus} phase {phase.upper()}",
        )
    axes[1].axhline(0.95, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Voltage [pu]")
    axes[1].set_xlabel("Time [UTC]")
    axes[1].legend(ncol=3)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(f"Voltage impact of {label}")
    fig.tight_layout()
    fig.savefig(OUTPUT / "voltage_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(timestamps, loads["grid_import_kw"], label="Grid import - loads only")
    axes[0].plot(timestamps, ev["grid_import_kw"], label=f"Grid import - {label}")
    axes[0].plot(timestamps, ev["ev_load_kw"], label="Aggregate EV load", linestyle="--")
    peak_power = max(ev["grid_import_kw"].max(), ev["ev_load_kw"].max())
    if 2000.0 <= 1.5 * peak_power:
        axes[0].axhline(2000.0, color="tab:red", linestyle=":", label="Grid limit 2000 kW")
    else:
        axes[0].text(
            0.01, 0.94, "Grid limit: 2000 kW (outside scale)",
            color="tab:red", ha="left", va="top", transform=axes[0].transAxes,
        )
    axes[0].set_ylabel("Active power [kW]")
    axes[0].legend(ncol=2)
    colors = plt.cm.tab10.colors
    for index, element in enumerate(ev["branch_kw"]):
        color = colors[index % len(colors)]
        axes[1].plot(timestamps, ev["branch_kw"][element], color=color, label=f"{element} - EV")
        axes[1].plot(
            timestamps, loads["branch_kw"][element], color=color,
            linestyle="--", linewidth=1, alpha=0.7, label=f"{element} - base",
        )
    axes[1].set_ylabel("Upstream active flow [kW]")
    axes[1].set_xlabel("Time [UTC]")
    axes[1].legend(ncol=2, fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle("Grid and branch active-power flows")
    fig.tight_layout()
    fig.savefig(OUTPUT / "power_flow_comparison.png", dpi=180)
    plt.close(fig)

    voltage_data = {
        "timestamp": timestamps,
        "loads_only_minimum_pu": loads["minimum_voltage_pu"],
        "all_evse_minimum_pu": ev["minimum_voltage_pu"],
    }
    for phase in "abc":
        voltage_data[f"{worst_bus}_{phase}_pu"] = ev["voltages"][(worst_bus, phase)]
    pd.DataFrame(voltage_data).to_csv(OUTPUT / "voltage_timeseries.csv", index=False)
    power_data = {
        "timestamp": timestamps,
        "loads_only_grid_import_kw": loads["grid_import_kw"],
        "all_evse_grid_import_kw": ev["grid_import_kw"],
        "all_evse_load_kw": ev["ev_load_kw"],
    }
    power_data.update({f"{name}_kw": values for name, values in ev["branch_kw"].items()})
    pd.DataFrame(power_data).to_csv(OUTPUT / "power_flow_timeseries.csv", index=False)

    buses = sorted({bus for bus, _ in ev["voltages"] if bus != "bus_001"})
    base_values = [
        np.concatenate([values for (name, _), values in loads["voltages"].items() if name == bus])
        for bus in buses
    ]
    ev_values = [
        np.concatenate([values for (name, _), values in ev["voltages"].items() if name == bus])
        for bus in buses
    ]
    positions = np.arange(len(buses)) + 1
    fig, axis = plt.subplots(figsize=(11, 6))
    base_boxes = axis.boxplot(
        base_values, positions=positions - 0.18, widths=0.3,
        patch_artist=True, flierprops={"markersize": 3, "alpha": 0.45},
    )
    ev_boxes = axis.boxplot(
        ev_values, positions=positions + 0.18, widths=0.3,
        patch_artist=True, flierprops={"markersize": 3, "alpha": 0.45},
    )
    for box in base_boxes["boxes"]:
        box.set_facecolor("tab:blue")
        box.set_alpha(0.65)
    for box in ev_boxes["boxes"]:
        box.set_facecolor("tab:orange")
        box.set_alpha(0.65)
    axis.set_xticks(positions, buses)
    axis.set_ylabel("Phase voltage [pu]")
    axis.set_title(f"Voltage distribution on the worst day - {SETTINGS['label']}")
    axis.legend(handles=[
        Patch(facecolor="tab:blue", alpha=0.65, label="Loads only"),
        Patch(facecolor="tab:orange", alpha=0.65, label=SETTINGS["label"]),
    ])
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "voltage_boxplot.png", dpi=180)
    plt.close(fig)

    boxplot_rows = []
    for case, values_by_bus in (("loads_only", base_values), ("evse_asap", ev_values)):
        for bus, values in zip(buses, values_by_bus):
            boxplot_rows.extend({"case": case, "bus": bus, "voltage_pu": value} for value in values)
    pd.DataFrame(boxplot_rows).to_csv(OUTPUT / "voltage_boxplot_data.csv", index=False)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    devices = json.loads(EV_DEVICES.read_text(encoding="utf-8"))["evse"]
    evse_ids = {device["id"] for device in devices}
    timestamps, ev_power = aggregate_asap_power(evse_ids)
    peak_step = int(ev_power.argmax())
    start_step = peak_step - peak_step % EPISODE_STEPS
    pd.DataFrame({"timestamp": timestamps, "ev_asap_kw": ev_power}).to_csv(
        OUTPUT / "ev_asap_aggregate_profile.csv", index=False
    )
    print(f"Peak aggregate EV load: {ev_power[peak_step]:.2f} kW at {timestamps.iloc[peak_step]}")
    print(f"Simulating 96 steps from {timestamps.iloc[start_step]}", flush=True)
    loads_metrics, loads_traces = run_case(LOADS_CASE, start_step)
    ev_metrics, ev_traces = run_case(EV_CASE, start_step)
    save_plots(loads_traces, ev_traces)
    comparison = {
        "scenario": SCENARIO,
        "evse_count": len(devices),
        "installed_power_kw": sum(device["p_max_kw"] for device in devices),
        "start_step": start_step,
        "start_timestamp": timestamps.iloc[start_step].isoformat(),
        "peak_ev_kw": float(ev_power[peak_step]),
        "peak_ev_timestamp": timestamps.iloc[peak_step].isoformat(),
        "loads_only": loads_metrics,
        "evse_asap": ev_metrics,
    }
    (OUTPUT / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2))
    return comparison


if __name__ == "__main__":
    main()
