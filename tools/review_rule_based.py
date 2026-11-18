"""Compare recorded controller operation, including phase powers and SoC."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "results/baselines/annual_2023_pv"
CANDIDATE = ROOT / "results/baselines/annual_2023_pv_v2"
OUTPUT = CANDIDATE / "operation_review"
MODES = ("economic", "voltage", "balance")
DEVICE = "bess_b1"
BUS = "bus_004"
TIMEZONE = "Europe/Zurich"
START = "2023-01-01"
END = "2024-01-01"
REMAINDER_START = "2023-02-01"
PLOT_START = "2023-07-01"
PLOT_END = "2023-07-08"
COLORS = {"a": "tab:red", "b": "tab:blue", "c": "tab:green"}


def read_run(root, mode):
    wanted = {"timestamp", "price_per_kwh", "grid_cost", "minimum_voltage_pu", "maximum_vuf_pct",
              "voltage_violations", "price_low", "price_high", "recharge_active"}
    frames = []
    for filename in ("sim_devices_timeseries", "sim_buses_timeseries", "sim_diagnostics_timeseries",
                     "evaluation_timeseries", "actions"):
        frame = pd.read_csv(root / mode / f"{filename}.csv.gz", usecols=lambda column:
                            column in wanted or column.startswith(DEVICE + "_")
                            or column.startswith(BUS + "_") and column.endswith("voltage_pu"))
        frame.index = pd.to_datetime(frame.pop("timestamp"), utc=True).dt.tz_convert(TIMEZONE)
        frames.append(frame)
    frame = pd.concat(frames, axis=1)
    return frame.loc[(frame.index >= pd.Timestamp(START, tz=TIMEZONE))
                     & (frame.index < pd.Timestamp(END, tz=TIMEZONE))]


def summarize(frame):
    dt = (frame.index[1] - frame.index[0]).total_seconds() / 3600
    p = frame[[f"{DEVICE}_{phase}_p_net_kw" for phase in "abc"]]
    phases = frame.filter(regex=f"^{DEVICE}_[abc]_")
    soc = frame[f"{DEVICE}_soc_frac"]
    idle = (soc <= 0.2 + 1e-10) & (phases.abs().max(axis=1) <= 1e-6)
    return {
        "energy_cost": float(frame.grid_cost.sum()), "soc_min": float(soc.min()),
        "soc_final": float(soc.iloc[-1]), "reserve_hours": float((soc <= 0.2 + 1e-10).sum() * dt),
        "charge_kwh": float(frame[f"{DEVICE}_p_charge_kw"].sum() * dt),
        "discharge_kwh": float(frame[f"{DEVICE}_p_discharge_kw"].sum() * dt),
        "active_phase_hours": float((phases.abs().max(axis=1) > 1e-6).sum() * dt),
        "mixed_phase_p_hours": float(((p.max(axis=1) > 1e-8) & (p.min(axis=1) < -1e-8)).sum() * dt),
        "minimum_voltage_pu": float(frame.minimum_voltage_pu.min()),
        "maximum_vuf_pct": float(frame.maximum_vuf_pct.max()),
        "mean_maximum_vuf_pct": float(frame.maximum_vuf_pct.mean()),
        "p95_maximum_vuf_pct": float(frame.maximum_vuf_pct.quantile(0.95)),
        "reserve_idle_hours": float(idle.sum() * dt),
        "longest_reserve_idle_hours": float(idle.groupby(idle.ne(idle.shift()).cumsum()).sum().max() * dt),
        "voltage_violation_hours": float((frame.voltage_violations > 0).sum() * dt),
    }


def plot(mode, old, new):
    begin, end = pd.Timestamp(PLOT_START, tz=TIMEZONE), pd.Timestamp(PLOT_END, tz=TIMEZONE)
    interval = new.index.to_series().diff().median()
    old, new = (frame.loc[(frame.index >= begin) & (frame.index < end)] for frame in (old, new))
    if old.empty or new.empty:
        raise ValueError("Plot dates must overlap the analyzed interval")
    fig, axes = plt.subplots(5, 1, figsize=(13, 12), sharex=True)
    for frame, label, color in ((old, "Previous", "gray"), (new, "Revised", "tab:orange")):
        axes[0].step(frame.index + interval, frame[f"{DEVICE}_soc_frac"], where="post", label=label, color=color)
        voltage = frame[[f"{BUS}_{phase}_voltage_pu" for phase in "abc"]].min(axis=1)
        axes[3].step(frame.index, voltage, where="post", label=label, color=color, linewidth=0.8)
    width = interval / 7
    for axis, field in ((axes[1], "p_net_kw"), (axes[2], "q_injection_kvar")):
        for phase_index, (phase, color) in enumerate(COLORS.items()):
            for version, (frame, label, alpha) in enumerate(((old, "previous", 0.4), (new, "revised", 0.9))):
                axis.bar(frame.index + (2 * phase_index + version) * width,
                         frame[f"{DEVICE}_{phase}_{field}"], width=width, align="edge",
                         color=color, alpha=alpha, label=f"{phase.upper()} {label}")
    axes[0].axhline(0.2, color="black", linestyle="--", linewidth=0.7, label="Reserve")
    axes[3].axhline(0.95, color="red", linestyle="--", linewidth=0.7, label="Lower limit")
    axes[4].step(new.index, new.price_per_kwh, where="post", color="black", linewidth=0.8, label="Current price")
    for field in ("price_low", "price_high"):
        if field in new and new[field].notna().any():
            axes[4].step(new.index, new[field], where="post", linewidth=0.8, label=field)
    for axis, label in zip(axes, ("SoC [fraction]", "Phase P [kW]\npositive charging",
                                  "Phase Q [kvar]\npositive injection", "Local minimum V [pu]", "Price [currency/kWh]")):
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(loc="best", ncol=3, fontsize=8)
    axes[-1].set_xlabel(f"Time [{TIMEZONE}]")
    fig.suptitle(f"BESS operation: {mode}; {PLOT_START} to {PLOT_END} (exclusive)")
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{mode}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows, remainder = {}, {}
    for mode in MODES:
        old, new = read_run(REFERENCE, mode), read_run(CANDIDATE, mode)
        pd.testing.assert_index_equal(old.index, new.index)
        pd.testing.assert_series_equal(old.price_per_kwh, new.price_per_kwh)
        for label, frame in (("previous", old), ("revised", new)):
            rows[(mode, label)] = summarize(frame)
            rest = frame.loc[frame.index >= pd.Timestamp(REMAINDER_START, tz=TIMEZONE)]
            if len(rest) > 1:
                remainder[(mode, label)] = summarize(rest)
        plot(mode, old, new)
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.names = ["mode", "version"]
    table.to_csv(OUTPUT / "comparison.csv")
    if remainder:
        rest = pd.DataFrame.from_dict(remainder, orient="index")
        rest.index.names = ["mode", "version"]
        rest.to_csv(OUTPUT / "february_december.csv")
    print(table.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"Review: {OUTPUT}")


if __name__ == "__main__":
    main()
