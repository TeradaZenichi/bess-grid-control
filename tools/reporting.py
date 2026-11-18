"""Save tables and voltage plots produced by environment runs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from opendss_environment.results_scripts.summary import save_summary


PHASE_COLORS = {"a": "tab:red", "b": "tab:blue", "c": "tab:green"}


def save_results(results, output):
    output = Path(output)
    save_summary(results, output)
    frame = _voltage_frame(results)
    stats = frame.agg(["min", "mean", "max"]).T
    stats.index.names = ["bus", "phase"]
    stats.to_csv(output / "voltage_statistics.csv")
    _plot_voltages(frame, output)
    print(stats.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"Results: {output.resolve()}")


def _voltage_frame(results):
    voltages = results["results"].phase_voltages_pu
    return pd.DataFrame(
        {(bus, phase): values for bus, phases in voltages.items() for phase, values in phases.items()},
        index=pd.DatetimeIndex(results["timestamps"]),
    )


def _plot_voltages(frame, output):
    lower, upper = frame.min().min(), frame.max().max()
    margin = max((upper - lower) * 0.1, 0.0001)
    buses = dict.fromkeys(bus for bus, _ in frame)
    fig, axes = plt.subplots(len(buses), 1, figsize=(14, 2.2 * len(buses)), sharex=True, sharey=True, squeeze=False)
    axes = axes[:, 0]
    for axis, bus in zip(axes, buses):
        for phase in frame[bus]:
            axis.step(frame.index, frame[(bus, phase)], where="post", color=PHASE_COLORS[phase],
                      linewidth=0.6, label=f"Phase {phase.upper()}")
        axis.set_ylabel(f"{bus}\nVoltage [pu]")
        axis.set_ylim(lower - margin, upper + margin)
        axis.axhline(0.95, color="red", linestyle="--", linewidth=0.6)
        axis.axhline(1.05, color="red", linestyle="--", linewidth=0.6)
        axis.grid(alpha=0.25)
    axes[0].legend(loc="lower left", ncol=3)
    axes[-1].set_xlabel("Time [UTC]")
    fig.suptitle("Three-phase bus voltages\nConfigured limits: 0.95-1.05 pu")
    fig.tight_layout()
    fig.savefig(output / "voltages.png", dpi=160)
    plt.close(fig)
