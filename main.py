"""Run the five-bus feeder with its unbalanced, three-phase loads."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from opendss_environment import Action, EnvironmentConfig, EVSEAction, MicrogridEnv
from opendss_environment.results_scripts.summary import save_summary


ROOT = Path(__file__).resolve().parent
USE_EV = False
CASE_NAME = "config_unbalanced_ev.json" if USE_EV else "config_unbalanced.json"
CASE = ROOT / "data" / "switzerland" / "5bus" / CASE_NAME
EPISODE_STEPS = 96 if USE_EV else None
START_STEP = 0
SEED = 42
CONTROLLED_EV_POWER_PU = None  # None keeps controlled EVSEs idle; e.g. 0.022.
OUTPUT = ROOT / "results" / ("5bus_unbalanced_ev" if USE_EV else "5bus_unbalanced")
PHASE_COLORS = {"a": "tab:red", "b": "tab:blue", "c": "tab:green"}
BLOCK_STEPS = 96


def combine_results(blocks):
    combined = blocks[0]
    combined["steps"] = sum(block["steps"] for block in blocks)
    combined["timestamps"] = pd.concat([block["timestamps"] for block in blocks], ignore_index=True)
    combined["episode_reward"] = sum(block["episode_reward"] for block in blocks)
    for field in ("array_kw", "array_kvar", "prices", "sell_prices"):
        if hasattr(combined["grid"], field):
            setattr(combined["grid"], field,
                    np.concatenate([getattr(block["grid"], field) for block in blocks]))
    for index, load in enumerate(combined["load_list"]):
        for field in ("array_kw", "array_kvar"):
            setattr(load, field, np.concatenate([getattr(block["load_list"][index], field) for block in blocks]))
    result = combined["results"]
    for field in ("costs", "diagnostics"):
        setattr(result, field, [value for block in blocks for value in getattr(block["results"], field)])
    for field in ("voltages", "voltages_pu"):
        setattr(result, field, {bus: np.concatenate([getattr(block["results"], field)[bus] for block in blocks])
                                for bus in getattr(result, field)})
    for field in ("phase_voltages", "phase_voltages_pu", "phase_angles_deg"):
        setattr(result, field, {bus: {phase: np.concatenate([
            getattr(block["results"], field)[bus][phase] for block in blocks]) for phase in phases}
            for bus, phases in getattr(result, field).items()})
    return combined


def plot_voltages(results):
    timestamps = pd.DatetimeIndex(results["timestamps"])
    voltages = results["results"].phase_voltages_pu
    frame = pd.DataFrame({(bus, phase): values for bus, phases in voltages.items()
                          for phase, values in phases.items()}, index=timestamps)
    stats = frame.agg(["min", "mean", "max"]).T
    stats.index.names = ["bus", "phase"]
    stats.to_csv(OUTPUT / "voltage_statistics.csv")
    worst_time = frame.min(axis=1).idxmin()
    day_start = worst_time.normalize()
    windows = {
        "voltage_full_series": frame,
        "voltage_worst_day": frame.loc[(frame.index >= day_start) &
                                        (frame.index < day_start + pd.Timedelta(days=1))],
    }
    lower, upper = frame.min().min(), frame.max().max()
    margin = max((upper - lower) * 0.1, 0.0001)
    for name, window in windows.items():
        fig, axes = plt.subplots(len(voltages), 1, figsize=(14, 11), sharex=True, sharey=True,
                                 squeeze=False)
        for ax, (bus, phases) in zip(axes[:, 0], voltages.items()):
            for phase in phases:
                ax.plot(window.index, window[(bus, phase)], color=PHASE_COLORS[phase],
                        linewidth=0.6, label=f"Phase {phase.upper()}")
            ax.set_ylabel(f"{bus}\nVoltage [pu]")
            ax.set_ylim(lower - margin, upper + margin)
            ax.ticklabel_format(axis="y", style="plain", useOffset=False)
            ax.grid(alpha=0.25)
        axes[0, 0].legend(loc="lower left", ncol=3)
        axes[-1, 0].set_xlabel("Time [UTC]")
        fig.suptitle("Three-phase bus voltages — " +
                     ("complete series" if name.endswith("series") else str(day_start.date())) +
                     "\nConfigured voltage limits: 0.95–1.05 pu")
        fig.tight_layout()
        fig.savefig(OUTPUT / f"{name}.png", dpi=160)
        plt.close(fig)
    print(stats.to_string(float_format=lambda value: f"{value:.6f}"))


def main():
    if BLOCK_STEPS < 1:
        raise ValueError("BLOCK_STEPS must be positive")
    env = MicrogridEnv(EnvironmentConfig(CASE, episode_steps=EPISODE_STEPS, start_step=START_STEP))
    total_steps = env.episode_steps
    if (env.bess_list or env.pv_list or env.evse_list) and total_steps > BLOCK_STEPS:
        raise ValueError("Independent blocks require a loads-only case")
    env.episode_steps = min(BLOCK_STEPS, total_steps)
    env.num_episodes = (total_steps + BLOCK_STEPS - 1) // BLOCK_STEPS
    blocks = []
    violated_steps = 0
    print(f"Running {total_steps} steps in blocks of {BLOCK_STEPS}", flush=True)
    for offset in range(0, total_steps, BLOCK_STEPS):
        count = min(BLOCK_STEPS, total_steps - offset)
        if count < env.episode_steps:
            # Only an incomplete final block needs a differently sized environment.
            env = MicrogridEnv(EnvironmentConfig(CASE, episode_steps=count, start_step=START_STEP + offset))
            episode_idx = 0
        else:
            episode_idx = offset // BLOCK_STEPS
        env.reset(seed=SEED, options={"episode_idx": episode_idx})
        for _ in range(count):
            action = None
            if CONTROLLED_EV_POWER_PU is not None and env.evse_list:
                action = Action(evse={
                    evse.id: EVSEAction(CONTROLLED_EV_POWER_PU)
                    for evse in env.evse_list if evse.controlled
                })
            _, _, _, _, info = env.step(action)
            violated_steps += not info["diagnostics"]["within_limits"]
        blocks.append(env.get_episode_results())
        if len(blocks) % 30 == 0 or offset + count == total_steps:
            print(f"Completed {offset + count}/{total_steps} steps", flush=True)

    results = combine_results(blocks)
    save_summary(results, OUTPUT)
    plot_voltages(results)
    print(f"Completed {total_steps} steps; electrical violations in {violated_steps} steps.")
    print(f"Results: {OUTPUT.resolve()}")
    return results


if __name__ == "__main__":
    main()
