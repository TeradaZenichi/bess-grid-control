"""Compare full-copy protection with checkpoints, then time a continuous year.

Run after `python -m tools.compare_bess`, using `python -m tools.benchmark_environment`.
"""
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from controllers import Controller
from opendss_environment import Action, EnvironmentConfig, MicrogridEnv, PhaseAction, TotalAction


ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "results/bess_comparison_pv/with_bess/config.json"
OUTPUT = ROOT / "results/environment_benchmark"
EPISODE_STEPS = 365 * 96
REFERENCE_STEPS = 8
START_STEP = 0
SEED = 42


class FullCopyEnv(MicrogridEnv):
    """Original device/history copy and commit, retained only for measurement."""

    def _copy_for_step(self):
        pending = super()._copy_for_step()
        pending.bess_list, pending.pv_list, pending.evse_list, pending.grid, pending.results = deepcopy(
            (self.bess_list, self.pv_list, self.evse_list, self.grid, self.results)
        )
        return pending

    def _commit_step(self, pending):
        for devices, staged in ((self.bess_list, pending.bess_list),
                                (self.pv_list, pending.pv_list), (self.evse_list, pending.evse_list)):
            for device, updated in zip(devices, staged):
                device.commit(updated)
        self.grid.array_kw[:] = pending.grid.array_kw
        self.grid.array_kvar[:] = pending.grid.array_kvar
        self.results.commit(pending.results)
        super()._commit_step(pending)


def run(env_type, count, expected=None):
    started = perf_counter()
    env = env_type(EnvironmentConfig(CASE, EPISODE_STEPS, start_step=START_STEP))
    observation, _ = env.reset(seed=SEED)
    policies = [Controller(parameters) for parameters in env.data["devices"].get("bess", [])]
    setup_seconds = perf_counter() - started
    step_times, reference = [], []
    started = perf_counter()
    try:
        for index in range(count):
            commands = {}
            for policy in policies:
                local = env.observe_controller(policy.bus, {"bess": [policy.id]},
                                               extra_features=policy.extra_features, observation=observation)
                commands[policy.id] = policy.act(local, {})
            action = Action(bess=commands, pv={
                device.id: PhaseAction(observation["pv"][device.id]["available_pu"])
                if device.dispatch_mode == "per_phase"
                else TotalAction(sum(observation["pv"][device.id]["available_pu"].values()))
                for device in env.pv_list
            }) if policies else None
            tick = perf_counter()
            transition = env.step(action)
            step_times.append(perf_counter() - tick)
            observation, _, terminated, truncated, _ = transition
            if index < REFERENCE_STEPS:
                if expected is not None:
                    np.testing.assert_equal(transition, expected[index])
                else:
                    reference.append(deepcopy(transition))
            if (index + 1) % 3504 == 0:
                print(f"{env_type.__name__}: {index + 1}/{count}; {perf_counter() - started:.1f} s", flush=True)
            if terminated or truncated:
                break
        loop_seconds = perf_counter() - started
        metrics = {
            "steps": env.idx, "setup_seconds": setup_seconds, "loop_seconds": loop_seconds,
            "mean_step_ms": 1000 * float(np.mean(step_times)),
            "reference_mean_step_ms": 1000 * float(np.mean(step_times[:REFERENCE_STEPS])),
            "evse_count": len(env.evse_list), "session_count": sum(len(d.sessions) for d in env.evse_list),
            "cost": float(sum(env.results.costs)), "final_bess_soc": {d.id: d.soc for d in env.bess_list},
            "ev_delivered_kwh": sum(d.delivered_kwh for d in env.evse_list),
            "ev_unmet_kwh": sum(d.unmet_kwh for d in env.evse_list),
        }
        print(f"{env_type.__name__}: {json.dumps(metrics)}", flush=True)
        return metrics, reference
    finally:
        env.close()


def main():
    original, expected = run(FullCopyEnv, REFERENCE_STEPS)
    optimized, _ = run(MicrogridEnv, EPISODE_STEPS, expected)
    summary = {
        "case": str(CASE.relative_to(ROOT)), "episode_steps": EPISODE_STEPS,
        "seed": SEED, "start_step": START_STEP, "equivalent_prefix_steps": len(expected),
        "original": original, "optimized": optimized,
        "prefix_speedup": original["reference_mean_step_ms"] / optimized["reference_mean_step_ms"],
        "notes": ["Both variants use the same physical equations, actions, episode length and seed.",
                  "The full-copy reference also creates the new small checkpoint; this adds minor overhead.",
                  "Step times exclude controller work; loop time includes controllers and prefix comparisons.",
                  "Setup time excludes importing Python packages; no plots or CSV exports are timed."],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "benchmark.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Prefix speedup: {summary['prefix_speedup']:.1f}x; report: {OUTPUT / 'benchmark.json'}")


if __name__ == "__main__":
    main()
