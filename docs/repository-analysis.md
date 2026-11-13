# Simulation integration and reference controller

Reviewed revisions:

- `bess-edge-controller`: `fab105c1ba226b7180cae9d268f46e191a4a6ab9`.
- `powerflow-opendss-environment`: `b4f45abf9f60c98fee13564002e8e5de0765bdb6`.

## Reference implementation

[bess-edge-controller](https://github.com/ray-aguiar/bess-edge-controller)
keeps OpenDSS and an OPF teacher as Git submodules. Its current `train.py`
collects local, pre-action teacher histories and trains an MLP or LSTM through
behavior cloning. `tools/evaluation.py` runs the learned policy in OpenDSS,
padding initial history and recording requested actions and execution feedback.
The presence of other algorithm files does not establish a validated RL pipeline.

Useful patterns are explicit feature order, shared training/inference encoding,
seeded runs, saved configuration and closed-loop evaluation. Adaptation is needed:

- `tools/features.py::local_observation` selects phase `a`; our observations must
  explicitly select all relevant phases and permitted local measurements.
- The model outputs two aggregate values (`p_net_pu`, `q_injection_pu`). Per-phase
  control requires a declared phase order and converter-compatible action space.
- Training uses a random split of overlapping temporal windows. Our evaluation
  should separate chronological periods/scenarios before constructing windows.
- Its data loading and feature helpers depend on `opf_teacher`. Those helpers
  should not become mandatory dependencies of our RL pipeline.

## Adopted boundary

Only the simulator is a submodule, at `opendss_environment/`. The reference
controller remains an external reference. Its pinned simulator revision matches
the simulator revision adopted here. Editable installation exposes the inner
Python package; do not add the inner package directory to `sys.path`.

The simulator requires Python >=3.11. Its `pyproject.toml` declares numerical,
Gymnasium and OpenDSS dependencies, plus `pu-normalizer` from a pinned subdirectory
of `opf-teacher`. This is a normalization dependency, not a required OPF solver
or teacher training pipeline. Transitive Python dependencies are range-constrained,
not fully locked by the Git submodule revision.

## Runtime contract

- Construct `MicrogridEnv(EnvironmentConfig(case_path, episode_steps=...))`.
- `reset()` returns a pre-action named observation and info; `step(action)` returns
  observation, reward, terminated, truncated and info.
- BESS `p_net_pu > 0` charges; `q_injection_pu > 0` injects reactive power.
  Powers use the configured total network base, including per-phase entries.
- Observations expose phase-indexed bus measurements and one shared BESS SoC.
- Log the requested action separately from `info["executed_action"]` and
  `info["diagnostics"]`; device clipping does not guarantee network feasibility.
- The named environment returns `None` at termination. `GymAdapter` provides
  numeric Box spaces and a terminal numeric observation for RL integrations.
- Pass explicit local `observation_paths` and controlled `devices` to `GymAdapter`;
  its default observation selection exposes the whole network.

## Next integration step

Define the local three-phase observation/action contract, then validate reset,
charging/discharging, reactive injection, shared SoC and requested versus executed
powers in a short closed-loop run. The upstream three-phase fixtures can support
contract checks; they are not yet our research scenarios. Reward design, baseline
controllers and RL training remain project work.

## Validation

Installed the editable simulator in the project `.venv` with Python 3.14.4.
All 61 upstream tests passed, including three-phase dispatch, energy balance,
reactive-power signs, observation timing and the Gymnasium adapter:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s opendss_environment/tests -v
```

Gymnasium reports warnings for unbounded observation spaces, which the simulator
intentionally uses to represent out-of-limit measurements. This validates the
upstream simulator installation, not a trained controller or research results.
