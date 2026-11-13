# BESS Grid Control

**Reinforcement Learning for Three-Phase Battery Energy Storage Systems in Active Distribution Networks**

## Overview

This repository develops and evaluates control strategies for Battery Energy Storage Systems (BESS) operating in three-phase electrical distribution networks.

The primary research focus is **reinforcement learning (RL)** for local BESS control, considering limited observability, electrical constraints, and multiple operational objectives.

The project uses [powerflow-opendss-environment](https://github.com/TeradaZenichi/powerflow-opendss-environment) as its common simulation and evaluation platform.

The resulting controllers are intended to be reusable components in future nodal and hierarchical energy management architectures.

**Project status:** Initial research and development. Training pipelines, controller implementations, and experimental results are not yet available in this repository.

## Research Objective

The central research question is:

**Can a locally observable RL-based BESS controller provide effective and adaptable grid services in three-phase distribution networks while respecting device constraints and supporting coordination with higher-level controllers?**

The investigation focuses on:

- Active and reactive power management.
- Voltage regulation and phase-unbalance mitigation.
- Energy cost reduction, peak shaving, and grid support.
- Adaptation to different operating objectives and supervisory signals.
- Generalization across operating conditions and network configurations.
- Safe and reproducible closed-loop operation.

## Simulation Environment

The simulator is tracked as a Git submodule in `opendss_environment/`.
Requires Python 3.11+ and Git. From this repository root:

```powershell
git submodule update --init --recursive
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

[bess-edge-controller](https://github.com/ray-aguiar/bess-edge-controller) is an
implementation reference, not a dependency. See the
[repository analysis](docs/repository-analysis.md) for reusable patterns,
three-phase adaptation requirements and the simulator contract.

Run the five-bus unbalanced load scenario with:

```powershell
.\.venv\Scripts\python.exe main.py
```

Configure `EPISODE_STEPS`, `START_STEP`, `SEED` and `OUTPUT` at the top of
`main.py`. `EPISODE_STEPS=None` runs the complete remaining series; `96` runs
one day. This scenario contains loads only, with BESS and PV disabled.
`BLOCK_STEPS=96` splits the execution into independent daily blocks to keep
simulation history copies small. This is restricted to the loads-only scenario;
controllers and storage require continuity of their internal state.
The same environment and loaded data are reused across full blocks. An incomplete
final block creates one additional environment.
Results in `results/5bus_unbalanced/` include phase measurements, electrical
diagnostics, `voltage_statistics.csv`, `voltage_full_series.png` and
`voltage_worst_day.png`. The latter shows the UTC day containing the lowest
bus-phase voltage in the executed period. Both plots share a voltage scale
across buses and retain the original 15-minute samples.

All experiments will use the existing OpenDSS-based simulation framework.

The environment currently supports three-phase BESS models, device-level operating limits, total or per-phase power commands, and electrical measurements.

The controller will interact with the simulator through its supported observation and action interfaces.

Important conventions:

- BESS active power is positive when charging and negative when discharging.
- Reactive power is positive for injection.
- Public controller actions use per-unit values on the configured network base.
- BESS state of charge is shared across connected phases.
- Per-phase dispatch must be consistent with the physical capabilities of the modeled converter.

The simulator provides electrical-limit diagnostics, but these do not constitute a network-wide safety guarantee.

## Control Strategies

The following approaches are candidates for implementation and comparison.

| Approach | Role |
|---|---|
| Rule-based control | Reference operating strategy |
| Model Predictive Control | Optimization-based benchmark |
| Reinforcement Learning | Main research approach |
| Imitation Learning | Optional initialization or experimental comparison |

Soft Actor-Critic (SAC) is an initial RL candidate. Neural network architectures, observation histories, and training procedures will be selected through experiments rather than fixed in advance.

The optional [opf-teacher](https://github.com/TeradaZenichi/opf-teacher) may support optimization benchmarks or demonstrations. Its use is not a prerequisite for training.

## Controller Design

The proposed BESS controller will expose a reusable interface with four components.

**Observations:** Local voltage measurements, BESS state of charge, previous operating conditions, available operating limits, and other explicitly permitted measurements.

**Supervisory directives:** Optional operating modes, objective weights, power references, or operating envelopes received from a higher-level coordinator.

**Actions:** Requested active and reactive power commands, represented as aggregate or per-phase commands where physically appropriate.

**Feedback:** Executed actions, operational limits, saturation information, and safety-related indicators.

The interface should support both learned policies and conventional controllers without requiring changes to the underlying simulator.

## Experimental Methodology

Experiments will begin with one BESS in a small three-phase distribution network, progressively introducing:

1. Unbalanced loading and asymmetric operating conditions.
2. Multiple operational objectives.
3. Different levels of local observability.
4. Previously unseen demand, price, and network scenarios.
5. Supervisory directives from external coordinators.

Evaluation will be performed in closed loop using OpenDSS.

### Evaluation Metrics

- Operating cost and energy exchange.
- Voltage deviations and constraint violations.
- Voltage unbalance.
- Branch and transformer loading.
- BESS state of charge and energy throughput.
- Tracking error for supervisory references.
- Control feasibility and intervention frequency.
- Training convergence, computational effort, and inference latency.
- Generalization to unseen scenarios.

Comparisons should use consistent scenarios, physical limits, evaluation horizons, and documented random seeds.

## Safety and Physical Feasibility

Physical feasibility is a core requirement.

The project will distinguish between:

- Device-level feasibility, including SoC, power, and inverter limits.
- Network-level constraints, including bus voltages and equipment loading.
- Safety mechanisms applied before execution.
- Violations detected after executing an action.

Device-level clipping alone does not guarantee network-wide safety.

Proposed safety mechanisms must be implemented, tested, and evaluated explicitly. No formal safety guarantee is assumed at this stage.

## Planned Repository Organization

The following structure is proposed and will evolve as implementation progresses.

- `src/bess_grid_control/` — Controllers, policies, training, and evaluation utilities.
- `configs/` — Experimental and algorithm configurations.
- `experiments/` — Research experiments and ablation studies.
- `tests/` — Unit, integration, and physical-consistency tests.
- `results/` — Generated outputs, excluded from version control when appropriate.
- `docs/` — Modeling assumptions, methodological decisions, and experiment documentation.

## Development Roadmap

- [ ] **M0 — Simulation integration:** Validate observations, P/Q commands, physical units, and three-phase test cases.
- [ ] **M1 — Baselines:** Implement and validate rule-based control and an optimization-based comparator.
- [ ] **M2 — RL controller:** Implement a first trainable policy and reproducible training pipeline.
- [ ] **M3 — Experimental evaluation:** Compare strategies, observability settings, objectives, and operating scenarios.
- [ ] **M4 — Controller export:** Package a validated controller for integration with nodal and global coordinators.
- [ ] **M5 — Scientific publication:** Consolidate the independent research contribution, experiments, and reproducibility materials.

These milestones describe intended work, not completed functionality or fixed deadlines.

## Reproducibility

Each experimental release should include the source-code version, environment version, configuration files, dataset partitioning, random seeds, model checkpoints when applicable, and scripts necessary to reproduce the reported results.

Exported neural controllers must preserve their architecture, preprocessing, normalization, observation schema, action convention, operating assumptions, and safety interface.

## Relationship to Other Projects

This repository represents the **T1 — BESS Control** research work within a broader program on hierarchical control of distributed energy resources.

Its controllers are intended to be reusable by subsequent research projects addressing:

- Coordination of multiple devices at a common network node.
- Distribution-grid-level coordination.
- Recursive hierarchical control.
- Hierarchical safe reinforcement learning.

This repository focuses on **local BESS control**. The development of nodal coordinators, global coordinators, and general hierarchical training methods belongs to separate research projects.

## Research Context

This work is part of doctoral research in Electrical Engineering at the University of Campinas (UNICAMP), in collaboration with INESC TEC and the University of Porto, Portugal.

The research aims to contribute to adaptive, physically informed, and reusable control strategies for distributed energy resources.
