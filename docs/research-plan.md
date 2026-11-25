# BESS Grid Control

Historical research plan. For current implementation and measurements, see [SAC progress](sac-progress.md).

**Reinforcement Learning for Three-Phase Battery Energy Storage Systems in Active Distribution Networks**

## Overview

This repository develops and evaluates control strategies for Battery Energy Storage Systems (BESS) operating in three-phase electrical distribution networks.

The primary research focus is **reinforcement learning (RL)** for local BESS control, considering limited observability, electrical constraints, and multiple operational objectives.

The project uses [powerflow-opendss-environment](https://github.com/TeradaZenichi/powerflow-opendss-environment) as its common simulation and evaluation platform.

The resulting controllers are intended to be reusable components in future nodal and hierarchical energy management architectures.

**Project status:** Initial research and development. Local rule-based BESS control, neural observation encoding and SAC actor/critic networks are implemented. SAC training and HPO are commented scaffolds; trained neural policies and comparative RL results are planned.

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
See the [complete network diagram](network-diagram.md) for the active
topology, device/phase placement and the implemented MV/LV transformers.
Requires Python 3.11+ and Git. From this repository root:

```powershell
git submodule update --init --recursive
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

[bess-edge-controller](https://github.com/ray-aguiar/bess-edge-controller) is an
implementation reference, not a dependency. See the
[repository analysis](repository-analysis.md) for reusable patterns,
three-phase adaptation requirements and the simulator contract.

Run one day of the realistic five-bus scenario with a 100 kWh / 40 kW BESS at
`bus_004_lv`, ASAP EVSEs and a fixed 200 kW PV at `bus_005` with:

```powershell
.\.venv\Scripts\python.exe main.py
```

Conventional loads and EVSEs are on 220 V phase-neutral LV buses behind four
distribution transformers. The BESS shares bus 004's LV aggregator; PV connects
at 11 kV on bus 005, outside that aggregator. To evaluate this network
with BESS P=Q=0, run `python -m tools.run_idle_bess`; outputs are saved in
`results/mv_lv_idle_bess_day/`.

Configure `EPISODE_STEPS`, `START_STEP`, `SEED` and `OUTPUT` at the top of
`main.py`. Its loop contains only environment iteration and the controller call;
reporting and impact analysis live in `tools/`. Results are written to
`results/5bus_realistic_ev_pv/`.

Time-series plots use post-step curves. Dispatched BESS active/reactive power
uses bars aligned with each dispatch interval. Boxplots and aggregate comparisons
retain their own chart types.

Only active inputs remain visible in `data/EV/` and `data/switzerland/`.
Generators and metadata are grouped in `data/EV/support/`; other datasets and
historical cases are in `data/old/`. See the [data index](../data/README.md).

All experiments will use the existing OpenDSS-based simulation framework.

The environment currently supports three-phase BESS models, device-level operating limits, total or per-phase power commands, and electrical measurements.

The controller will interact with the simulator through its supported observation and action interfaces.

Important conventions:

- BESS and EVSE net active power is positive when charging and negative when discharging.
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

Local BESS implementations belong in [controllers/](../controllers/README.md).
Each controller type groups its scripts and Python configuration in one
subdirectory. Neural architectures are variants within each algorithm,
independent of operating modes. Documentation is centralized in
`controllers/README.md`. SAC networks are implemented and training/HPO are scaffolds; trained neural policies and the mode selector are planned;
EVSEs remain in ASAP mode.

The controller contract has four components.

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

- `controllers/` — Local BESS policies, common interface and per-controller documentation.
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
