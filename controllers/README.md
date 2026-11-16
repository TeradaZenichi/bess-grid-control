# Local BESS controllers

Local BESS policies only; EVSEs remain ASAP. Physics and rewards stay in the
environment submodule. The interface and configurations exist; policies,
training and the mode selector are planned. The active case still needs a BESS.

## Organization

| Type | Role | Configuration | Status |
|---|---|---|---|
| `rule_based` | Deterministic local baseline | [config.py](rule_based/config.py) | Planned |
| `sac` | RL with selectable architectures | [config.py](sac/config.py) | Planned |

Each type groups its code and configuration. This README is the single
controller documentation file.

```text
controllers/
    base.py
    rule_based/config.py
    sac/config.py
    sac/architectures/mlp/config.py
    sac/architectures/gru/config.py
```

Add `controller.py`, `train.py` and `evaluate.py` inside the relevant type when
implemented; architecture-specific network code belongs under `architectures/`.

## Configuration

- Edit Python globals in `config.py`; no argument parser.
- `BESS_ID`, `BUS`, `EXTRA_FEATURES`: controlled device and local observation scope.
  Set the device and bus after adding a BESS to the scenario.
- SAC `ARCHITECTURE`: `mlp` or `gru`, independent of `MODE`.
- [MLP settings](sac/architectures/mlp/config.py): layer sizes and history length.
- [GRU settings](sac/architectures/gru/config.py): hidden size and history length.
- SAC learning rate, discount, batch size and seed stay in `sac/config.py`.

Values are initial proposals, not tuned settings. GRU requires sequence replay,
warm-up and recurrent-state handling; selecting a config does not implement them.

| `MODE` | Main objective |
|---|---|
| `economic` | Energy cost and throughput |
| `voltage` | Voltage-band compliance and reference regulation |
| `balance` | Local voltage unbalance |
| `tracking` | Requested P/Q reference |

For example, SAC + GRU + economic and SAC + MLP + economic compare architectures
for one objective. Each architecture/mode can have separately trained weights.

## Interface

[base.py](base.py) defines:

```python
controller.reset()
command = controller.act(local_observation, objective)
```

- Input: filtered pre-action `env.observe_controller()` data and current
  directives. No global state or future EV sessions enter the local policy.
- Output: one `TotalAction` or `PhaseAction`, in pu on the total network base.
  P > 0 charges; Q > 0 injects. The caller builds `Action` and steps the environment.
- Reset policy memory at episode start. Switching modes preserves physical SoC
  and requires current history for an activated recurrent expert.

Device limits apply in every mode; per-phase control requires compatible hardware.
Device feasibility does not guarantee network-wide safety. Tracking reward Q is
positive for import, opposite to action Q; convert references explicitly.

## Recording a controller

Add its entry and details here: implementation status, observation order/units,
actions, reward methods/scopes/weights/scales, architecture/history handling,
training configuration, reproduction command and measured results. Keep code
and this README aligned. Use planned, implemented, trained or validated accurately.

Artifacts belong in `results/controllers/<type>/<architecture>/<mode>/<run>/`
(ignored by Git); use `default` for non-neural architectures. Record scenario,
data split, seeds, code versions and checkpoint identity. Report differing
observation/history budgets and distinguish discounted training returns from
undiscounted economic scores. The teacher is optional for comparison or initialization.
