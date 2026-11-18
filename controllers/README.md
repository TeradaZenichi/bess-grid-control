# Local BESS controllers

Local BESS policies only; EVSEs remain ASAP. Physics and rewards stay in the
environment submodule. Rule-based control, local observation encoding and MLP/GRU
networks, a Gaussian actor and twin critics are implemented. SAC training and
HPO are commented scaffolds; the automatic mode selector is planned. The active BESS
is `b1` at `bus_004_lv`.

## Organization

| Type | Role | Configuration | Status |
|---|---|---|---|
| `rule_based` | Deterministic local baseline | [config.json](rule_based/config.json) | Implemented; automated tests |
| `sac` | RL with selectable architectures | [config.json](sac/config.json) | Actor and twin critics implemented; training/HPO scaffold |

Each type groups its code and configuration. This README is the single
controller documentation file.

```text
controllers/
    __init__.py
    base.py
    config.json
    normalization.py
    rule_based/controller.py
    rule_based/config.json
    sac/controller.py
    sac/__init__.py  # SAC Config: common, algorithm and architecture values
    sac/config.json
    sac/observation.py
    sac/networks.py
    sac/train.py
    sac/architectures/__init__.py
    sac/architectures/mlp/__init__.py
    sac/architectures/mlp/config.json
    sac/architectures/gru/__init__.py
    sac/architectures/gru/config.json
```

Policies live in each type's `controller.py`; `train.py` contains `SACTrainer`.
Add `evaluate.py` when needed. Architecture-specific network code belongs under
`architectures/`.

Each architecture package defines its class in `__init__.py`. The common
`ARCHITECTURES` dictionary selects the class used by `SACController`:

```python
from pathlib import Path
from controllers.base import Config
from controllers.sac.architectures import MLP, GRU

folder = Path("controllers/sac/architectures")
mlp_config = Config(folder / "mlp/config.json")
gru_config = Config(folder / "gru/config.json")
mlp = MLP(input_size=264, output_size=6, config=mlp_config)
gru = GRU(input_size=32, output_size=6, config=gru_config)
```

MLP accepts tensors shaped `(batch, 264)`. GRU accepts `(batch, time, 32)`
and an optional boolean `(batch, time)` mask, skipping padded steps. Both return
`(batch, output_size)` with unrestricted outputs. `networks.py` wraps these
architectures in the SAC actor and critics described below.

## Configuration

- Edit the JSON values; no argument parser. `Config` in `base.py` reads JSON
  values into a `SimpleNamespace`, accessed as `config.mode` or `config.hidden_sizes`.
  `Config.controller("sac")` merges common values with the SAC-specific settings;
  explicit per-run overrides take precedence and do not change the defaults.
- Select `controller_type` in [config.json](config.json): `rule_based` or `sac`.
  This selection is read at package import; restart the process after changing it.
  Shared values also live here: `architecture`, `mode`, `bess_id`, `bus`,
  `extra_features`, `observation_timezone`, `seed`, `case_path` and `reward_term`.
  `Config` resolves `case_path` against the repository root on loading; the JSON
  keeps a relative path, while the object stores an absolute string for JSON metadata.
- `bess_id`, `bus`, `extra_features`: controlled device and local observation scope.
  Set `bess_id` explicitly (`b1` in the active case); training selects that ID
  without falling back to the first BESS. `bus=null` uses the device's bus.
- SAC `architecture`: `mlp` or `gru`, independent of `mode`.
- [MLP settings](sac/architectures/mlp/config.json): layer sizes and history length.
- [GRU settings](sac/architectures/gru/config.json): hidden size and history length.
- SAC learning rate, discount, batch/replay sizes and episode length stay in
  `sac/config.json`. Rule thresholds stay in `rule_based/config.json`.

Training output is derived from the effective settings, for example:
`results/controllers/switzerland_5bus/sac/economic/node_cost/mlp/trial_003/seed_42/`.
Call `trainer.hpo(trial="trial_003")` to name a trial; by default its ID contains
the UTC time. Different rewards, architectures, trials and seeds have separate
folders. An existing training folder is not overwritten. Rule-based runs omit
the architecture level. `main.py` uses the same layout with `simulation` as the
run ID, unless its `OUTPUT` global is set explicitly.

Network/learning values are initial proposals, not tuned settings. Both input
encoders use eight measured steps. GRU sequence replay and recurrent policy
state remain part of the planned training implementation.

| `mode` | Main objective |
|---|---|
| `economic` | Energy cost and throughput |
| `voltage` | Voltage-band compliance and reference regulation |
| `balance` | Local voltage unbalance |
| `tracking` | Requested P/Q reference |

For example, SAC + GRU + economic and SAC + MLP + economic compare architectures
for one objective. Each architecture/mode can have separately trained weights.

## Interface

[base.py](base.py) defines `BaseController`, the abstract inference contract.
The public class in [__init__.py](__init__.py) inherits the selected implementation:

```text
BaseController -> RuleBasedController -> Controller   (rule_based)
BaseController -> SACController       -> Controller   (sac)
```

Callers use the same import for either type:

```python
from controllers import Controller

controller = Controller(bess_parameters)
controller.reset()
command = controller.act(local_observation, objective)
```

For rule-based control, `bess_parameters` is one BESS record from `devices.json`.
Only static device parameters are passed, not the environment or its live state.
`Controller(bess_parameters, settings={"mode": "voltage"})` overrides JSON values
for one instance without modifying the file or other controllers. The instance exposes `parameters`, `config`, `id`,
`bus` and `extra_features`. SAC additionally requires `scaling=scaling`; its actor
starts with random weights and is not a trained control policy.
Settings use lowercase names. `vars(controller.config)` provides a dictionary
for saving experiment metadata. SAC also accepts `architecture_settings`, such
as `{"hidden_sizes": [128, 64], "history_steps": 16}`; the resulting object is
merged into `controller.config`, which is passed to the network.
`controller.architecture_config` remains an alias for the same object.

`main.py` creates a local policy for each configured BESS and follows available
PV generation without optimization. With no BESS it retains the original ASAP
simulation. EVSEs remain ASAP; adding a BESS is a separate scenario choice.

- Input: filtered pre-action `env.observe_controller()` data and current
  directives. No global state or future EV sessions enter the local policy.
- Output: one `TotalAction` or `PhaseAction`, in pu on the total network base.
  P > 0 charges; Q > 0 injects. The caller builds `Action` and steps the environment.
- Reset policy memory at episode start. Switching modes preserves physical SoC
  and requires current history for an activated recurrent expert.

Device limits apply in every mode; per-phase control requires compatible hardware.
Device feasibility does not guarantee network-wide safety. Tracking reward Q is
positive for import, opposite to action Q; convert references explicitly.

## SAC local observations

[observation.py](sac/observation.py) defines a fixed order in `FEATURE_NAMES`:

| Features | Count | Representation |
|---|---:|---|
| Phase voltage deviations | 3 | `(V_pu - 1) / voltage_deviation_ref_pu` |
| Relative B-A and C-A voltage angles | 4 | Sine/cosine; invariant to the absolute phase reference |
| Local meter phase P/Q | 6 | Positive import, divided by meter apparent rating per phase |
| Phase current loading | 3 | Fraction of local terminal normal current rating |
| Local VUF | 1 | Negative/positive sequence voltage percentage divided by `vuf_ref_pct` |
| BESS measured SoC | 1 | Fraction |
| Previous executed phase P/Q | 6 | P positive charging; Q positive injection, divided by inverter rating per phase |
| Current buy/sell prices | 2 | Both divided by `price_ref_per_kwh` |
| Daily, weekly and annual calendar | 6 | Sine/cosine in `observation_timezone` |
| **Total** | **32** | Float32, without clipping |

The encoder expects one local three-phase branch terminal. At
`bus_004_lv` it is the secondary of `Transformer.t4`. Resolve its element and
terminal once with `ObservationScaling.from_environment()` after reset. Pass
`meter_element` explicitly there when multiple branches are present.
Inputs follow the environment contract without additional defensive validation.
Raw conventional-load P/Q from the CSV, remote device states, rewards and feeder
power are not encoder inputs. Local flows already reflect the aggregate load,
EVSEs and BESS. All samples are pre-action, measured with held device setpoints.

```python
from controllers.normalization import ObservationScaling
from controllers.sac.observation import LocalObservation

scaling = ObservationScaling.from_environment(env, "bus_004_lv")
bess = env.data["devices"]["bess"][0]
encoder = LocalObservation(bess, scaling, history_steps=8)
local = env.observe_controller("bus_004_lv", {"bess": ["b1"]},
                               extra_features=("grid", "branches"))
sequence = encoder.encode(local)               # history (8, 32), mask (8,)
flat = encoder.encode(local, flatten=True)      # 256 values + 8 validity flags
```

MLP inputs have **264 entries**, including the mask. GRU inputs expose a
dictionary with `history` and `mask`; the GRU skips invalid padded steps.
Histories are ordered oldest to newest, right-aligned,
zero-padded on the left, with a separate validity mask. Store these encoded
observations in replay rather than calling the live encoder on shuffled samples.

Call `reset()` at every episode start. Repeated timestamps replace the current
sample without extending history. Gaps and backwards time clear history. Naive timestamps
are interpreted as UTC; calendar encoding defaults to Europe/Zurich and handles
DST through UTC chronology. Returned arrays are private copies.

`SACController` initializes this encoder and exposes `observe()`, `reset()`,
`sample()` and `act()`. `act()` uses deterministic actor inference. MLP and
GRU share the same features/history budget. [normalization.py](normalization.py)
provides fixed scales shared by neural controllers; no mean/variance estimation
or clipping is applied. The scenario `observation_scaling` block supplies voltage,
VUF and price references. Transformer winding kVA and local voltage base come
from OpenDSS; line meters use phase count times nominal phase voltage times
normal current rating. BESS inverter kVA comes from its device parameters.

The environment's physical observations and rule-based thresholds stay unchanged.
Raw phase currents are omitted from the neural vector because loading already
expresses current relative to its fixed normal rating. Values are transformed
before entering history, preserving zero padding. Feature names use `scaled`
instead of claiming physical pu or currency units after this transformation.

Create a SAC implementation with `scaling=scaling`. The controller receives
static references, not the environment. Save `vars(scaling)`, the controlled
device parameters, feature order and architecture settings with checkpoints.
Restore scales with `ObservationScaling(**saved_values)` and use the same
physical pu contract and saved device references for inference. For this case the
references are 1000 kVA network base, 250 kVA T4 meter, 50 kVA inverter, 0.05 pu
voltage deviation, 2 percent VUF and 0.20 currency/kWh price. These scaling
references do not change operational limits. The encoder remains three-phase.

## SAC networks and training scaffold

[networks.py](sac/networks.py) defines the common roles; MLP and GRU remain
architecture classes. The actor produces Gaussian mean/log-standard-deviation
for each action, samples with reparameterization, and applies `tanh`. Its
log-probability includes the change-of-variables correction; log-standard-deviation
is bounded to [-20, 2]. Deterministic inference uses `tanh(mean)`.
Each critic has its own state encoder, then concatenates the current action and
uses a hidden ReLU layer plus one scalar output. Actor, Q1 and Q2 share no weights.
The initial MLP uses [64, 64]; GRU uses hidden size 64. The critic joint layer
uses that architecture's final hidden size. See the [SAC reference](https://spinningup.openai.com/en/latest/algorithms/sac.html).

For the active BESS, the action order is `[P_a, P_b, P_c, Q_a, Q_b, Q_c]`.
The actor operates in [-1, 1]; `controller.command()` maps to device P/Q bounds
in network pu. P is positive charging; Q is positive injection. These are box
bounds: the environment still applies joint apparent-power and SoC feasibility.
Replay stores requested normalized actions; next observations contain the
executed previous P/Q. Entropy/log-probabilities use normalized action coordinates.

[train.py](sac/train.py) also defines `ReplayBuffer`, a uniform circular buffer.
It keeps current/next feature vectors, normalized requested actions, reward and
separate termination/truncation flags in preallocated NumPy arrays. Values use
float32, flags use bool. It never stores full overlapping observation windows.
An extra `history_steps - 1` transitions preserve the prefix needed by the oldest
sampleable transition. `sample()` reconstructs causal windows and masks, returns
CPU tensors and flattens only for MLP. Episode IDs prevent mixing reset windows.

```python
self.replay = ReplayBuffer(self.config, self.initial_state, action_size)
# After each environment/encoder reset:
self.replay.start_episode()
# After collecting a transition:
self.replay.add(state, action, reward, next_state, terminated, truncated)
batch = self.replay.sample()  # config.batch_size; sampling with replacement
```

Collect from a freshly reset encoder; warm-up histories supplied before the first
stored transition are not retained. Insertion copies values. End flags also mark
an episode boundary automatically; explicit `start_episode()` handles early resets.
The buffer remains across episodes, but each training/trial creates a fresh one.
Evaluation transitions must never enter it. `len(buffer)` counts sampleable
transitions; call `sample()` only after collecting data. `memory_bytes` reports
array storage, excluding temporary batches and network memory. Persistent storage
scales with `(capacity + history_steps) * feature_count`, while batch memory grows
with `batch_size * history_steps * feature_count`.

A true terminal transition may use `next_state=None`; its next feature vector is
zero-filled and must be excluded from bootstrap using `terminated`. A time-limit
truncation requires a valid final next observation, never the reset observation.
The environment's current episode-end semantics still need alignment before SAC
updates are implemented. The training scaffold allocates an empty replay buffer;
it does not yet collect transitions or update networks.

[train.py](sac/train.py) defines `SACTrainer` with a training/HPO contract and
an architecture factory. Run `python train.py` through
the [root entry point](../train.py), which also supports VS Code's Run/Debug Python
File commands. Breakpoints can be placed in either file. The module command
`python -m controllers.sac.train` also works. It initializes the local observation,
actor/critics and two environments under `trainer.env.train` and `trainer.env.eval`.
The existing episode loop runs default device operation; it does not insert
transitions into replay or update weights. Both environments are closed at the end. It saves
`initialization.json` with the effective settings, architecture settings and
episode/validation plan in the derived output folder.

`SACTrainer(config=None)` accepts a SAC `Config` instance or creates one from the
JSON defaults. It merges controller, SAC and architecture values in `self.config`.
`train(*, trial=None)` uses that object directly. Change its attributes before
calling training; no per-trial config copies or JSON reloads occur in `train()`.
`config.network` selects the architecture class and `config.flatten` follows it.
Assigning a different `config.architecture` loads its defaults; assign tuned
history/layer values after selecting the architecture.
It returns `{"status": "episodes_ready", "score": None, "episodes": [...],
"validation": [...], "settings": {...}, "flatten": true}`, together
with the first timestamp and input/action dimensions. `trainer.initial_state`
contains the encoded first observation for debugging. After implementation, `score` will be a float:
mean validation reward to maximize, not training return. No score is fabricated
while the training loop remains a comment. Each run must create fresh
networks, targets, replay and optimizers, train for its allocated budget,
evaluate on validation periods/seeds, save the run and close the environment.
The requested normalized action belongs in replay; executed P/Q remains in
the next observation. Episode boundaries must handle termination/truncation
and clear the observation history correctly.

`hpo(*, trial=None)` currently calls `train()` and returns that result.
The root entry point stores it as `result` for inspection in the debugger.
Later, its objective will assign suggestions directly to `self.config`, call `train()`
for each trial/seed with comparable budgets, and select the best validation
score. Test periods/seeds remain separate. There is no HPO library or search
algorithm configured yet. Shared scenario/seed values and the SAC episode length
control initialization; reward/replay/optimizer settings remain reserved. The
default economic term is `node_cost`.
Switch to `feeder_cost` for the other experiment, preserving actor observations.

The scenario's `training_split` defines year 2023, Europe/Zurich time, and days
15-21 of each month for validation (`validation_end_day=22` is exclusive).
[episodes.py](episodes.py) samples one eligible local-midnight start per month,
shuffles the month order and uses the trial seed. Every training timestamp in
each window must belong to 2023 and fall outside those validation days.
Consecutive allowed days can span a month boundary. All twelve months are
represented in each planned cycle; later training will repeat and resample cycles.

`episode_steps=672` means 168 elapsed hours at 15-minute resolution.
`num_training_episodes=12` is the initial total training episode budget, editable
through `trainer.config.num_training_episodes`. The existing loop samples windows
with replacement; this count alone does not guarantee visiting all twelve months.

Near DST,
the local end hour can differ from midnight; the plan records the exact end.
This duration is separate from the network's eight-step observation history.
`EnvironmentConfig(start_step=i, episode_steps=n)` selects `[i, i+n)`.
The environment now also supports `env.reset(options={"start_step": i,
"episode_steps": n})`, with absolute indices and overrides for that reset only.
Reuse the training environment for subsequent windows and reset the observation
encoder separately. Validation should use another environment and encoder;
actor/critics/replay stay with training, and validation transitions never enter
replay. `trainer.validation` stores the twelve fixed windows, including
`start_step`, `end_step_exclusive` and `episode_steps`. They reuse the single
`trainer.env.eval`, initially configured for January's validation week.
`trainer.eval_observations` is separate from the training encoder and must be
reset between validation windows. The default architecture history remains eight
steps. The annual evaluator already creates its own environments. SAC updates
and periodic validation execution remain pending. Each training reset restores
the configured initial SoC. Window cutoffs and
SAC bootstrap handling must be implemented before collecting training updates.

## Evaluation by year

Run `python test.py` from the root, or use VS Code's Run/Debug Python File command
on that entry point. The `evaluation` key in [config.json](config.json) selects controller overrides,
the output root and year groups: **2023 is `train_eval` and
2024 is `test`**. These are whole-year inference runs, not weight updates.
The SAC monthly training/validation split is separate; the annual evaluator
still combines all of 2023 under `train_eval`.
The controller type follows `controllers/config.json`; the default remains
rule-based economic control. Scenario, seed and observation timezone come from
the shared controller config. SAC currently has no trained checkpoint loader,
so selecting it evaluates random initial weights and labels the report `untrained`.

[evaluate_controller.py](../tools/evaluate_controller.py) selects years in
Europe/Zurich time: 2023 has 35,040 quarter-hours and leap year 2024 has 35,136.
Each year creates a fresh environment/controller with configured initial SoC,
empty policy history and the same parameters/seed. SoC evolves continuously
within the year. PV follows its fixed profile; EVSEs remain ASAP. This prevents
carrying device or controller state from the training/evaluation period into test.
The 2024 PV profile is a synthetic repetition of the 2023 shape, as documented
in [data/README.md](../data/README.md).

Results use the same derived hierarchy under `results/evaluations/`, with
`evaluation_<UTC-run>` as the trial ID and `<group>/<year>/` after the seed:

- `summary.json` and `evaluation.json`: energy, costs, voltage metrics, SoC,
  exact intervals, effective settings, device parameters and scaling references.
- `sim_devices_timeseries.csv.gz`: BESS/EVSE/PV powers and device energy/SoC.
- `sim_buses_timeseries.csv.gz`: phase voltages/angles, loads and feeder flows/costs.
- `sim_diagnostics_timeseries.csv.gz`, `branches.csv.gz` and `node_costs.csv.gz`:
  violations, phase branch powers/currents/loading and each controlled node's bill.

`comparison.csv` compares all evaluated groups/years. CSV timestamps mark interval
starts; saved BESS SoC is measured at interval ends. Compare initial/final stored
energy alongside cost; no cyclic terminal condition is imposed. `step_limit=null`
runs complete years; set a positive limit for a short integration check, recorded
as `complete_year=false`. Programmatic calls can restrict evaluation to a group:
`evaluate(groups=("train_eval",))`, leaving the final test group untouched.

Build fresh networks directly when debugging, using dimensions from the encoder
and action contract (the following are the current eight-step/per-phase sizes):

```python
trainer = SACTrainer()
trainer.config.architecture = "mlp"
trainer.config.hidden_sizes = [128, 64]
trainer.config.actor_lr = 0.0001
actor, critics = trainer.architecture(264, 6)
result = trainer.hpo(trial="trial_003")
# After real training: the HPO trial objective is result["score"].
```

`architecture()` also initializes `log_alpha`, with `alpha = exp(log_alpha)`
exposed through `trainer.alpha`, and target entropy `-action_size` in normalized
action coordinates. `config.alpha` is the initial coefficient. Optimizers are
`trainer.optimizer.actor`, `.critic` (both online critics) and `.alpha` (log-alpha
only), with JSON learning rates `actor_lr`, `critic_lr` and `alpha_lr`. Targets
are excluded. Rebuilding the networks resets alpha and all optimizer states.
Losses and automatic temperature updates remain pending in the training loop.

The factory selects MLP/GRU through `ARCHITECTURES`, creates an independent actor,
and frozen independent copies under `trainer.target.actor`, `trainer.target.q1`
and `trainer.target.q2`. Target synchronization remains pending in the training loop.
It also creates the online twin critics,
Q1 and Q2, and returns them. Critics are available as `critics.q1`/`critics.q2`
or `trainer.critic.q1`/`trainer.critic.q2`. Calling it again replaces the trainer's
network references with fresh modules; no weights or replay are reused across trials.

## Rule-based control

The controller reads local phase voltages/angles, measured SoC, previous executed P/Q
and, for economic control, current/past buy prices and local branch flows. No future
EV information is used. BESS parameters and observations must describe the same
bus/phases. Wye connections are supported; delta is not implemented here.

| Mode | Rule |
|---|---|
| `economic` | Use causal price bands; arbitrage between `soc_recharge_start` and `soc_target`, preserving the lower buffer for voltage support |
| `voltage` | Reactive droop toward the soft band; active assistance only outside the hard voltage limits |
| `balance` | Reactive descent on squared VUF using phase magnitudes and angles; requires reactive-enabled three-phase independent dispatch |
| `tracking` | Follow `p_reference_pu` and `q_reference_pu` in action signs; scalars are device totals, dictionaries identify connected phases |

For voltage control, `e = [v_low_pu-V]+ - [V-v_high_pu]+` and
`Q = S_phase * q_gain * e`. Outside the hard limits, active assistance is
`P = -S_phase * p_gain * sign(e) * [abs(e)-p_assist_error]+`, all on the network pu base.
The default `p_assist_error=0.04` leaves routine soft-band regulation to Q.
Economic discharge is capped by current nodal import minus previous BESS power
when `allow_export=False`; this estimate cannot guarantee no export after
unobserved changes within the interval.

With `adaptive_prices=True`, price bands use lower-rank empirical 25th/75th
percentiles from the last 24 hours of regularly observed prices. Fixed
`price_low`/`price_high` values apply during warm-up, after observation gaps,
or with adaptation disabled. Repeated calls at one timestamp add no samples.
Future tariffs are not read. Arbitrage requires
`price_low < price_high * eta_charge * eta_discharge`; this checks nominal
conversion losses, not network losses or guaranteed realized profit.

All autonomous modes maintain an energy buffer. Recharge starts at 40% SoC
and continues toward 60%, with hysteresis and a 25%-of-rating power cap.
The power budget includes modeled reactive converter losses so charging can
reach the target and stop. Economic recovery waits for the low-price band;
voltage/balance prioritize availability whenever local voltage allows it.
Arbitrage stops at 40%; the 20%-40% band remains available for voltage support.
Tracking follows its explicit references without autonomous recharge.

Routine charging is bounded by
`P_charge_max * clip((min(V_phase)-0.97)/(1.00-0.97), 0, 1)`.
This weakest-phase limit applies after ramps, preventing healthy phases from
continuing tariff-driven charging while another phase needs undervoltage
support. Hard overvoltage support can still request charging on the affected
phase. These local rules do not guarantee network-wide limits.

For balance control, let `a=exp(j*2*pi/3)`, `z_phase=V_phase*exp(j*theta_phase)`,
`z1=(za+a*zb+a^2*zc)/3`, `z2=(za+a^2*zb+a*zc)/3`, and `f=|z2/z1|^2`.
The direction is
`g_phase=Re((conj(z2)*c_phase-f*conj(z1)*d_phase)*exp(j*theta_phase))/|z1|^2`,
where `c=(1,a^2,a)` and `d=(1,a,a^2)`.
Request `Q_phase=-2*S_phase*balance_gain*(g_phase-mean(g))`; the gain absorbs
the common sequence-transformation factor. This is a local gradient heuristic
assuming an inductive voltage response, not a solved feeder sensitivity model.
Projection and voltage support can change the requested direction.

Use `objective={"mode": "tracking", "p_reference_pu": -0.01,
"q_reference_pu": 0.005}`, for example. Positive reference Q requests injection;
negate it and multiply by the network power base for the reward's Q-import reference.
Supported directive keys are `mode`, `p_reference_pu` and `q_reference_pu`.

`allow_export=False` caps economic gross discharge to estimated non-BESS net
import at that node: local branch import minus previous BESS P. It requires
`branches` observations when discharging and is rechecked after optional ramps.
It does not estimate demand elsewhere or forecast future prices.

`max_ramp_frac` optionally limits each phase's P/Q change per step, as a fraction
of phase apparent rating. Outside `[v_min_pu, v_max_pu]`, local voltage support
overrides mode requests and ramps. Aggregate dispatch uses mean voltage for
its normal droop and the violating extreme for support; simultaneous high/low
phase violations produce an idle request because aggregate support is ambiguous.

Finally, a private one-step BESS model projects commands using current observed
SoC, phase/total power limits, inverter capability, efficiencies and reactive
losses. It does not advance OpenDSS or modify observations. Operational reserve
is protected without inventing energy when initial SoC is below reserve.
`soc_target` is an economic charging ceiling, not a terminal equality guarantee.
The environment still applies its physical limits to every executed command.

`last_requested_action` records the mode request before ramp/support/projection;
`last_action` records the returned command. `last_decision` records pre-action
SoC/minimum voltage, the recharge latch, price bands and charging power limit.
The latch indicates pending recovery, not necessarily positive executed P.
`reset()` clears these diagnostics and price history.
The policy uses previous measured actions, not hidden simulated state.
The balance rule uses phase angles as well as magnitudes; reduced VUF at every
step is not guaranteed because voltage support takes priority.
Thresholds/gains are initial proposals, not tuned network-safety guarantees.

Run verification from the repository root:
`python -m unittest discover -s tests -v`.

Run a reproducible with/without-BESS comparison with
`python -m tools.compare_bess`. Experimental battery parameters and episode
selection are globals in `tools/compare_bess.py`; source scenario files stay
unchanged. Results, generated case configurations and plots are saved in
`results/bess_comparison_pv/`. Compare initial/final SoC alongside grid cost:
the rule does not enforce a cyclic terminal state.

Run the annual baselines with `python -m tools.run_baselines`. Globals in
`tools/run_baselines.py` select the output, horizon, seed and worker count;
the battery is the same experimental device defined in `tools/compare_bess.py`.
The default is the complete 2023 local calendar year, with the active fixed PV
and separate runs for
`without_bess`, `economic`, `voltage` and `balance`. Tracking awaits supervisor
references. All runs use identical loads and ASAP EVSE sessions, with one reset
and continuous SoC. Current thresholds and gains are preserved without tuning.

New runs use `results/baselines/annual_2023_pv_v2/` for the comparison table/plot, source archive,
input hashes and run manifest. Each mode stores generated case configurations,
settings, metrics and compressed CSVs for devices, buses, branch flows,
diagnostics and evaluation; controlled modes also store phase requests and
commands. CSV timestamps identify interval starts; device SoC is measured at
interval ends. Compare final SoC as well as raw cost. Existing output folders
are preserved; select a new `OUTPUT` for a new experiment.

The completed baselines in `results/baselines/annual_2023/` were measured
**without PV** and remain unchanged. Their generated case configurations keep
that scenario. The completed evaluation with PV is saved separately in
`results/baselines/annual_2023_pv/`.

Measured 2023 baselines without PV (35,040 intervals, seed 42, one 100 kWh / 40 kW BESS at
`bus_004`; initial SoC 50%; the same 24 ASAP EVSEs):

| Mode | Raw energy cost | Minimum V [pu] | Maximum VUF [%] | Voltage violation hours | Final SoC |
|---|---:|---:|---:|---:|---:|
| Without BESS | 208487.5642 | 0.931498 | 2.456400 | 149.25 | — |
| Economic | 208456.2027 | 0.930618 | 2.363726 | 103.00 | 80% |
| Voltage | 208481.9063 | 0.931498 | 2.456400 | 149.25 | 20% |
| Balance | 208482.4910 | 0.931498 | 2.456400 | 149.00 | 20% |

Costs use the configured tariff currency. Violation hours count intervals with
at least one bus/phase outside 0.95–1.05 pu, once per interval. Loads and EV
profiles match exactly across runs. Saved exports also passed timestamp,
cost, voltage, branch-row, SoC-bound and battery-energy-balance checks.

In the original fixed-band policy, the 2023 tariff range is 0.1505–0.1905/kWh,
below `price_high=0.23`. Its economic discharges are voltage-support overrides, rather than
high-price triggers. This reduces violation duration but does not improve
the worst instantaneous voltage. Voltage and balance reach the operational
20% reserve on January 5 and 9 (UTC), respectively, and remain idle thereafter;
neither mode replenishes energy in this case. Their small raw cost reductions
include using initial stored energy. These are baselines of the current rules,
not tuned policies or evidence of year-long voltage regulation.

Measured 2023 baselines with the fixed 200 kW / 200 kVA PV at `bus_005`,
using the same BESS, rules, loads, EV sessions, seed and episode horizon:

| Mode | Raw energy cost | Minimum V [pu] | Maximum VUF [%] | Voltage violation hours | Final SoC |
|---|---:|---:|---:|---:|---:|
| Without BESS | 160804.2228 | 0.932925 | 2.456400 | 108.25 | - |
| Economic | 160790.6336 | 0.930618 | 2.363726 | 75.50 | 80% |
| Voltage | 160798.5145 | 0.932925 | 2.456400 | 108.25 | 20% |
| Balance | 160799.0643 | 0.932925 | 2.456400 | 108.25 | 20% |

PV produces 276632.9668 kWh at Q=0 in every run. Without BESS, it reduces
annual cost by 22.8711% relative to the scenario without PV. Within the PV
scenario, economic BESS control saves 13.5892 currency units (0.00845%) and
reduces violation duration from 108.25 to 75.50 hours, while the worst voltage
is lower. Voltage and balance still reach the 20% reserve on January 5 at
15:00 UTC and January 10 at 18:00 UTC, respectively, and remain idle thereafter.
No charging occurs in those two modes. Initial/final energy differences still
apply when interpreting raw cost. `pv_impact.csv` compares both scenarios;
`validation.json` records the export consistency checks.

### Rule review (v2)

The original traces exposed missing recharge in support modes, unreachable
economic price triggers, simultaneous charging/discharging between phases
during undervoltage overrides, and magnitude-only balance control that could
worsen VUF. The revised rules above address those operating decisions while
retaining the environment's physical model and action signs.

Closed-loop engineering validation uses January 2023 (2,976 intervals), with
identical loads, PV, EV sessions and initial SoC. Results in
`results/baselines/review_2023_01_revised/operation_review/` compare the same
month from the original annual runs with the revised controller:

| Mode | Violation hours, previous → revised | Minimum V [pu], previous → revised | Maximum VUF [%], previous → revised | Revised minimum SoC |
|---|---:|---:|---:|---:|
| Economic | 9.75 → 4.50 | 0.933533 → 0.943200 | 2.157673 → 2.189783 | 25.31% |
| Voltage | 13.00 → 0.75 | 0.937445 → 0.948389 | 2.190492 → 2.228059 | 38.42% |
| Balance | 13.00 → 4.25 | 0.937445 → 0.938519 | 2.190492 → 1.758992 | 33.03% |

No revised run reaches the 20% reserve or mixes opposite phase P signs in
this window. Voltage/balance continue operating for the whole month, with
recharge cycles. Activity is counted by individual phase powers: total Q can
be zero during useful balancing. Economic/voltage improve voltage compliance
but increase maximum VUF slightly; objectives are not interchangeable.
Raw costs and terminal SoC remain in the comparison CSV.

`tools/review_rule_based.py` selects the comparison period with `START`/`END`
and the displayed phase-P/Q, SoC, local-voltage and price window with
`PLOT_START`/`PLOT_END`. Its current defaults reproduce the annual comparison
below and a July operation window. The month above is engineering validation
on the investigated data, not held-out evidence. Existing baselines remain
immutable.

### Annual evaluation of v2

`results/baselines/annual_2023_pv_v2/` contains 35,040 continuous intervals per
case, with the same initial SoC, hardware, PV, EV sessions, loads and seed as
v1. The no-BESS run reproduces v1 exactly. Model/input hashes and exported
phase powers, energy balances, causal price bands and timestamps passed the
recorded checks. Controller settings were frozen throughout the evaluation.

| Mode | Energy cost, v1 → v2 | Violation hours, v1 → v2 | Minimum V [pu], v1 → v2 | Maximum VUF [%], v1 → v2 |
|---|---:|---:|---:|---:|
| Economic | 160790.6336 → 160795.8149 | 75.50 → 50.00 | 0.930618 → 0.933585 | 2.363726 → 2.561383 |
| Voltage | 160798.5145 → 160781.6463 | 108.25 → 8.75 | 0.932925 → 0.944939 | 2.456400 → 2.374062 |
| Balance | 160799.0643 → 160803.3459 | 108.25 → 47.25 | 0.932925 → 0.935463 | 2.456400 → 2.317085 |

Voltage control reduces violation duration by 91.92%, improves minimum voltage
and VUF, and lowers raw energy cost. Its minimum SoC is 35.83%, with no
reserve-idle interval. Balance reduces maximum VUF by 5.67%, mean maximum VUF
by 9.70% and violations by 56.35%, at 4.2816 higher raw energy cost. Its final
SoC is 44.35% versus 20% in v1; the terminal-recharge estimates differ by only
0.4248 currency units. Those estimates are not additional executed operation.

Economic control does not improve its cost objective: cost rises by 5.1813
(0.00322%), and maximum VUF rises by 8.36%, despite fewer voltage violations.
An observational diagnostic finds 746.33 kWh of net BESS discharge during
global export in v2, versus 12.29 kWh in v1. Local import and retail prices do
not establish the global marginal value of discharge. This is a limitation
to investigate, not a causal cost attribution from an ablation.

Availability remains incomplete: economic/balance accumulate 464.00/302.25
reserve-idle hours, with longest runs of 158.00/156.50 hours. V2 is an advance
for voltage support and VUF balance, not a uniform improvement in all modes.
Final SoC is 59.47%, 53.20% and 44.35% for economic, voltage and balance;
compare it alongside raw cost. Battery degradation is not included in energy cost.

`assessment.csv` compares full-year metrics; `operation_review/comparison.csv`
adds phase activity and availability. `operation_review/february_december.csv`
shows the remaining eleven months separately. The voltage/VUF improvements
persist there, while economic cost is worse. This evaluates the investigated
2023 scenario, not generalization to an independent dataset.

## Planned economic training comparison

The active BESS is at `bus_004_lv`, behind T4, sharing the node with conventional
loads and ASAP EVSEs. The fixed PV remains at `bus_005` (11 kV), outside that
aggregator. This deliberately separates the local bill from the feeder's bill.
Historical baselines used a different BESS connection; do not treat them as
results from this new topology.

Both SAC policies will receive the same filtered local observations: local phase
voltages/angles, SoC, previous executed P/Q, time, buy/sell prices and local branch
flows (`extra_features=("grid", "branches")`). Neither actor receives remote PV
measurements or global grid power. Only the economic objective changes:

| Experiment | Cost term | Meter boundary |
|---|---|---|
| `feeder` | `feeder_cost` | External grid connection, including network losses |
| `local` | `node_cost` | `bus_004_lv`: loads, EVSEs and BESS, excluding remote PV and transformer losses |

For either experiment, create the same scope and select one term:

```python
from opendss_environment import Reward

cost_term = "feeder_cost"  # Use "node_cost" for the local experiment.
reward = Reward(controllers={"b1": {
    "devices": {"bess": ["b1"]}, "buses": ["bus_004_lv"],
    "weights": {cost_term: 1.0}, "scales": {cost_term: 1.0},
}})
# Pass reward as EnvironmentConfig.reward_function, then extract after step():
training_reward = Reward.get_local(info["rewards"], "b1")
```

The default scalar from `env.step()` remains the feeder cost reward; the local
experiment must explicitly extract the configured controller reward above.
Keep architecture, action limits, history length, episode starts, initial SoC,
tariffs, training budget and common penalty terms/scales identical. Compare both
unscaled bills, voltages, transformer loading, throughput and terminal energy
for both policies, on the same evaluation periods. This is a planned comparison:
Actor inference is implemented; training/HPO remain commented scaffolds and
trained policy evaluation is not implemented.

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
