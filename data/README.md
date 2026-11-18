# Data

- `switzerland/5bus/`: active network, demand and device configuration.
- `switzerland/5bus/pv.csv`: fixed solar generation in kW, aligned with demand.
- `switzerland/prices/standard_prices.csv`: active electricity tariff.
- `EV/ev_sessions.csv`: EV sessions consumed by the environment.
- `EV/evse_connections.csv`: companion connection matrix aligned with demand timestamps.
- `EV/support/`: generators, metadata and extraction reports.
- `old/`: other datasets, networks, tariffs and historical configurations.

Run the active scenario with `python main.py` from the repository root.
For data regeneration, see [EV support](EV/support/README.md).

The active feeder has five 11 kV buses and four LV buses at 220 V phase-neutral.
Four grounded delta/wye transformers feed conventional loads and ASAP EVSEs:
400 kVA at bus 002 and 250 kVA at buses 003-005. The BESS shares `bus_004_lv`
with local loads/EVSEs; the PV remains at `bus_005` (11 kV), outside this aggregator.
The original demand CSV is unchanged; `network.load_bus_map` maps its bus names
to the `_lv` buses. See [network parameters and diagram](../docs/network-diagram.md).
Run `python -m tools.run_idle_bess` for the reproducible idle-BESS day evaluation.

The scenario's `observation_scaling` block sets fixed voltage-deviation, VUF and
price references for neural controllers. Shared scaling also reads meter ratings
from OpenDSS and inverter ratings from device parameters; it does not change
the environment's physical observations. See [controller inputs](../controllers/README.md#sac-local-observations).

`training_split` reserves days 15-21 of every local month in 2023 for SAC
validation. Training episode windows exclude those dates and all of 2024.
The annual baseline evaluator still reports 2023 as one `train_eval` group.

The active BESS is `b1` at `bus_004_lv`: 100 kWh, 40 kW charge/discharge and a
50 kVA inverter, with independent phase dispatch. Initial SoC is 50%, with
physical limits of 10-100%; charge/discharge efficiencies are both 95%.
Its hardware ratings match the historical baselines, but its connection is now
on LV, behind T4, instead of MV. `main.py` creates its local controller from
`switzerland/5bus/devices.json`; EVSEs remain ASAP.

The active PV is 200 kW at `bus_005`, balanced across three phases, with fixed
unity power factor, no curtailment and no reactive control. Its power rating is
configured in `switzerland/5bus/devices.json`. Regenerate the profile after
changing that rating with `python -m tools.build_pv_profile`.

The generator uses the archived `old/switzerland/pv/pv_generation_Switzerland.csv`
as a normalized solar shape: it divides the 2023 values by their reference-year
peak, then scales them to the configured PV rating. The source's `power_kWh`
header is not interpreted as interval energy. The three missing final night
intervals of 2023 UTC are zero-filled. Other missing values fail validation.
The 2023 shape is repeated by UTC month/day/time for 2024; February 29 is the
mean of February 28 and March 1 at the same time. Thus 2024 is a scenario
assumption, not observed 2024 solar data. Original demand timestamp strings
and the archived source remain unchanged.
