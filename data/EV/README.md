# EV charging scenario

Run from the repository root, using the project environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r gridcharge-data/requirements.txt
.\.venv\Scripts\python.exe data/EV/download_gridcharge.py
.\.venv\Scripts\python.exe data/EV/create_ev_csv.py
```

`download_gridcharge.py` contains the downloader used for this extraction. It
preserves originals and checks hashes in `gridcharge-data/data/raw/`. The pinned
GridCharge revision contains 23 months; September 2024 comes from the official
ChargePlace Scotland XLSX, with separate provenance. `create_ev_csv.py` uses the
audited Parquets; if they are absent, it first executes `gridcharge-data/pipeline.py`.
The pipeline source and dependency list must accompany these scripts.

## CSVs

- `evse_connections.csv`: exactly the timestamp strings and row order of
  `../switzerland/5bus/unbalanced_demand.csv`; each EVSE column contains `0` or
  the integer session ID connected **at the start** of that interval. Columns use short
  names `e001`, `e002`, etc., ranked by 2023 activity; the JSON maps each to its source.
- `ev_sessions.csv`: session lookup with short column names (legend below).
  `arr <= timestamp < dep` defines connection. Times are UTC.

The selection uses 2023 activity/quality only: all 12 training months, >=365
basic-valid sessions, >=90% training quality and known rated power. All connector
types and powers qualify; the previous Type 2 / 22 kW restriction is removed.
Only sessions flagged `valid_for_rl` are exported. Choosing a less strict station
quality threshold does not reinstate its invalid individual sessions. Numeric
IDs and column mappings are regenerated together; old EVSE codes must not be
mixed with this export. `src_id` retains original provenance.

| Column | Meaning |
|---|---|
| `id`, `evse` | Numeric session ID and EVSE column name |
| `arr`, `dep` | Exact mapped arrival and departure UTC |
| `soc0`, `soct` | Assumed arrival and target SoC (fraction) |
| `cap_kwh`, `eta` | Assumed battery capacity and charging efficiency |
| `e_kwh`, `p_kw` | Grid-side target energy and EVSE nominal power |
| `src_id`, `src_evse`, `src` | Source session key, connector key and dataset origin |
| `src_arr`, `src_dep`, `tz` | Original UTC times and timezone interpretation |
| `split` | Train 2023 / test 2024 |
| `feasible`, `edge`, `sampled` | Fits rated power, crosses calendar edge, appears in sampled matrix |

`evse_config.json` contains the full column legend. `evse_activity.csv` reports
session counts by year and sampled occupancy for each EVSE.

## Explicit assumptions

Each session uses a **120 kWh** battery, **100%** charging efficiency and a target
SoC of **1.0**. These are configurable scenario assumptions, not measurements.
`arrival_soc = target_soc - efficiency * target_energy_kwh / battery_capacity_kwh`.
Sessions incompatible with the battery capacity or with unresolved calendar
mappings are excluded explicitly in `excluded_sessions.csv`; SoC is never
silently clipped. Any overlap created by calendar mapping is also excluded
(both involved sessions). The energy
target is historical delivered energy assumed sufficient to reach the target SoC.

Scottish local arrival time is mapped to the same Zurich local date/time and
original elapsed duration is retained. This is a synthetic joint scenario with
the Swiss loads. The source clock convention remains unconfirmed: original
strings have no offset, while GridCharge declares UTC; the official supplement
uses UTC as an explicit assumption.

The connection matrix is a sampled view, not the full event schedule. A session
between two timestamps may never appear in the matrix; it remains in the session
CSV, marked `sampled=False`. The environment must process the
session events within each interval to account for partial connection times and
multiple sequential sessions, rather than ignoring these events or granting a
full 15 minutes to every arrival. The half-open interval convention also means
a departure exactly at a timestamp is already disconnected there.

`feasible=False` flags sessions whose energy cannot fit into their
historical duration at the selected charger's nominal power. They are preserved
for explicit treatment, not repaired. Calendar-boundary crossings are also flagged;
full duration and energy remain in the session table.

`evse_config.json` maps columns to source connectors. `bus` and `phases` are null
until a physical placement is explicitly chosen. Do not interpret final energy,
departure or modeled SoC as measured causal arrival data. Whether the controller
receives the assumed target/deadline must be specified in the experiment.

`generation_report.json` records counts, flags and hashes of the generated files.

## OpenDSS environment

`../switzerland/5bus/config_unbalanced_ev.json` is a small integration scenario
with six EVSEs. Three use external `controlled` commands and three use `asap`.
Their feeder buses and phases are explicit synthetic scenario choices; they do
not come from GridCharge.
In the repository `main.py`, set `USE_EV = True` to run one day. Keep
`CONTROLLED_EV_POWER_PU = None` to leave controlled EVSEs idle, or set a
nonnegative per-unit charging command such as `0.022`.

The environment reads this session table directly. It processes exact events
inside each 15-minute interval and sends the energy-equivalent average load to
OpenDSS; `evse_connections.csv` remains an inspection artifact.
