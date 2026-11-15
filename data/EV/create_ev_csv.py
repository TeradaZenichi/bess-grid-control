"""Build a numeric EVSE connection matrix and a session lookup table."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
SOURCE = ROOT / "gridcharge-data/data/processed"
DEMAND = ROOT / "data/switzerland/5bus/unbalanced_demand.csv"
MIN_TRAIN_SESSIONS = 365
MIN_TRAIN_QUALITY = 0.90
BATTERY_CAPACITY_KWH = 120.0
CHARGING_EFFICIENCY = 1.0
TARGET_SOC = 1.0
SHORT_COLUMNS = {
    "session_id": "id", "evse_id": "evse", "arrival_time": "arr", "departure_time": "dep",
    "arrival_soc": "soc0", "target_soc": "soct", "battery_capacity_kwh": "cap_kwh",
    "charging_efficiency": "eta", "target_energy_kwh": "e_kwh", "rated_power_kw": "p_kw",
    "source_session_id": "src_id", "source_connector_key": "src_evse", "source_kind": "src",
    "original_arrival_utc": "src_arr", "original_departure_utc": "src_dep", "timezone_basis": "tz",
    "split": "split", "feasible_at_rated_power": "feasible", "crosses_calendar_boundary": "edge",
    "visible_in_connections": "sampled",
}


def select_events():
    continuity = ROOT / "gridcharge-data/data/reports/connector_continuity.csv"
    if not continuity.exists():
        subprocess.run([sys.executable, str(ROOT / "gridcharge-data/pipeline.py")], check=True)
    selected = pd.read_csv(continuity, dtype={"charger_id": str, "connector_id": str})
    selected["train_n"] = selected[[f"2023-{m:02d}" for m in range(1, 13)]].sum(axis=1)
    selected = selected.loc[selected.months_2023.eq(12) & selected.rated_metadata
        & selected.training_quality_fraction.ge(MIN_TRAIN_QUALITY)
        & selected.train_n.ge(MIN_TRAIN_SESSIONS)].sort_values(
            ["train_n", "connector_key"], ascending=[False, True]).copy()
    selected["evse_id"] = [f"e{i + 1:03d}" for i in range(len(selected))]
    if selected.empty:
        raise ValueError("No EVSE matches the selection")
    columns = ["session_id", "connector_key", "energy_kwh", "start_time_local", "start_time_utc",
               "end_time_utc", "rated_power_kw", "source_kind", "timezone_basis", "year"]
    events = pd.concat([pd.read_parquet(SOURCE / f"sessions_{year}.parquet", columns=columns,
        filters=[("connector_key", "in", selected.connector_key.tolist()), ("valid_for_rl", "==", True)])
        for year in (2023, 2024)], ignore_index=True)
    wall_clock = events.start_time_local.dt.tz_localize(None)
    events["scenario_arrival_utc"] = wall_clock.dt.tz_localize(
        "Europe/Zurich", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    events["split"] = events.year.map({2023: "train", 2024: "test"})
    bad_time = events.scenario_arrival_utc.isna()
    bad_capacity = events.energy_kwh.mul(CHARGING_EFFICIENCY).gt(BATTERY_CAPACITY_KWH * TARGET_SOC)
    rejected = {"unresolved_calendar_mapping": int(bad_time.sum()), "incompatible_capacity": int(bad_capacity.sum())}
    excluded = events.loc[bad_time | bad_capacity, ["session_id", "connector_key"]].copy()
    excluded["reason"] = np.where(bad_time.loc[excluded.index], "calendar_mapping", "battery_capacity")
    events = events.loc[~(bad_time | bad_capacity)].copy()
    # Check overlaps again after the synthetic local-clock mapping.
    sys.path.insert(0, str(ROOT / "gridcharge-data"))
    from src.validation import overlap_flags
    mapped = events[["connector_key"]].copy()
    mapped["start_time_utc"] = events.scenario_arrival_utc
    mapped["end_time_utc"] = events.scenario_arrival_utc + (events.end_time_utc - events.start_time_utc)
    overlaps = overlap_flags(mapped)
    rejected["mapped_overlap_sessions"] = int(overlaps.sum())
    extra = events.loc[overlaps, ["session_id", "connector_key"]].copy()
    extra["reason"] = "mapped_overlap"
    pd.concat([excluded, extra]).to_csv(OUTPUT / "excluded_sessions.csv", index=False)
    return selected, events.loc[~overlaps].copy(), rejected


def arrival_soc(energy, capacity, efficiency, target):
    if capacity <= 0 or not 0 < efficiency <= 1 or not 0 < target <= 1:
        raise ValueError("Invalid battery capacity, efficiency or target SoC")
    soc = target - efficiency * np.asarray(energy) / capacity
    if not np.isfinite(soc).all() or (soc < 0).any() or (soc > target).any():
        raise ValueError("Energy is incompatible with the battery assumptions; do not clip SoC")
    return soc


def connection_matrix(timestamps, sessions, evse_ids):
    """A session occupies t exactly when arrival <= t < departure."""
    matrix = np.zeros((len(timestamps), len(evse_ids)), dtype=np.int32)
    for column, evse in enumerate(evse_ids):
        group = sessions.loc[sessions.evse_id.eq(evse)].sort_values("arrival_time")
        if group.arrival_time.iloc[1:].reset_index(drop=True).lt(
                group.departure_time.cummax().iloc[:-1].reset_index(drop=True)).any():
            raise ValueError(f"Overlapping mapped sessions at {evse}")
        for row in group.itertuples():
            start = timestamps.searchsorted(row.arrival_time, side="left")
            stop = timestamps.searchsorted(row.departure_time, side="left")
            matrix[start:stop, column] = row.session_id
    return pd.DataFrame(matrix, columns=evse_ids)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    calendar = pd.read_csv(DEMAND, usecols=["timestamp"])
    timestamps = pd.DatetimeIndex(pd.to_datetime(calendar.timestamp, utc=True))
    if not timestamps.is_unique or not timestamps.is_monotonic_increasing:
        raise ValueError("Demand timestamps must be unique and increasing")
    if not (timestamps[1:] - timestamps[:-1] == pd.Timedelta(minutes=15)).all():
        raise ValueError("Expected a 15-minute demand calendar")
    selected, events, rejected = select_events()
    events = events.merge(selected[["connector_key", "evse_id"]], on="connector_key", validate="many_to_one")
    events = events.sort_values(["scenario_arrival_utc", "evse_id", "session_id"]).reset_index(drop=True)
    sessions = pd.DataFrame({
        "session_id": np.arange(1, len(events) + 1, dtype=np.int32), "evse_id": events.evse_id,
        "arrival_time": events.scenario_arrival_utc,
        # Use the original exact timedelta, avoiding rounding through float minutes.
        "departure_time": events.scenario_arrival_utc + (events.end_time_utc - events.start_time_utc),
        "arrival_soc": arrival_soc(events.energy_kwh, BATTERY_CAPACITY_KWH, CHARGING_EFFICIENCY, TARGET_SOC),
        "target_soc": TARGET_SOC, "battery_capacity_kwh": BATTERY_CAPACITY_KWH,
        "charging_efficiency": CHARGING_EFFICIENCY, "target_energy_kwh": events.energy_kwh,
        "rated_power_kw": events.rated_power_kw, "source_session_id": events.session_id,
        "source_connector_key": events.connector_key, "source_kind": events.source_kind,
        "original_arrival_utc": events.start_time_utc, "original_departure_utc": events.end_time_utc,
        "timezone_basis": events.timezone_basis, "split": events.split,
    })
    duration_h = (sessions.departure_time - sessions.arrival_time).dt.total_seconds() / 3600
    sessions["feasible_at_rated_power"] = sessions.target_energy_kwh.le(sessions.rated_power_kw * duration_h + 1e-9)
    sessions["crosses_calendar_boundary"] = sessions.arrival_time.lt(timestamps[0]) | sessions.departure_time.gt(timestamps[-1] + pd.Timedelta(minutes=15))
    matrix = connection_matrix(timestamps, sessions, selected.evse_id.tolist())
    matrix.insert(0, "timestamp", calendar.timestamp)
    visible = np.unique(matrix.iloc[:, 1:].to_numpy())
    sessions["visible_in_connections"] = sessions.session_id.isin(visible)
    matrix.to_csv(OUTPUT / "evse_connections.csv", index=False)
    sessions.rename(columns=SHORT_COLUMNS).to_csv(OUTPUT / "ev_sessions.csv", index=False, float_format="%.12g")
    activity = sessions.groupby("evse_id").agg(n=("session_id", "size"), e_kwh=("target_energy_kwh", "sum")).reindex(selected.evse_id, fill_value=0)
    for label in ("train", "test"):
        activity[f"n_{label}"] = sessions.loc[sessions.split.eq(label)].groupby("evse_id").size().reindex(activity.index, fill_value=0)
    activity["occ_pct"] = 100 * matrix.iloc[:, 1:].ne(0).mean()
    activity.to_csv(OUTPUT / "evse_activity.csv", index_label="evse")
    config = {
        "schema_version": 2, "units": {"soc": "fraction", "energy": "kWh", "power": "kW"},
        "session_columns": {short: long for long, short in SHORT_COLUMNS.items()},
        "selection": {"min_train_sessions": MIN_TRAIN_SESSIONS, "min_train_quality": MIN_TRAIN_QUALITY,
                      "training_months": 12, "types": "all with known rated power", "order": "2023 activity descending, then source key"},
        "connection_semantics": "0=disconnected; positive integer=session_id at interval start; arrival<=timestamp<departure",
        "energy_assumption": "Historical delivered energy is the grid-side energy required to reach target_soc",
        "battery_assumption": {"capacity_kwh": BATTERY_CAPACITY_KWH, "charging_efficiency": CHARGING_EFFICIENCY, "target_soc": TARGET_SOC},
        "mapping": "Source Scottish local arrival mapped to Zurich local arrival on the same date; original duration retained. Synthetic joint scenario.",
        "timezone_note": "Original timestamps have no offset: GridCharge declares UTC; official supplement assumes UTC. Producer confirmation remains necessary.",
        "evses": {row.evse_id: {"source_charger_id": row.charger_id, "source_connector_id": row.connector_id,
            "source_connector_key": row.connector_key, "connector_type": row.connector_type,
            "rated_power_kw": row.rated_power_kw, "train_n": row.train_n,
            "train_quality": row.training_quality_fraction, "bus": None, "phases": None} for row in selected.itertuples()},
    }
    (OUTPUT / "evse_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = {"timestamps": len(matrix), "evses": len(selected), "sessions": len(sessions),
        "selection": config["selection"], "excluded": rejected,
        "connector_types": selected.connector_type.value_counts().to_dict(),
        "sessions_per_evse_day": len(sessions) / len(selected) / (len(timestamps) / 96),
        "sampled_occupancy_pct": float(100 * matrix.iloc[:, 1:].ne(0).to_numpy().mean()),
        "sessions_between_samples": int((~sessions.visible_in_connections).sum()),
        "infeasible_at_rated_power": int((~sessions.feasible_at_rated_power).sum()),
        "crosses_calendar_boundary": int(sessions.crosses_calendar_boundary.sum()),
        "arrival_soc_min": float(sessions.arrival_soc.min()), "arrival_soc_max": float(sessions.arrival_soc.max()),
        "sha256": {name: hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest() for name in
                   ("ev_sessions.csv", "evse_connections.csv", "evse_config.json")},
    }
    (OUTPUT / "generation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
