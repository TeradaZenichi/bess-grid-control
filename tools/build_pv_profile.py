"""Scale the archived solar shape to the fixed PV and demand timestamps."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/old/switzerland/pv/pv_generation_Switzerland.csv"
CASE = ROOT / "data/switzerland/5bus"
REFERENCE_YEAR = 2023
PV_ID = "pv1"


def build_profile(source, timestamps, power_kw):
    if not np.isfinite(power_kw) or power_kw <= 0:
        raise ValueError("PV rated power must be finite and positive")
    source = source.set_index(pd.to_datetime(source.datetime, utc=True)).power_kWh
    calendar = pd.date_range(f"{REFERENCE_YEAR}-01-01", f"{REFERENCE_YEAR + 1}-01-01",
                             freq="15min", inclusive="left", tz="UTC")
    reference = source.reindex(calendar)
    missing = reference.index[reference.isna()]
    # The archive ends at 23:00 UTC: the remaining three night intervals are zero.
    if len(missing) and (not missing.isin(calendar[-3:]).all() or reference.iloc[-4] != 0):
        raise ValueError("Unexpected gaps in the reference solar year")
    reference = reference.fillna(0)
    if not np.isfinite(reference).all() or (reference < 0).any() or reference.max() <= 0:
        raise ValueError("Reference solar shape must be finite, nonnegative and nonzero")
    shape = pd.Series((reference / reference.max()).to_numpy(), index=calendar.strftime("%m-%d %H:%M"))
    # Repeat the reference by UTC calendar; interpolate February 29 between adjacent days.
    hours = pd.date_range("2024-02-29", periods=96, freq="15min").strftime("%H:%M")
    leap = (shape.reindex("02-28 " + hours).to_numpy() + shape.reindex("03-01 " + hours).to_numpy()) / 2
    shape = pd.concat([shape, pd.Series(leap, index="02-29 " + hours)])
    keys = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).strftime("%m-%d %H:%M")
    values = shape.reindex(keys).to_numpy() * power_kw
    if not np.isfinite(values).all():
        raise ValueError("Solar profile does not cover every demand timestamp")
    return values


def main():
    timestamps = pd.read_csv(CASE / "unbalanced_demand.csv", usecols=["timestamp"]).timestamp
    device = next(pv for pv in json.loads((CASE / "devices.json").read_text())["pv"] if pv["id"] == PV_ID)
    filename, column = device["profile"].rsplit(":", 1)
    values = build_profile(pd.read_csv(SOURCE), timestamps, device["p_max_kw"])
    pd.DataFrame({"timestamp": timestamps, column: values}).to_csv(CASE / filename, index=False)
    print(f"PV: {len(values)} intervals; peak={values.max():.1f} kW; reference year={REFERENCE_YEAR}")


if __name__ == "__main__":
    main()
