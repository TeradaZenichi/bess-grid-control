"""Chronological contiguous windows covering the complete training year."""
import numpy as np
import pandas as pd


def split_episodes(data):
    split = data["training_split"]
    times = pd.DatetimeIndex(pd.to_datetime(data["timestamps"], utc=True)).tz_convert(split["timezone"])
    year = times.year == split["year"]
    held_out = (times.day >= split["validation_start_day"]) & (times.day < split["validation_end_day"])

    def windows(mask):
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        return [{"month": times[start].month, "start_step": int(start), "end_step_exclusive": int(end),
                 "episode_steps": int(end - start), "start": times[start].isoformat(),
                 "end_exclusive": (times[end - 1] + pd.Timedelta(hours=data["dt"])).isoformat()}
                for start, end in zip(edges[::2], edges[1::2])]

    return windows(year & ~held_out), windows(year & held_out)
