"""Monthly training episodes and held-out validation weeks for a scenario."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


def monthly_episodes(case_path, episode_steps, seed):
    case_path = Path(case_path)
    case = json.loads(case_path.read_text(encoding="utf-8"))
    split = case["training_split"]
    frame = pd.read_csv(case_path.parent / case["files"]["demand"], usecols=["timestamp"])
    times = pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True)).sort_values().tz_convert(split["timezone"])
    held_out = (times.day >= split["validation_start_day"]) & (times.day < split["validation_end_day"])
    training = (times.year == split["year"]) & ~held_out
    midnight = np.flatnonzero(training & (times.hour == 0) & (times.minute == 0))
    starts = [int(i) for i in midnight if i + episode_steps <= len(times) and training[i:i + episode_steps].all()]
    rng = np.random.default_rng(seed)
    selected = [int(rng.choice([i for i in starts if times[i].month == month]))
                for month in rng.permutation(np.arange(1, 13))]
    duration = (times[1] - times[0]) * episode_steps
    episodes = [{"month": times[i].month, "start_step": i, "end_step_exclusive": i + episode_steps,
                 "episode_steps": episode_steps,
                 "start": times[i].isoformat(), "end_exclusive": (times[i] + duration).isoformat()}
                for i in selected]
    validation = [{"month": month,
                   "start": pd.Timestamp(year=split["year"], month=month, day=split["validation_start_day"],
                                         tz=split["timezone"]).isoformat(),
                   "end_exclusive": pd.Timestamp(year=split["year"], month=month, day=split["validation_end_day"],
                                                 tz=split["timezone"]).isoformat()}
                  for month in range(1, 13)]
    for window in validation:
        start = int(times.get_loc(pd.Timestamp(window["start"])))
        end = int(times.get_loc(pd.Timestamp(window["end_exclusive"])))
        window.update(start_step=start, end_step_exclusive=end, episode_steps=end - start)
    return episodes, validation
