"""Causal local measurements and bounded history for BESS neural policies."""
from calendar import isleap
from collections import deque
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np


PHASES = ("a", "b", "c")
FEATURE_NAMES = (
    *(f"v_{p}_scaled" for p in PHASES),
    *(f"angle_{p}_minus_a_{fn}" for p in ("b", "c") for fn in ("sin", "cos")),
    *(f"{power}_import_{p}_scaled" for power in ("p", "q") for p in PHASES),
    *(f"loading_{p}_frac" for p in PHASES),
    "vuf_scaled", "soc_frac",
    *(f"previous_{power}_{p}_scaled" for power in ("p", "q") for p in PHASES),
    "buy_price_scaled", "sell_price_scaled",
    *(f"{period}_{fn}" for period in ("day", "week", "year") for fn in ("sin", "cos")),
)


class LocalObservation:
    """Encode one BESS/local meter; raw phase-load profiles are not actor inputs."""

    feature_names = FEATURE_NAMES

    def __init__(self, bess, scaling, history_steps=8, timezone_name="Europe/Zurich"):
        self.id, self.bus = bess["id"], bess["bus"]
        self.scaling = scaling
        self.device_rating = bess["s_max_kva"]
        self.device_phases = len(bess["phases"])
        self.history_steps = history_steps
        self.shape = (history_steps, len(FEATURE_NAMES))
        self.flat_size = history_steps * (len(FEATURE_NAMES) + 1)
        self.timezone = ZoneInfo(timezone_name)
        self.reset()

    def reset(self):
        self.history = deque(maxlen=self.history_steps)
        self.last_timestamp = None

    def encode(self, observation, *, flatten=False):
        """Add the current sample and return the padded history and validity mask."""
        timestamp = datetime.fromisoformat(str(observation["timestamp"]))
        timestamp = timestamp.replace(tzinfo=timestamp.tzinfo or timezone.utc).astimezone(timezone.utc)
        values = self._features(observation, timestamp)
        if self.last_timestamp and timestamp not in (
            self.last_timestamp, self.last_timestamp + timedelta(hours=float(observation["dt_h"]))
        ):
            self.history.clear()
        if timestamp == self.last_timestamp:
            self.history.pop()
        self.history.append(values)
        self.last_timestamp = timestamp
        padding = self.history_steps - len(self.history)
        history = np.pad(np.asarray(self.history), ((padding, 0), (0, 0)))
        mask = np.arange(self.history_steps) >= padding
        return np.concatenate((history.ravel(), mask)) if flatten else {"history": history, "mask": mask}

    def _features(self, observation, timestamp):
        bus, state = observation["bus"], observation["components"]["bess"][self.id]
        voltage = np.array([bus["v_before_pu"][p] for p in PHASES], dtype=float)
        angles = np.deg2rad([bus["angle_before_deg"][p] for p in PHASES])
        relative = angles[1:] - angles[0]
        meter = {row["phase"]: row for row in observation["branches"]
                 if row["bus"].lower() == self.bus.lower()
                 and row["element"].lower() == self.scaling.meter_element.lower()
                 and row["terminal"] == self.scaling.meter_terminal}
        p, q, loading = np.array([[meter[phase][key] for phase in PHASES]
                                 for key in ("p_flow_pu", "q_flow_pu", "loading_pct")])
        phasors, a = voltage * np.exp(1j * angles), np.exp(2j * np.pi / 3)
        positive, negative = phasors @ [1, a, a**2], phasors @ [1, a**2, a]
        previous = [[state[f"previous_{power}_pu"][phase] for phase in PHASES] for power in ("p", "q")]
        prices = [observation["grid"][f"{side}_price_per_kwh"] for side in ("buy", "sell")]
        imported = self.scaling.power([-p, -q], self.scaling.meter_s_kva, self.scaling.meter_phases)
        previous = self.scaling.power(previous, self.device_rating, self.device_phases)
        return np.concatenate((
            self.scaling.voltage(voltage), np.column_stack((np.sin(relative), np.cos(relative))).ravel(),
            *imported, loading / 100,
            [self.scaling.vuf(100 * abs(negative) / abs(positive)), state["soc_before_frac"]],
            *previous, self.scaling.price(prices), self._calendar(timestamp),
        ), dtype=np.float32)

    def _calendar(self, timestamp):
        clock = timestamp.astimezone(self.timezone)
        day = (clock.hour * 3600 + clock.minute * 60 + clock.second) / 86400
        cycles = np.array([day, (clock.weekday() + day) / 7,
                           (clock.timetuple().tm_yday - 1 + day) / (366 if isleap(clock.year) else 365)])
        angles = 2 * np.pi * cycles
        return np.column_stack((np.sin(angles), np.cos(angles))).ravel()
