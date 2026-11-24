"""Normalized total active power for a BESS with balanced dispatch and Q=0."""
import numpy as np
from gymnasium.spaces import Box

from opendss_environment import Action, ControllerActions, TotalAction
from opendss_environment.elements import BESSConfig
from .observation import FEATURE_NAMES


def bess_actions(bess, base_kva, config, phases=3):
    if config.action_mode == "per_phase":
        return ControllerActions({"bess": [BESSConfig(**bess)]}, base_kva, phases=phases, seed=config.seed)
    return ActivePowerActions(bess, base_kva, config)


class ActivePowerActions:
    def __init__(self, bess, base_kva, config):
        if config.action_mode not in ("p_fixed", "p_soc"):
            raise ValueError(f"Unknown SAC action mode: {config.action_mode}")
        self.bess = BESSConfig(**bess)
        self.mode, self.history_steps = config.action_mode, config.history_steps
        self.dt = getattr(config, "dt_h", None)
        self.charge = min(self.bess.p_charge_max_kw, self.bess.s_max_kva)
        self.discharge = min(self.bess.p_discharge_max_kw, self.bess.s_max_kva)
        self.rating = max(self.charge, self.discharge)
        if self.rating <= 0:
            raise ValueError("Active power control requires a positive power rating")
        self.base_kva = base_kva
        self.low, self.high = np.array([-self.discharge / base_kva]), np.array([self.charge / base_kva])
        self.action_space = Box(-1., 1., shape=(1,), dtype=np.float32, seed=config.seed)

    def project(self, state, action):
        lo, hi = -self.discharge, self.charge
        if self.mode == "p_soc":
            idx = FEATURE_NAMES.index("soc_frac")
            if isinstance(state, dict):
                soc = state["history"][..., -1, idx:idx + 1]
            else:
                idx -= self.history_steps + len(FEATURE_NAMES)
                soc = state[..., idx:idx + 1]
            b = self.bess
            lo = -((soc - b.soc_min_frac) * b.e_cap_kwh * b.eta_discharge / self.dt).clip(0, self.discharge)
            hi = ((b.soc_max_frac - soc) * b.e_cap_kwh / (self.dt * b.eta_charge)).clip(0, self.charge)
        return (lo + (action + 1) * (hi - lo) / 2) / self.rating

    def command(self, action):
        if not self.action_space.contains(np.asarray(action, dtype=np.float32)):
            raise ValueError("Expected one normalized active power")
        return Action(bess={self.bess.id: TotalAction(float(action[0]) * self.rating / self.base_kva)})
