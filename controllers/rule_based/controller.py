"""Local BESS rules with a private one-step device feasibility projection."""
from collections import deque
from cmath import rect
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
import math

from opendss_environment import PhaseAction, TotalAction
from opendss_environment.elements import BESS, BESSConfig
from opendss_environment.network import device_phases

from ..base import BaseController, Config


def _finite(value):
    if isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError("Controller inputs must be finite numbers")
    return float(value)


def _clip(value, lower, upper):
    return min(max(value, lower), upper)


class RuleBasedController(BaseController):
    """Return one BESS command without reading or advancing the environment."""

    def __init__(self, bess, settings=None, *, scaling=None):
        self.parameters = BESSConfig(**deepcopy(bess))
        self.scaling = scaling
        self.config = Config.controller("rule_based", settings)
        self.id, self.bus = self.parameters.id, self.parameters.bus
        for name, expected in (("bess_id", self.id), ("bus", self.bus)):
            if getattr(self.config, name) not in (None, expected):
                raise ValueError(f"{name} does not match the supplied BESS")
        self.extra_features = tuple(self.config.extra_features)
        self._validate()
        self.reset()

    def _validate(self):
        c, b = self.config, self.parameters
        if c.mode not in {"economic", "voltage", "balance", "tracking"}:
            raise ValueError("Unknown operation mode")
        if set(self.extra_features) - {"grid", "branches"}:
            raise ValueError("EXTRA_FEATURES must be grid or branches")
        for name in ("price_low", "price_high", "soc_reserve", "soc_target", "v_min_pu",
                     "v_low_pu", "v_high_pu", "v_max_pu", "q_gain", "p_gain", "p_assist_error", "balance_gain",
                     "price_window_hours", "price_low_quantile", "price_high_quantile", "soc_recharge_start",
                     "soc_recharge_target", "recharge_power_frac", "charge_v_min_pu", "charge_v_full_pu"):
            setattr(c, name, _finite(getattr(c, name)))
        if c.price_low >= c.price_high:
            raise ValueError("PRICE_LOW must be below PRICE_HIGH")
        if not 0 < c.v_min_pu <= c.v_low_pu <= c.v_high_pu <= c.v_max_pu:
            raise ValueError("Voltage thresholds must be positive and ordered")
        if not 0 <= b.soc_min_frac <= c.soc_reserve <= c.soc_target <= b.soc_max_frac <= 1:
            raise ValueError("Reserve and target must lie inside the BESS SoC limits")
        if not c.soc_reserve <= c.soc_recharge_start < c.soc_recharge_target <= c.soc_target:
            raise ValueError("Recharge start/target must lie between reserve and economic target")
        if not c.v_min_pu < c.charge_v_min_pu < c.charge_v_full_pu <= c.v_max_pu:
            raise ValueError("Charging voltage thresholds must leave margin above the lower limit")
        if not 0 < c.recharge_power_frac <= 1:
            raise ValueError("RECHARGE_POWER_FRAC must be in (0, 1]")
        if c.price_window_hours <= 0 or not 0 <= c.price_low_quantile < c.price_high_quantile <= 1:
            raise ValueError("Invalid price history window or quantiles")
        if any(getattr(c, name) < 0 for name in ("q_gain", "p_gain", "p_assist_error", "balance_gain")):
            raise ValueError("Rule gains must be nonnegative")
        if any(not isinstance(getattr(c, name), bool) for name in ("allow_export", "adaptive_prices")):
            raise ValueError("ALLOW_EXPORT and ADAPTIVE_PRICES must be boolean")
        if c.max_ramp_frac is not None and _finite(c.max_ramp_frac) <= 0:
            raise ValueError("MAX_RAMP_FRAC must be positive or None")
        for value in (b.e_cap_kwh, b.p_charge_max_kw, b.p_discharge_max_kw, b.s_max_kva):
            if _finite(value) <= 0:
                raise ValueError("BESS energy and power ratings must be positive")
        if not 0 < _finite(b.eta_charge) <= 1 or not 0 < _finite(b.eta_discharge) <= 1:
            raise ValueError("BESS efficiencies must be in (0, 1]")
        if _finite(b.q_loss_rated_kw) < 0 or b.dispatch_mode not in {"aggregate", "per_phase"}:
            raise ValueError("Invalid BESS converter loss or dispatch mode")
        if b.connection != "wye":
            raise ValueError("The rule-based voltage controller currently supports wye BESS only")

    def reset(self):
        self.last_requested_action = None
        self.last_action = None
        self.last_decision = None
        self.price_history = deque()
        self.last_price_timestamp = None
        self.recharging = False

    def act(self, observation, objective=None):
        objective = objective or {}
        if set(objective) - {"mode", "p_reference_pu", "q_reference_pu"}:
            raise ValueError("Supported directives are mode and P/Q references in pu")
        c, b = self.config, self.parameters
        mode = objective.get("mode", c.mode)
        if mode not in {"economic", "voltage", "balance", "tracking"}:
            raise ValueError("Unknown operation mode")
        if observation.get("units") != "pu" or observation.get("controller", {}).get("bus") != self.bus:
            raise ValueError("Expected a local pu observation at the configured bus")
        state = observation["components"]["bess"][self.id]
        phases = tuple(state["phases"])
        if not phases or len(phases) != len(set(phases)) or state["bus"] != self.bus:
            raise ValueError("Invalid BESS observation phases or bus")
        if set(phases) != set(device_phases(b, len(observation["phase_order"]))):
            raise ValueError("Observed phases do not match the BESS configuration")
        base = _finite(observation["base"]["s_base_kva"])
        dt = _finite(observation["dt_h"])
        soc = _finite(state["soc_before_frac"])
        if min(base, dt) <= 0 or not b.soc_min_frac - 1e-9 <= soc <= b.soc_max_frac + 1e-9:
            raise ValueError("Invalid observation base, timestep or SoC")
        voltages = {phase: _finite(observation["bus"]["v_before_pu"][phase]) for phase in phases}
        if min(voltages.values()) <= 0:
            raise ValueError("Measured voltages must be positive")
        count = len(phases)
        rating = b.s_max_kva / base / count
        p, q = dict.fromkeys(phases, 0.0), dict.fromkeys(phases, 0.0)
        errors = {phase: max(c.v_low_pu - value, 0) - max(value - c.v_high_pu, 0)
                  for phase, value in voltages.items()}
        charge_limit = b.p_charge_max_kw / base * _clip(
            (min(voltages.values()) - c.charge_v_min_pu)
            / (c.charge_v_full_pu - c.charge_v_min_pu), 0, 1,
        )
        if soc <= c.soc_recharge_start:
            self.recharging = True
        elif soc >= c.soc_recharge_target:
            self.recharging = False
        low, high, price = None, None, None

        if mode == "economic":
            price = _finite(observation["grid"]["buy_price_per_kwh"])
            low, high = self._price_limits(price, dt, observation.get("timestamp"))
            profitable = low < high * b.eta_charge * b.eta_discharge
            power = b.p_charge_max_kw / base if profitable and price <= low and soc < c.soc_target else 0.0
            if profitable and price >= high and soc > c.soc_recharge_start:
                power = -min(b.p_discharge_max_kw / base,
                             (soc - c.soc_recharge_start) * b.e_cap_kwh * b.eta_discharge / (dt * base))
            p = dict.fromkeys(phases, power / count)
        elif mode == "tracking":
            if "p_reference_pu" not in objective or "q_reference_pu" not in objective:
                raise ValueError("Tracking requires explicit P/Q references in action signs")
            p = self._reference(objective["p_reference_pu"], phases)
            q = self._reference(objective["q_reference_pu"], phases)
        elif mode == "balance":
            if b.dispatch_mode != "per_phase" or count != 3 or not b.reactive_control:
                raise ValueError("Balance requires a reactive-enabled per-phase three-phase BESS")
            q = self._balance_q(voltages, observation["bus"]["angle_before_deg"], rating)
        else:
            if b.dispatch_mode == "aggregate":
                mean = sum(voltages.values()) / count
                error = max(c.v_low_pu - mean, 0) - max(mean - c.v_high_pu, 0)
                errors = dict.fromkeys(phases, error)
            for phase, error in errors.items():
                p[phase], q[phase] = self._voltage_power(error, rating)
                if c.v_min_pu <= voltages[phase] <= c.v_max_pu:
                    p[phase] = 0.0

        if mode != "tracking" and self.recharging and (price is None or price <= low):
            if not any(p.values()):
                loss = sum(b.q_loss_rated_kw / count * (_clip(value, -rating, rating) / rating) ** 2
                           for value in q.values()) / base
                recovery = min(c.recharge_power_frac * b.p_charge_max_kw / base,
                               (max(c.soc_recharge_target - soc, 0) * b.e_cap_kwh / (dt * base) + loss)
                               / b.eta_charge)
                p = dict.fromkeys(phases, recovery / count)

        if b.dispatch_mode == "aggregate" and any(
                not math.isclose(value, next(iter(values.values())), abs_tol=1e-12)
                for values in (p, q) for value in values.values()):
            raise ValueError("Unequal phase references require per_phase dispatch")
        self.last_requested_action = self._command(p, q)
        if c.max_ramp_frac is not None:
            step = float(c.max_ramp_frac) * rating
            for values, previous in ((p, state["previous_p_pu"]), (q, state["previous_q_pu"])):
                for phase in phases:
                    old = _finite(previous[phase])
                    values[phase] = _clip(values[phase], old - step, old + step)

        # Routine charging must respect the weakest phase, including after a ramp.
        if mode != "tracking":
            p = {phase: min(value, charge_limit / count) for phase, value in p.items()}

        # Local voltage support takes precedence over a mode request or ramp.
        guarded = voltages
        if b.dispatch_mode == "aggregate":
            if min(voltages.values()) < c.v_min_pu and max(voltages.values()) > c.v_max_pu:
                p, q = dict.fromkeys(phases, 0.0), dict.fromkeys(phases, 0.0)
                guarded = {}
            else:
                worst = min(voltages.values()) if min(voltages.values()) < c.v_min_pu else max(voltages.values())
                guarded = dict.fromkeys(phases, worst)
        for phase, value in guarded.items():
            if value < c.v_min_pu or value > c.v_max_pu:
                error = max(c.v_low_pu - value, 0) - max(value - c.v_high_pu, 0)
                support_p, support_q = self._voltage_power(error, rating)
                if value < c.v_min_pu:
                    p[phase], q[phase] = min(p[phase], support_p), max(q[phase], support_q)
                else:
                    p[phase], q[phase] = max(p[phase], support_p), min(q[phase], support_q)

        discharge = sum(max(-value, 0.0) for value in p.values())
        if mode == "economic" and not c.allow_export and discharge > 0:
            if "branches" not in observation:
                raise ValueError("Economic discharge without export requires local branch measurements")
            imported = -sum(_finite(flow["p_flow_pu"]) for flow in observation["branches"])
            demand = max(imported - sum(_finite(value) for value in state["previous_p_pu"].values()), 0.0)
            scale = min(demand / discharge, 1.0)
            p = {phase: value * scale if value < 0 else value for phase, value in p.items()}

        minimum = min(soc, c.soc_reserve)
        maximum = max(soc, c.soc_target) if mode == "economic" else b.soc_max_frac
        projector = BESS(replace(b, soc_min_frac=minimum, soc_max_frac=maximum))
        projector.soc = soc
        p_kw, q_kvar = projector.operate_phases({key: value * base for key, value in p.items()},
                                                {key: value * base for key, value in q.items()}, dt)
        self.last_action = self._command({key: value / base for key, value in p_kw.items()},
                                        {key: value / base for key, value in q_kvar.items()})
        self.last_decision = {"soc_before_frac": soc, "v_min_before_pu": min(voltages.values()),
                              "recharge_active": self.recharging, "price_low": low, "price_high": high,
                              "charge_limit_pu": charge_limit}
        return deepcopy(self.last_action)

    def _price_limits(self, price, dt, timestamp):
        c = self.config
        if not c.adaptive_prices:
            return c.price_low, c.price_high
        if timestamp is None or timestamp != self.last_price_timestamp:
            if timestamp is not None and self.last_price_timestamp is not None:
                gap = (datetime.fromisoformat(str(timestamp))
                       - datetime.fromisoformat(str(self.last_price_timestamp))).total_seconds() / 3600
                if gap < 0 or gap > dt + 1e-9:
                    self.price_history.clear()
            self.price_history.append(price)
            self.last_price_timestamp = timestamp
        window = max(1, math.ceil(c.price_window_hours / dt))
        while len(self.price_history) > window:
            self.price_history.popleft()
        if len(self.price_history) < window:
            return c.price_low, c.price_high
        prices = sorted(self.price_history)
        return tuple(prices[int((len(prices) - 1) * getattr(c, name))]
                     for name in ("price_low_quantile", "price_high_quantile"))

    def _voltage_power(self, error, rating):
        active_error = math.copysign(max(abs(error) - self.config.p_assist_error, 0), error)
        return -rating * self.config.p_gain * active_error, rating * self.config.q_gain * error

    def _balance_q(self, voltages, angles, rating):
        unit = {phase: rect(1, math.radians(_finite(angles[phase]))) for phase in voltages}
        a = rect(1, 2 * math.pi / 3)
        positive, negative = {"a": 1, "b": a, "c": a**2}, {"a": 1, "b": a**2, "c": a}
        v1 = sum(positive[p] * voltages[p] * unit[p] for p in voltages) / 3
        v2 = sum(negative[p] * voltages[p] * unit[p] for p in voltages) / 3
        if abs(v1) < 1e-6:
            raise ValueError("Positive-sequence voltage is too small for balance control")
        vuf_squared = abs(v2 / v1) ** 2
        # VUF gradient for a local inductive voltage response; the gain sets its scale.
        gradient = {p: ((v2.conjugate() * negative[p] - vuf_squared * v1.conjugate() * positive[p])
                        * unit[p]).real / abs(v1)**2 for p in voltages}
        mean = sum(gradient.values()) / len(gradient)
        return {p: -2 * rating * self.config.balance_gain * (value - mean)
                for p, value in gradient.items()}

    @staticmethod
    def _reference(value, phases):
        if isinstance(value, dict):
            if set(value) != set(phases):
                raise ValueError("Phase references must match the connected phases")
            return {phase: _finite(value[phase]) for phase in phases}
        return dict.fromkeys(phases, _finite(value) / len(phases))

    def _command(self, p, q):
        if self.parameters.dispatch_mode == "aggregate":
            return TotalAction(sum(p.values()), sum(q.values()))
        return PhaseAction(dict(p), dict(q))
