"""Fixed observation scales shared by neural controllers in one scenario."""
import json
from types import SimpleNamespace

import numpy as np


class ObservationScaling(SimpleNamespace):
    @classmethod
    def from_environment(cls, env, bus, meter_element=None):
        settings = json.loads(env.config_path.read_text(encoding="utf-8")).get("observation_scaling", {})
        meter = next(row for row in env.pre_action_branch_measurements
                     if row["bus"].lower() == bus.lower()
                     and (meter_element is None or row["element"].lower() == meter_element.lower()))
        dss = env.dss._dss
        active_element, active_bus = dss.CktElement.Name(), dss.Bus.Name()
        transformer = dss.Transformers.Name() if dss.Transformers.Count() else None
        winding = dss.Transformers.Wdg() if transformer else None
        try:
            dss.Circuit.SetActiveElement(meter["element"])
            phases = dss.CktElement.NumPhases()
            dss.Circuit.SetActiveBus(bus)
            voltage = dss.Bus.kVBase()
            if meter["element"].lower().startswith("transformer."):
                dss.Transformers.Name(meter["element"].split(".", 1)[1])
                dss.Transformers.Wdg(meter["terminal"])
                rating = dss.Transformers.kVA()
            else:
                rating = phases * voltage * meter["normal_current_a"]
            return cls(**settings, s_base_kva=env.data["base_kva"], bus=bus,
                       v_base_ln_kv=voltage, meter_element=meter["element"],
                       meter_terminal=meter["terminal"], meter_s_kva=rating, meter_phases=phases)
        finally:
            if transformer:
                dss.Transformers.Name(transformer)
                dss.Transformers.Wdg(winding)
            dss.Circuit.SetActiveElement(active_element)
            dss.Circuit.SetActiveBus(active_bus)

    def voltage(self, values):
        return (np.asarray(values) - 1) / self.voltage_deviation_ref_pu

    def power(self, values, rating_kva, phases):
        return np.asarray(values) * self.s_base_kva / (rating_kva / phases)

    def price(self, values):
        return np.asarray(values) / self.price_ref_per_kwh

    def vuf(self, percent):
        return percent / self.vuf_ref_pct
