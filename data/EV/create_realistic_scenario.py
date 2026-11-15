"""Create a diversified 24-EVSE scenario for the five-bus feeder."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVSE_CONFIG = ROOT / "data" / "EV" / "evse_config.json"
OUTPUT = ROOT / "data" / "switzerland" / "5bus" / "devices_ev_realistic.json"
SLOW_BUSES = ("bus_002",) * 3 + ("bus_003",) * 5 + ("bus_004",) * 5 + ("bus_005",) * 5
PHASES = ("a", "b", "c")


def select(metadata, predicate, count, used):
    candidates = sorted(
        ((name, values) for name, values in metadata.items() if predicate(values)),
        key=lambda item: (-int(item[1]["train_n"]), item[0]),
    )
    selected, chargers = [], set()
    for name, values in candidates:
        charger = str(values["source_charger_id"])
        if name in used or charger in chargers:
            continue
        selected.append((name, values))
        chargers.add(charger)
        used.add(name)
        if len(selected) == count:
            return selected
    raise ValueError("Not enough distinct EVSEs for the requested category")


def device(item, bus, phases):
    name, values = item
    return {
        "id": name, "bus": bus, "phases": list(phases),
        "p_max_kw": float(values["rated_power_kw"]), "control_mode": "asap",
    }


def main():
    metadata = json.loads(EVSE_CONFIG.read_text(encoding="utf-8"))["evses"]
    used = set()
    slow = select(metadata, lambda x: x["connector_type"] == "Type 2" and x["rated_power_kw"] <= 7.4, 18, used)
    ac = select(metadata, lambda x: x["connector_type"] == "Type 2" and 20 <= x["rated_power_kw"] <= 22, 4, used)
    dc = select(metadata, lambda x: x["connector_type"] in {"CCS", "CHAdeMO"} and x["rated_power_kw"] >= 43, 2, used)
    devices = [device(item, bus, (PHASES[index % 3],)) for index, (item, bus) in enumerate(zip(slow, SLOW_BUSES))]
    devices += [device(item, bus, PHASES) for item, bus in zip(ac, ("bus_002", "bus_003", "bus_004", "bus_005"))]
    devices += [device(item, "bus_002", PHASES) for item in dc]
    scenario = {
        "scenario_note": "Synthetic 24-EVSE placement: 18x7 kW, 4x22 kW and 2x50/51 kW.",
        "bess": [], "pv": [], "evse": devices,
    }
    OUTPUT.write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")
    print(f"Created {OUTPUT} with {len(devices)} ASAP EVSEs")


if __name__ == "__main__":
    main()
