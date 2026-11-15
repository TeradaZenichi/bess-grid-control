"""Create a deterministic five-bus scenario using every exported EVSE."""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVSE_CONFIG = ROOT / "data" / "EV" / "evse_config.json"
OUTPUT = ROOT / "data" / "switzerland" / "5bus" / "devices_ev_all_asap.json"
BUSES = ("bus_002", "bus_003", "bus_004", "bus_005")
PHASES = ("a", "b", "c")
SINGLE_PHASE_MAX_KW = 7.4


def main():
    metadata = json.loads(EVSE_CONFIG.read_text(encoding="utf-8"))["evses"]
    chargers = sorted({str(values["source_charger_id"]) for values in metadata.values()})
    charger_index = {charger: index for index, charger in enumerate(chargers)}
    devices = []
    for evse_id, values in metadata.items():
        index = charger_index[str(values["source_charger_id"])]
        power = float(values["rated_power_kw"])
        phases = [PHASES[index % len(PHASES)]] if power <= SINGLE_PHASE_MAX_KW else list(PHASES)
        devices.append({
            "id": evse_id,
            "bus": BUSES[index % len(BUSES)],
            "phases": phases,
            "p_max_kw": power,
            "control_mode": "asap",
        })
    scenario = {
        "scenario_note": "Synthetic balanced placement; GridCharge does not provide feeder buses or phases.",
        "bess": [], "pv": [], "evse": devices,
    }
    OUTPUT.write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")
    print(f"Created {OUTPUT} with {len(devices)} ASAP EVSEs")


if __name__ == "__main__":
    main()
