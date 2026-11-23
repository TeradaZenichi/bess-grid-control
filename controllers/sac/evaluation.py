"""Score SAC training snapshots and save the best policy."""
import json

import torch


class Evaluator:
    def __init__(self, trainer, output):
        self.trainer = trainer
        self.output = output
        self.checkpoint = output / "actor.pt"
        self.best = None
        device = trainer.device
        self.stream = torch.cuda.current_stream(device) if device.type == "cuda" else None

    def validate(self, actor, critic, episode, training_steps):
        trainer = self.trainer
        with torch.cuda.stream(self.stream):
            current = trainer.evaluate(actor=actor) | {"episode": episode, "training_steps": training_steps}
            if self.best is None or current["score"] > self.best["score"]:
                temporary = self.checkpoint.with_suffix(".tmp")
                torch.save({"actor": actor.state_dict(), "critic": critic,
                            "settings": vars(trainer.config), "scaling": vars(trainer.scaling), "evaluation": current}, temporary)
                temporary.replace(self.checkpoint)
                self.best = current
        with (self.output / "validation.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(current, allow_nan=False) + "\n")

    @staticmethod
    def energy_metrics(env, bess_id, cost):
        battery = next(device for device in env.bess_list if device.id == bess_id)
        deficit = (battery.soc_init_frac - battery.soc) * battery.e_cap_kwh
        # Settle missing energy as an import and surplus as an export at the closing tariffs.
        settlement = (max(deficit, 0.0) / battery.eta_charge * env.grid.prices[env.idx - 1]
                      + min(deficit, 0.0) * battery.eta_discharge * env.grid.sell_prices[env.idx - 1])
        return {"raw_cost": cost, "terminal_energy_cost": float(settlement), "adjusted_cost": float(cost + settlement),
                "soc_initial": battery.soc_init_frac, "soc_final": battery.soc,
                "charge_kwh": sum(battery.array_charge_kw) * env.dt, "discharge_kwh": sum(battery.array_discharge_kw) * env.dt,
                "voltage_violation_hours": sum(d["voltage_violations"] > 0 for d in env.results.diagnostics) * env.dt}
