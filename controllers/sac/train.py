"""SAC training/HPO contract and architecture construction."""
from pathlib import Path

from torch import nn

from ..config import load_config
from .architectures import ARCHITECTURES
from .networks import Actor, Critic


CONFIG = Path(__file__).with_name("config.json")


class SACTrainer:
    def __init__(self):
        self.config = load_config(CONFIG)

    def _trial_config(self, settings=None, architecture_settings=None):
        config = load_config(CONFIG, vars(self.config) | (settings or {}))
        architecture = load_config(CONFIG.parent / "architectures" / config.architecture / "config.json",
                                   architecture_settings)
        return config, architecture

    def architecture(self, input_size, action_size, settings=None, architecture_settings=None):
        """Create fresh actor/Q1/Q2 using the trial's architecture and dimensions."""
        config, network_config = self._trial_config(settings, architecture_settings)
        network = ARCHITECTURES[config.architecture]
        self.actor = Actor(network, input_size, action_size, network_config)
        self.critic = nn.ModuleDict({
            name: Critic(network, input_size, action_size, network_config) for name in ("q1", "q2")
        })
        return self.actor, self.critic

    def train(self, settings=None, architecture_settings=None):
        """Return status/configs and score: mean validation reward, or None before implementation."""
        config, network_config = self._trial_config(settings, architecture_settings)
        # Seed a fresh run; create environment, local scales and observation encoder.
        # Derive input_size/action_size from the encoder and controlled device.
        # self.architecture(input_size, action_size, vars(config), vars(network_config))
        # Create target critics, replay and optimizers; collect transitions and apply SAC updates.
        # Handle termination/truncation and reset history at each episode boundary.
        # Evaluate on validation periods/seeds; score is mean validation reward (higher is better).
        # Save weights, configs, scales and metrics; always close the environment.
        return {"status": "not_trained", "score": None,
                "settings": vars(config), "architecture_settings": vars(network_config)}

    def hpo(self, settings=None, architecture_settings=None):
        # Replace this single call with trials suggesting algorithm and architecture settings.
        # Each train() run must use fresh networks/replay and comparable budgets.
        result = self.train(settings, architecture_settings)
        # The study will maximize result["score"]; keep final test periods/seeds separate.
        return result


if __name__ == "__main__":
    SACTrainer().hpo()
