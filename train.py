"""Run SAC training/HPO from the terminal or editor debugger."""
from controllers.sac.train import SACTrainer


if __name__ == "__main__":
    trainer = SACTrainer()
    result = trainer.hpo()
