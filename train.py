"""Run SAC training/HPO from the terminal or editor debugger."""
from argparse import ArgumentParser
from datetime import datetime, timezone

from controllers.sac.train import SACTrainer


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", action="store_true", help="Compare two discounts across three paired seeds")
    args = parser.parse_args()
    trainer = SACTrainer()
    if args.campaign:
        run = datetime.now(timezone.utc).strftime("trial_%Y%m%d_%H%M%S_%f")
        for gamma in (0.99, 0.999):
            for seed in (42, 43, 44):
                trainer.config.gamma, trainer.config.seed = gamma, seed
                trainer.train(trial=f"{run}_gamma_{gamma}")
    else:
        result = trainer.hpo()
