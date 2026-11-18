"""Public BESS controller, inheriting only the configured implementation."""
from pathlib import Path

from .config import load_config

CONTROLLER_TYPE = load_config(Path(__file__).with_name("config.json")).controller_type

if CONTROLLER_TYPE == "rule_based":
    from .rule_based.controller import RuleBasedController as _SelectedController
elif CONTROLLER_TYPE == "sac":
    from .sac.controller import SACController as _SelectedController
else:
    raise ValueError(
        f"Unknown CONTROLLER_TYPE {CONTROLLER_TYPE!r}; expected 'rule_based' or 'sac'"
    )


class Controller(_SelectedController):
    """Expose the selected controller through a stable public import."""


__all__ = ["Controller"]
