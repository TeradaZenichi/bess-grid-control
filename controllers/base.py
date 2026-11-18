"""Shared inference contract; training belongs to each implementation."""
from abc import ABC, abstractmethod
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from opendss_environment import PhaseAction, TotalAction


ROOT = Path(__file__).resolve().parents[1]
FOLDER = Path(__file__).parent


class Config(SimpleNamespace):
    """JSON values with shared controller settings and reproducible run paths."""

    def __init__(self, path, settings=None):
        super().__init__(**(json.loads(Path(path).read_text(encoding="utf-8")) | deepcopy(settings or {})))
        if hasattr(self, "case_path"):
            self.case_path = str((ROOT / self.case_path).resolve())

    @classmethod
    def controller(cls, controller_type=None, settings=None):
        common = cls(FOLDER / "config.json")
        kind = controller_type or common.controller_type
        return cls(FOLDER / kind / "config.json",
                   vars(common) | (settings or {}) | {"controller_type": kind})

    def bess(self, data):
        return next(device for device in data["devices"]["bess"] if device["id"] == self.bess_id)

    def output_path(self, trial, root=None):
        case = Path(self.case_path).parent
        case = case.relative_to(ROOT) if case.is_relative_to(ROOT) else case
        case_id = "_".join(case.parts[1:]) if case.parts[:1] == ("data",) else case.name
        folder = Path(root) if root is not None else ROOT / "results/controllers"
        folder = folder / case_id / self.controller_type / self.mode / self.reward_term
        if self.controller_type == "sac":
            folder /= self.architecture
        return folder / trial / f"seed_{self.seed}"


class BaseController(ABC):
    def reset(self) -> None:
        """Clear policy memory at episode start without changing device state."""

    @abstractmethod
    def act(self, observation: dict, objective: dict) -> TotalAction | PhaseAction:
        """Return one BESS command from its filtered pre-action observation."""
        raise NotImplementedError
