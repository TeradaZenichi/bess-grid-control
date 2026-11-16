"""Shared inference contract; training belongs to each implementation."""
from abc import ABC, abstractmethod

from opendss_environment import PhaseAction, TotalAction


class BaseController(ABC):
    def reset(self) -> None:
        """Clear policy memory at episode start without changing device state."""

    @abstractmethod
    def act(self, observation: dict, objective: dict) -> TotalAction | PhaseAction:
        """Return one BESS command from its filtered pre-action observation."""
        raise NotImplementedError
