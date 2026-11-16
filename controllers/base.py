"""Shared inference contract; training belongs to each implementation."""
from typing import Protocol

from opendss_environment import PhaseAction, TotalAction


class Controller(Protocol):
    def reset(self) -> None:
        """Clear policy memory at episode start without changing device state."""
        ...

    def act(self, observation: dict, objective: dict) -> TotalAction | PhaseAction:
        """Return one BESS command from its filtered pre-action observation."""
        ...
