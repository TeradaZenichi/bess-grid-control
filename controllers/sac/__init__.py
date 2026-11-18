"""SAC configuration shared by inference and training."""
from pathlib import Path

from ..base import Config as ControllerConfig
from .architectures import ARCHITECTURES


FOLDER = Path(__file__).parent


class Config(ControllerConfig):
    def __init__(self, settings=None, architecture_settings=None):
        common = ControllerConfig.controller("sac", settings)
        super().__init__(FOLDER / "architectures" / common.architecture / "config.json",
                         vars(common) | (architecture_settings or {}))

    def __setattr__(self, name, value):
        if name == "architecture" and value != self.architecture:
            self.__dict__.update(vars(ControllerConfig(FOLDER / "architectures" / value / "config.json")))
        super().__setattr__(name, value)

    @property
    def network(self):
        return ARCHITECTURES[self.architecture]

    @property
    def flatten(self):
        return self.network.flatten
