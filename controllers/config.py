"""Load JSON values into a simple configuration object."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace


def load_config(path, settings=None):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    return SimpleNamespace(**(values | deepcopy(settings or {})))
