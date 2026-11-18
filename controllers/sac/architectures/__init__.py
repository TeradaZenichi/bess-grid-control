"""Available neural architectures."""
from .mlp import MLP
from .gru import GRU

ARCHITECTURES = {"mlp": MLP, "gru": GRU}
