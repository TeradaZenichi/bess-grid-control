"""Initial SAC settings, independent of network architecture and operating mode."""

ARCHITECTURE = "mlp"
MODE = "economic"
BESS_ID = None
BUS = None
EXTRA_FEATURES = ("grid",)
SEED = 42

LEARNING_RATE = 3e-4
GAMMA = 0.99
BATCH_SIZE = 256
