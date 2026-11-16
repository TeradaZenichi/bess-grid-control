"""Run controllers in the realistic five-bus environment."""
from pathlib import Path

from opendss_environment import EnvironmentConfig, MicrogridEnv
from tools.reporting import save_results


ROOT = Path(__file__).resolve().parent
CASE = ROOT / "data" / "switzerland" / "5bus" / "config.json"
OUTPUT = ROOT / "results" / "5bus_realistic_ev"
EPISODE_STEPS = 96
START_STEP = 0
SEED = 42


def controller(observation):
    """Return the combined controller action; None keeps every EVSE in ASAP mode."""
    return None


def main():
    env = MicrogridEnv(EnvironmentConfig(CASE, EPISODE_STEPS, start_step=START_STEP))
    observation, _ = env.reset(seed=SEED)

    for _ in range(EPISODE_STEPS):
        action = controller(observation)
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    results = env.get_episode_results()
    save_results(results, OUTPUT)
    return results


if __name__ == "__main__":
    main()
