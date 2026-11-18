"""Run controllers in the realistic five-bus environment."""
from pathlib import Path

from controllers import Controller
from controllers.base import Config
from controllers.normalization import ObservationScaling
from opendss_environment import Action, EnvironmentConfig, MicrogridEnv, PhaseAction, TotalAction
from tools.reporting import save_results


ROOT = Path(__file__).resolve().parent
CONFIG = Config.controller()
CASE = ROOT / CONFIG.case_path
OUTPUT = None
EPISODE_STEPS = 96
START_STEP = 0
SEED = CONFIG.seed


def main(controller_settings=None):
    env = MicrogridEnv(EnvironmentConfig(CASE, EPISODE_STEPS, start_step=START_STEP))
    observation, _ = env.reset(seed=SEED)
    policies = [Controller(parameters, controller_settings,
                           scaling=ObservationScaling.from_environment(env, parameters["bus"]))
                for parameters in env.data["devices"].get("bess", [])]

    for _ in range(EPISODE_STEPS):
        commands = {}
        for policy in policies:
            local = env.observe_controller(policy.bus, {"bess": [policy.id]},
                                           extra_features=policy.extra_features, observation=observation)
            commands[policy.id] = policy.act(local, {})
        action = Action(bess=commands, pv={
            device.id: PhaseAction(observation["pv"][device.id]["available_pu"])
            if device.dispatch_mode == "per_phase"
            else TotalAction(sum(observation["pv"][device.id]["available_pu"].values()))
            for device in env.pv_list
        }) if policies else None
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    results = env.get_episode_results()
    save_results(results, OUTPUT or Config.controller(settings=controller_settings).output_path("simulation"))
    return results


if __name__ == "__main__":
    main()
