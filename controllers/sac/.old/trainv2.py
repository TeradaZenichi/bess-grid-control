"""SAC with parallel CPU validation and optional overlap with training."""
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from multiprocessing import get_context

import torch
from opendss_environment import Action, EnvironmentConfig, MicrogridEnv

from .controller import SACController
from .train import SACTrainer, evaluate_windows, validation_result


_worker = None


def _init_worker(data, window, bess, settings, scaling, reward):
    global _worker
    torch.set_num_threads(1)
    env = MicrogridEnv(EnvironmentConfig(settings["case_path"], window["episode_steps"],
                                        start_step=window["start_step"], reward_function=reward), data=data)
    policy = SACController(bess, settings, scaling=scaling)
    policy.actor.eval()
    _worker = env, policy


def _evaluate_batch(windows, weights):
    env, policy = _worker
    policy.actor.load_state_dict(weights)
    scope = {"bess": [policy.id]}

    def policy_action(observation):
        local = env.observe_controller(policy.config.bus or policy.bus, scope, observation=observation,
                                       extra_features=policy.extra_features)
        return Action(bess={policy.id: policy.act(local)})

    return evaluate_windows(env, windows, policy.config.seed, policy.id, policy.reset, policy_action)


class SACTrainerV2(SACTrainer):
    def __init__(self, config=None, *, eval_workers=None):
        super().__init__(config)
        self.overlap_evaluation = self.device.type == "cuda"
        self.eval_workers = (1 if self.overlap_evaluation else 2) if eval_workers is None else eval_workers

    def train(self, *, trial=None):
        self._pool = self._collect = None
        with ExitStack() as self._resources:
            return super().train(trial=trial)

    def _episode_evaluations(self, episode):
        if self._collect is not None:
            yield self._collect()
        if self._pool is None:
            args = (self.data, self.validation[0], self.config.bess(self.data), vars(self.config),
                    self.scaling, self.env.train.reward_function)
            self._pool = self._resources.enter_context(ProcessPoolExecutor(
                max_workers=self.eval_workers, mp_context=get_context("spawn"), initializer=_init_worker, initargs=args))

        weights = {name: value.detach().to("cpu", copy=True) for name, value in self.actor.state_dict().items()}
        metadata = {"episode": episode, "training_steps": self.replay.total}
        size = (len(self.validation) + self.eval_workers - 1) // self.eval_workers
        jobs = [self._pool.submit(_evaluate_batch, self.validation[start:start + size], weights)
                for start in range(0, len(self.validation), size)]

        def collect():
            monthly = [row for job in jobs for row in job.result()]
            return validation_result(monthly) | metadata, weights

        self._collect = collect
        if not self.overlap_evaluation or episode == self.config.num_training_episodes:
            yield self._collect()
            self._collect = None


if __name__ == "__main__":
    SACTrainerV2().hpo()
