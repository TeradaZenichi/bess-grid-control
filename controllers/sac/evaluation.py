"""Sequential and parallel validation for SAC policies."""
from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import get_context

import torch
from tqdm import tqdm

from opendss_environment import EnvironmentConfig, MicrogridEnv, Reward

from . import Config
from .networks import Actor, observation_batch
from .observation import LocalObservation


class SACEvaluator:
    _worker = None

    def __init__(self, config, data, windows, scaling, reward, *, workers=0, overlap=False):
        self.config, self.data, self.windows = config, data, windows
        self.scaling, self.reward = scaling, reward
        self.workers, self.overlap = workers, overlap
        self.env = self._pool = self._pending = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = self._pending = None
        if self.env is not None:
            self.env.close()
            self.env = None

    def _setup_local(self):
        first = self.windows[0]
        self.env = MicrogridEnv(EnvironmentConfig(self.config.case_path, first["episode_steps"],
                                start_step=first["start_step"], reward_function=self.reward), data=self.data)
        self.scope = {"bess": [self.config.bess_id]}
        bess = self.config.bess(self.data)
        self.observations = LocalObservation(bess, self.scaling, self.config.history_steps,
                                             self.config.observation_timezone)
        self.actions = self.env.controller_actions(self.scope, seed=self.config.seed)

    def encode_local(self, observation):
        local = self.env.observe_controller(self.config.bus or self.observations.bus, self.scope,
                                            observation=observation, extra_features=self.config.extra_features)
        return self.observations.encode(local, flatten=self.config.flatten)

    @torch.no_grad()
    def _evaluate_windows(self, actor, windows, *, show_progress=False, completed=None):
        device = next(actor.parameters()).device
        monthly = []
        progress_options = {} if show_progress else {"disable": True}
        for window in windows:
            observation, _ = self.env.reset(seed=self.config.seed, options={"start_step": window["start_step"],
                                            "episode_steps": window["episode_steps"]})
            self.observations.reset()
            reward_sum = 0.0
            terminated = truncated = False
            with tqdm(total=window["episode_steps"], desc=f'Validation month {window["month"]}', unit="step",
                      position=1, leave=False, **progress_options) as progress:
                while not (terminated or truncated):
                    batch = observation_batch([self.encode_local(observation)])
                    batch = {name: value.to(device) for name, value in batch.items()} if isinstance(batch, dict) else batch.to(device)
                    action, _ = actor(batch, deterministic=True)
                    command = self.env.compose_actions(self.actions.command(action[0].cpu().numpy()), observation=observation)
                    observation, _, terminated, truncated, info = self.env.step(command, full_info=False)
                    reward_sum += Reward.get_local(info["rewards"], self.config.bess_id)
                    progress.update()
            steps = self.env.idx
            monthly.append({"month": window["month"], "steps": steps, "reward_sum": reward_sum,
                            "mean_reward": reward_sum / steps})
            if completed is not None:
                with completed.get_lock():
                    completed.value += 1
        return monthly

    @staticmethod
    def _result(monthly):
        return {"score": sum(row["reward_sum"] for row in monthly) / sum(row["steps"] for row in monthly),
                "monthly": monthly}

    def evaluate(self, actor):
        if self.env is None:
            self._setup_local()
        return self._result(self._evaluate_windows(actor, self.windows, show_progress=True))

    @staticmethod
    def _init_worker(settings, data, windows, scaling, reward, completed):
        torch.set_num_threads(1)
        config = Config(settings)
        evaluator = SACEvaluator(config, data, windows, scaling, reward)
        evaluator._setup_local()
        observations = evaluator.observations
        input_size = observations.flat_size if config.flatten else observations.shape[1]
        actor = Actor(config.network, input_size, evaluator.actions.action_space.shape[0], config).eval()
        SACEvaluator._worker = evaluator, actor, completed

    @staticmethod
    def _evaluate_batch(windows, weights):
        evaluator, actor, completed = SACEvaluator._worker
        actor.load_state_dict(weights)
        return evaluator._evaluate_windows(actor, windows, completed=completed)

    def _collect_pending(self):
        jobs, metadata, weights = self._pending
        self._pending = None
        with tqdm(total=len(self.windows), initial=self._completed.value, desc=f'Validation episode {metadata["episode"]}',
                  unit="window", position=1, leave=False) as progress:
            while wait(jobs, timeout=0.2).not_done:
                progress.update(self._completed.value - progress.n)
            monthly = [row for job in jobs for row in job.result()]
            progress.update(self._completed.value - progress.n)
        return self._result(monthly) | metadata, weights

    def evaluate_episode(self, actor, episode, training_steps, *, final=False):
        metadata = {"episode": episode, "training_steps": training_steps}
        if self.workers == 0:
            yield self.evaluate(actor) | metadata, actor.state_dict()
            return
        if self._pending is not None:
            yield self._collect_pending()
        if self._pool is None:
            context = get_context("spawn")
            self._completed = context.Value("i", 0)
            args = (vars(self.config), self.data, self.windows, self.scaling, self.reward, self._completed)
            self._pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=context,
                                             initializer=self._init_worker, initargs=args)

        weights = {name: value.detach().to("cpu", copy=True) for name, value in actor.state_dict().items()}
        self._completed.value = 0
        size = (len(self.windows) + self.workers - 1) // self.workers
        jobs = [self._pool.submit(self._evaluate_batch, self.windows[start:start + size], weights)
                for start in range(0, len(self.windows), size)]
        self._pending = jobs, metadata, weights
        if not self.overlap or final:
            yield self._collect_pending()
