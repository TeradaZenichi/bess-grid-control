"""Score SAC training snapshots and save the best policy."""
import json

from concurrent.futures import ProcessPoolExecutor, wait
from multiprocessing import get_context

import torch
from tqdm import tqdm

from .actions import ActivePowerActions, bess_actions
from opendss_environment import EnvironmentConfig, MicrogridEnv, Reward

from . import Config
from .networks import Actor, observation_batch
from .observation import LocalObservation


class Evaluator:
    def __init__(self, trainer, output):
        self.trainer = trainer
        self.output = output
        self.checkpoint = output / "actor.pt"
        self.best = None
        device = trainer.device
        self.stream = torch.cuda.current_stream(device) if device.type == "cuda" else None

    def validate(self, actor, critic, episode, training_steps):
        trainer = self.trainer
        with torch.cuda.stream(self.stream):
            if trainer.eval_workers:
                current, _ = next(trainer.parallel_eval.evaluate_episode(actor, episode, training_steps, final=True))
            else:
                current = trainer.evaluate(actor=actor) | {"episode": episode, "training_steps": training_steps}
            if self.best is None or current["score"] > self.best["score"]:
                temporary = self.checkpoint.with_suffix(".tmp")
                torch.save({"actor": actor.state_dict(), "critic": critic,
                            "settings": vars(trainer.config), "scaling": vars(trainer.scaling), "evaluation": current}, temporary)
                temporary.replace(self.checkpoint)
                self.best = current
        with (self.output / "validation.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(current, allow_nan=False) + "\n")

    @staticmethod
    def energy_metrics(env, bess_id, cost):
        battery = next(device for device in env.bess_list if device.id == bess_id)
        deficit = (battery.soc_init_frac - battery.soc) * battery.e_cap_kwh
        # Settle missing energy as an import and surplus as an export at the closing tariffs.
        settlement = (max(deficit, 0.0) / battery.eta_charge * env.grid.prices[env.idx - 1]
                      + min(deficit, 0.0) * battery.eta_discharge * env.grid.sell_prices[env.idx - 1])
        return {"raw_cost": cost, "terminal_energy_cost": float(settlement), "adjusted_cost": float(cost + settlement),
                "soc_initial": battery.soc_init_frac, "soc_final": battery.soc,
                "charge_kwh": sum(battery.array_charge_kw) * env.dt, "discharge_kwh": sum(battery.array_discharge_kw) * env.dt,
                "voltage_violation_hours": sum(d["voltage_violations"] > 0 for d in env.results.diagnostics) * env.dt}


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
                                start_step=first["start_step"], reward_function=self.reward, observe_evse=self.config.observe_evse), data=self.data)
        self.scope = {"bess": [self.config.bess_id]}
        bess = self.config.bess(self.data)
        self.observations = LocalObservation(bess, self.scaling, self.config.history_steps,
                                             self.config.observation_timezone)
        self.actions = bess_actions(bess, self.scaling.s_base_kva, self.config, self.data["phases"])

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
            if self.config.reward_term in ("node_cost", "feeder_cost"):
                monthly[-1].update(Evaluator.energy_metrics(self.env, self.config.bess_id, -reward_sum))
            if completed is not None:
                with completed.get_lock():
                    completed.value += 1
        return monthly

    @staticmethod
    def _result(monthly):
        return {"score": sum(row["reward_sum"] - row.get("terminal_energy_cost", 0.0) for row in monthly) / sum(row["steps"] for row in monthly),
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
        if isinstance(evaluator.actions, ActivePowerActions):
            actor.power_actions = evaluator.actions
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
