"""SAC training/HPO contract and architecture construction."""
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from contextlib import ExitStack
from datetime import datetime, timezone
from types import SimpleNamespace
from copy import deepcopy
import json
from time import perf_counter

from torch import nn
import numpy as np
import torch
from tqdm import tqdm

from opendss_environment import EnvironmentConfig, MicrogridEnv, Reward
from opendss_environment.data import load_data

from .networks import Actor, Critic, observation_batch
from .actions import ActivePowerActions, bess_actions
from ..normalization import ObservationScaling
from .replay import ReplayBuffer
from torch.optim.swa_utils import get_ema_multi_avg_fn
from .evaluation import Evaluator, SACEvaluator
from .collection import Collector
from .observation import LocalObservation
from ..episodes import split_episodes
from . import Config


class SACTrainer:
    def __init__(self, config=None, *, eval_workers=0):
        self.eval_workers = eval_workers
        self.config = config if config is not None else Config()
        self.device = torch.device(self.config.device)
        torch.set_num_threads(self.config.cpu_threads)
        self.data   = load_data(self.config.case_path)
        self.episodes, self.validation = split_episodes(self.data)

    def architecture(self, input_size, action_size):
        self.actor        = Actor(self.config.network, input_size, action_size, self.config)
        if isinstance(getattr(self, "actions", None), ActivePowerActions):
            self.actor.power_actions = self.actions
        self.critic       = nn.Module()
        self.critic.q1    = Critic(self.config.network, input_size, action_size, self.config)
        self.critic.q2    = Critic(self.config.network, input_size, action_size, self.config)
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target       = nn.Module()
        self.target.q1    = deepcopy(self.critic.q1)
        self.target.q2    = deepcopy(self.critic.q2)
        self.target.requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(self.config.alpha, dtype=torch.float32, device=self.device).log())
        self.target_entropy = -float(action_size)
        self.optimizer = SimpleNamespace(
            actor=torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr),
            critic=torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr),
            alpha=torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr),
        )
        return self.actor, self.critic

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _to_device(self, value):
        if isinstance(value, dict):
            return {key: self._to_device(tensor) for key, tensor in value.items()}
        return value.to(self.device, non_blocking=True)

    def encode_local(self, observation, *, evaluation=False):
        env = self.env.eval if evaluation else self.env.train
        encoder = self.eval_observations if evaluation else self.observations
        local = env.observe_controller(
            self.config.bus or encoder.bus, {"bess": [self.config.bess_id]},
            observation=observation, extra_features=self.config.extra_features,
        )
        return encoder.encode(local, flatten=self.config.flatten)

    def train(self, *, trial=None):
        """Train on chronological blocks and score the actor on held-out windows."""
        run_started, validation_wait = perf_counter(), 0.0
        trial = trial or datetime.now(timezone.utc).strftime("trial_%Y%m%d_%H%M%S_%f")
        output = self.config.output_path(trial)
        output.mkdir(parents=True, exist_ok=False)
        torch.manual_seed(self.config.seed)
        self.env = SimpleNamespace(
            train=MicrogridEnv(EnvironmentConfig(self.config.case_path, self.episodes[0]["episode_steps"],
                                                observe_evse=self.config.observe_evse), data=self.data),
            eval=MicrogridEnv(EnvironmentConfig(self.config.case_path, self.validation[0]["episode_steps"],
                                               start_step=self.validation[0]["start_step"], observe_evse=self.config.observe_evse), data=self.data),
        )
        start, steps = self.episodes[0]["start_step"], self.episodes[0]["episode_steps"]
        bess = self.config.bess(self.env.train.data)
        observation, _      = self.env.train.reset(seed=self.config.seed, options={"start_step": start, "episode_steps": steps})
        initial_timestamp = observation["timestamp"]
        self.scaling        = ObservationScaling.from_environment(self.env.train, bess["bus"])
        self.observations   = LocalObservation(bess, self.scaling, self.config.history_steps, self.config.observation_timezone)
        self.eval_observations = LocalObservation(bess, self.scaling, self.config.history_steps, self.config.observation_timezone)
        self.initial_state  = self.encode_local(observation)
        input_size          = self.observations.flat_size if self.config.flatten else self.observations.shape[1]
        self.config.dt_h = self.env.train.dt
        self.actions = bess_actions(bess, self.scaling.s_base_kva, self.config, self.data["phases"])
        action_size = self.actions.action_space.shape[0]

        self.architecture(input_size, action_size)

        count = self.config.collectors
        if type(count) is not int or count not in (1, 2):
            raise ValueError("collectors must be 1 or 2")
        replays = []
        for index in range(count):
            settings = deepcopy(self.config)
            settings.replay_buffer_size = (settings.replay_buffer_size + index) // count
            settings.seed += index
            replays.append(ReplayBuffer(settings, self.initial_state, action_size))
        self.replay = replays[0]
        rng = np.random.default_rng(self.config.seed)
        self.env.train.reward_function = Reward(controllers={self.config.bess_id: {"devices": {"bess": [self.config.bess_id]}, "buses": [bess["bus"]], "weights": {self.config.reward_term: 1.0}}})

        evaluator = Evaluator(self, output)
        pending = None
        with ExitStack() as resources:
            resources.callback(self.env.train.close)
            resources.callback(self.env.eval.close)
            if self.eval_workers:
                self.parallel_eval = resources.enter_context(SACEvaluator(self.config, self.data, self.validation, self.scaling,
                    self.env.train.reward_function, workers=self.eval_workers))
            executor = resources.enter_context(ThreadPoolExecutor(max_workers=1))
            local = Collector(self.env.train, self.observations, self.actions, self.config)
            pools = []
            if count > 1:
                for _ in replays:
                    pools.append(resources.enter_context(ProcessPoolExecutor(1, mp_context=get_context("spawn"),
                        initializer=Collector.initialize, initargs=(self.config, self.scaling, self.env.train.reward_function, self.data))))
            queues = self.episode_queues(count)
            steps = sum(max(1, w["episode_steps"] - 1) for queue in queues for w in queue)
            progress = resources.enter_context(tqdm(total=steps, desc="Training transitions", unit="step", position=1, leave=False))
            rows = self.collect(local, pools, replays, rng, progress, queues)
            for completed, (row, updated) in enumerate(tqdm(rows, total=self.config.num_training_episodes, desc="Train episodes"), 1):
                row["episode"] = completed
                with (output / "training.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")

                if updated:
                    if pending is not None:
                        wait_started = perf_counter()
                        pending.result()
                        validation_wait += perf_counter() - wait_started
                    pending = executor.submit(evaluator.validate, deepcopy(self.actor).eval(), deepcopy(self.critic.state_dict()),
                                              completed, row["total_steps"])

            if pending is not None:
                wait_started = perf_counter()
                pending.result()
                validation_wait += perf_counter() - wait_started

        evaluation = evaluator.best
        result = {"status": "trained" if evaluation else "episodes_ready", "score": evaluation["score"] if evaluation else None,
                  "elapsed_seconds": perf_counter() - run_started, "validation_wait_seconds": validation_wait,
                  "episodes": self.episodes, "validation": self.validation, "evaluation": evaluation,
                  "initial_timestamp": initial_timestamp, "input_size": input_size, "action_size": action_size,
                  "settings": deepcopy(vars(self.config)), "flatten": self.config.flatten,
                  "trial": trial, "output_path": str(output), "checkpoint": str(evaluator.checkpoint) if evaluation else None}
        (output / "initialization.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        return result

    def episode_queues(self, count):
        """Divide each pass through the training year into contiguous seasonal blocks."""
        queues = [[] for _ in range(count)]
        budget = self.config.num_training_episodes
        for start in range(0, budget, len(self.episodes)):
            windows = self.episodes[:budget - start]
            width = (len(windows) + count - 1) // count
            for i, queue in enumerate(queues):
                queue.extend(windows[i * width:(i + 1) * width])
        return queues

    @torch.no_grad()
    def collection_actions(self, states, total):
        warmup = max(0, self.config.warmup_steps - total)
        actions = [self.actor.project(state, self.actions.action_space.sample()) for state in states[:warmup]]
        if states[warmup:]:
            batch = self._to_device(observation_batch(states[warmup:]))
            sampled, _ = self.actor(batch)
            actions.extend(sampled.cpu().numpy())
        return actions

    def collect(self, local, pools, replays, rng, progress, queues):
        windows = [iter(queue) for queue in queues]
        states, rows = {}, {}
        total, updated = 0, False
        while True:
            for i in range(len(windows)):
                if i in states:
                    continue
                window = next(windows[i], None)
                if window is None:
                    continue
                states[i] = pools[i].submit(Collector.run, "reset", window).result() if pools else local.reset(window)
                replays[i].start_episode()
                rows[i] = dict(start_step=window["start_step"], steps=0, updates=0, reward_sum=0., started=perf_counter(), metrics={})
            if not states:
                return
            ids = list(states)
            actions = self.collection_actions(list(states.values()), total)
            jobs = [pools[i].submit(Collector.run, "step", action) for i, action in zip(ids, actions)] if pools else []
            finished = []
            due = [i for step, i in enumerate(ids, total + 1)
                   if step >= self.config.learning_starts and step % self.config.update_every == 0]
            # Run the learner while both DSS steps are in flight.
            if jobs:
                for i in due.copy():
                    if self.replay_update(rows[i], replays, rng):
                        due.remove(i)
                        updated = True
            for j, (i, action) in enumerate(zip(ids, actions)):
                total += 1
                row = rows[i]
                state, reward, terminated, truncated = jobs[j].result() if jobs else local.step(action)
                replays[i].add(states[i], action, reward, state, terminated, truncated)
                if i in due:
                    updated |= self.replay_update(row, replays, rng)
                states[i] = state
                row["reward_sum"] += reward
                row["steps"] += 1
                if terminated or truncated:
                    del states[i]
                    finished.append(row)
            progress.update(len(ids))
            for row in finished:
                row.update(total_steps=total, mean_reward=row["reward_sum"] / row["steps"],
                           elapsed_seconds=perf_counter() - row.pop("started"))
                metrics = row.pop("metrics")
                if metrics and self.device.type == "cuda":
                    metrics = dict(zip(metrics, torch.stack(list(metrics.values())).tolist()))
                for name in ("critic_loss", "actor_loss", "alpha_loss", "alpha"):
                    row["mean_" + name] = metrics[name] / row["updates"] if row["updates"] else None
                yield row, updated

    def replay_update(self, row, replays, rng):
        sizes = [len(buffer) for buffer in replays]
        available = sum(sizes)
        if available < self.config.batch_size:
            return False
        index = rng.choice(len(sizes), p=np.asarray(sizes) / available) if len(sizes) > 1 else 0
        batch = replays[index].sample()
        metrics = self.update(batch, device_metrics=True) if self.device.type == "cuda" else self.update(batch)
        for name, value in metrics.items():
            row["metrics"][name] = row["metrics"].get(name, 0.) + value
        row["updates"] += 1
        return True

    def update(self, batch, *, device_metrics=False):
        batch = {key: self._to_device(batch[key]) for key in ("state", "next_state", "action", "reward", "discount")}
        # Bellman target
        with torch.no_grad():
            next_action, next_log_prob = self.actor(batch["next_state"])
            next_q1 = self.target.q1(batch["next_state"], next_action)
            next_q2 = self.target.q2(batch["next_state"], next_action)
            next_value = torch.minimum(next_q1, next_q2) - self.alpha * next_log_prob
            backup = batch["reward"] + batch["discount"] * next_value

        # Critic Update
        q1 = self.critic.q1(batch["state"], batch["action"])
        q2 = self.critic.q2(batch["state"], batch["action"])
        critic_loss = (q1 - backup).square().mean() + (q2 - backup).square().mean()
        self.optimizer.critic.zero_grad()
        critic_loss.backward()
        self.optimizer.critic.step()

        # Actor Update
        self.critic.requires_grad_(False)
        action, log_prob = self.actor(batch["state"])
        q1 = self.critic.q1(batch["state"], action)
        q2 = self.critic.q2(batch["state"], action)
        actor_loss = (self.alpha.detach() * log_prob - torch.minimum(q1, q2)).mean()
        self.optimizer.actor.zero_grad()
        actor_loss.backward()
        self.optimizer.actor.step()
        self.critic.requires_grad_(True)

        # Entropy temperature
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.optimizer.alpha.zero_grad()
        alpha_loss.backward()
        self.optimizer.alpha.step()

        # Move each target critic towards its online counterpart.
        update_ema = get_ema_multi_avg_fn(decay=1 - self.config.tau)
        with torch.no_grad():
            update_ema(list(self.target.q1.parameters()), list(self.critic.q1.parameters()), 0)
            update_ema(list(self.target.q2.parameters()), list(self.critic.q2.parameters()), 0)

        metrics = {"critic_loss": critic_loss.detach(), "actor_loss": actor_loss.detach(),
                   "alpha_loss": alpha_loss.detach(), "alpha": self.log_alpha.detach().exp()}
        return metrics if device_metrics else {name: value.item() for name, value in metrics.items()}

    @torch.no_grad()
    def evaluate(self, *, actor=None):
        actor = self.actor if actor is None else actor
        env = self.env.eval
        env.reward_function = self.env.train.reward_function
        monthly = []
        for window in self.validation:
            observation, _ = env.reset(seed=self.config.seed, options={"start_step": window["start_step"], "episode_steps": window["episode_steps"]})
            self.eval_observations.reset()
            reward_sum = 0.0
            terminated = truncated = False
            with tqdm(total=window["episode_steps"], desc=f'Validation month {window["month"]}', unit="step", position=2, leave=False) as progress:
                while not (terminated or truncated):
                    state = self.encode_local(observation, evaluation=True)
                    action = actor.deterministic_action(self._to_device(observation_batch([state])))
                    command = env.compose_actions(self.actions.command(action[0].cpu().numpy()), observation=observation)
                    observation, _, terminated, truncated, info = env.step(command, full_info=False)
                    reward_sum += Reward.get_local(info["rewards"], self.config.bess_id)
                    progress.update()

            row = {"month": window["month"], "steps": env.idx, "reward_sum": reward_sum, "mean_reward": reward_sum / env.idx}
            if self.config.reward_term in ("node_cost", "feeder_cost"):
                row.update(Evaluator.energy_metrics(env, self.config.bess_id, -reward_sum))
            monthly.append(row)
        return {"score": sum(row["reward_sum"] - row.get("terminal_energy_cost", 0.0) for row in monthly) / sum(row["steps"] for row in monthly),
                "monthly": monthly}

    def hpo(self, *, trial=None):
        # Create the objective, then assign trial suggestions directly to self.config.
        # Set architecture first (loads its defaults), then history/layer/algorithm settings.
        # Each train() run must use fresh networks/replay and comparable budgets.
        result = self.train(trial=trial)
        # The study will maximize result["score"]; keep final test periods/seeds separate.
        return result


if __name__ == "__main__":
    SACTrainer().hpo()
