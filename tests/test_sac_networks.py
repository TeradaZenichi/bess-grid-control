import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from controllers.normalization import ObservationScaling
from controllers.sac.architectures import MLP
from controllers.sac.controller import SACController
from controllers.sac.networks import Actor, Critic, observation_batch
from controllers.sac.train import SACTrainer


ROOT = Path(__file__).resolve().parents[1]
BESS = json.loads((ROOT / "data/switzerland/5bus/devices.json").read_text())["bess"][0]
SCALING = ObservationScaling(s_base_kva=1000)


def state(architecture):
    if architecture == "mlp":
        return np.linspace(-1, 1, 264, dtype=np.float32)
    return {"history": np.linspace(-1, 1, 256, dtype=np.float32).reshape(8, 32),
            "mask": np.array([False] * 5 + [True] * 3)}


class SACNetworksTest(unittest.TestCase):
    def test_optimizer_groups_and_positive_learnable_alpha_reset_with_networks(self):
        trainer = SACTrainer()
        trainer.architecture(4, 2)
        self.assertAlmostEqual(trainer.alpha.item(), trainer.config.alpha, places=6)
        self.assertEqual(trainer.target_entropy, -2)
        for name, parameters in (("actor", trainer.actor.parameters()),
                                 ("critic", trainer.critic.parameters()), ("alpha", [trainer.log_alpha])):
            optimized = {id(p) for group in getattr(trainer.optimizer, name).param_groups for p in group["params"]}
            self.assertEqual(optimized, {id(p) for p in parameters})
            self.assertTrue(optimized.isdisjoint({id(p) for p in trainer.target.parameters()}))
        before = trainer.alpha.item()
        trainer.optimizer.alpha.zero_grad()
        (-trainer.log_alpha).backward()
        trainer.optimizer.alpha.step()
        self.assertGreater(trainer.alpha.item(), before)
        self.assertTrue(all(p.grad is None for p in trainer.actor.parameters()))
        trainer.architecture(4, 2)
        self.assertAlmostEqual(trainer.alpha.item(), trainer.config.alpha, places=6)
        self.assertEqual(len(trainer.optimizer.alpha.state), 0)

    def test_actor_twin_critics_shapes_gradients_and_independence(self):
        for architecture in ("mlp", "gru"):
            with self.subTest(architecture=architecture):
                controller = SACController(BESS, {"architecture": architecture}, scaling=SCALING)
                critics = [Critic(controller.architecture, controller.input_size, controller.action_size,
                                  controller.architecture_config) for _ in range(2)]
                observations = observation_batch([state(architecture)] * 4)
                actions, log_prob = controller.actor(observations)
                deterministic, _ = controller.actor(observations, deterministic=True)
                torch.testing.assert_close(controller.actor.deterministic_action(observations), deterministic,
                                           rtol=0, atol=0)
                values = [critic(observations, actions) for critic in critics]
                self.assertEqual(actions.shape, (4, 6))
                self.assertEqual(log_prob.shape, (4, 1))
                self.assertTrue(torch.all(actions.abs() <= 1))
                self.assertEqual([tuple(value.shape) for value in values], [(4, 1), (4, 1)])
                parameters = [list(controller.actor.parameters()), *(list(c.parameters()) for c in critics)]
                pointers = [{p.data_ptr() for p in group} for group in parameters]
                self.assertTrue(all(pointers[i].isdisjoint(pointers[j]) for i in range(3) for j in range(i)))
                (sum(value.square().mean() for value in values) + log_prob.mean()).backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for group in parameters for p in group))

    def test_actor_log_probability_remains_finite_when_tanh_saturates(self):
        actor = Actor(MLP, 4, 2, SimpleNamespace(hidden_sizes=[]))
        with torch.no_grad():
            actor.network[0].weight.zero_()
            actor.network[0].bias.copy_(torch.tensor([1000., -1000., 0., 0.]))
        action, log_prob = actor(torch.zeros(3, 4), deterministic=True)
        self.assertTrue(torch.isfinite(log_prob).all())
        torch.testing.assert_close(action, torch.tensor([[1., -1.]] * 3))

    def test_normalized_requests_convert_to_network_pu_in_action_order(self):
        controller = SACController(BESS, scaling=SCALING)
        command = controller.command(np.array([1, -1, 0, -1, 1, 0]))
        np.testing.assert_allclose(list(command.p_pu.values()), [0.04 / 3, -0.04 / 3, 0])
        np.testing.assert_allclose(list(command.q_pu.values()), [-0.05 / 3, 0.05 / 3, 0])
        aggregate = SACController(BESS | {"dispatch_mode": "aggregate"}, scaling=SCALING)
        command = aggregate.command(np.array([-1, 1]))
        self.assertEqual(aggregate.action_size, 2)
        self.assertAlmostEqual(command.p_pu, -0.04)
        self.assertAlmostEqual(command.q_pu, 0.05)


if __name__ == "__main__":
    unittest.main()
