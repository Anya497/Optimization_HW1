import math
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data.common import make_loaders
from src.data.semicircles import sample_points as sample_semicircles
from src.data.log_regression import sample_points as sample_logreg
from src.grad_noise import per_example_gradients
from src.models import SemicircleNet
from src.optimizer import RobustSGD
from src.train import evaluate, train


class LinearLogit(nn.Module):
    """Small test fixture, independent of the simulation 2 architecture."""
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)
        nn.init.zeros_(self.layer.weight)
        nn.init.zeros_(self.layer.bias)

    def forward(self, x):
        return self.layer(x).squeeze(-1)


class RecordingRobustSGD(RobustSGD):
    """Record inputs while delegating every update to the real optimizer."""
    def __init__(self, parameters, lr=0.1, rho=0):
        super().__init__(parameters, lr=lr, rho=rho)
        self.observations = []
        self.mean_gradients = []
        self.learning_rates = []

    def step(self, closure=None, noisy_grads=None):
        parameters = [p for group in self.param_groups for p in group["params"]]
        self.observations.append(torch.cat([
            noisy_grads[p].reshape(noisy_grads[p].shape[0], -1) for p in parameters
        ], dim=1).clone())
        self.mean_gradients.append(torch.cat([
            noisy_grads[p].mean(dim=0).flatten() for p in parameters
        ]).clone())
        self.learning_rates.append([group["lr"] for group in self.param_groups])
        return super().step(closure=closure, noisy_grads=noisy_grads)


class TrainingTests(unittest.TestCase):
    def test_per_example_gradients_match_individual_backprop(self):
        model = LinearLogit()
        x, y = torch.tensor([[1.0, 2.0], [-2.0, 3.0]]), torch.tensor([0.0, 1.0])
        actual = per_example_gradients(model, x, y)
        for index in range(len(x)):
            expected = torch.autograd.grad(F.binary_cross_entropy_with_logits(model(x[index]), y[index]),
                                           tuple(model.parameters()))
            for parameter, gradient in zip(model.parameters(), expected):
                torch.testing.assert_close(actual[parameter][index], gradient)

    def test_zero_lr_and_corruption_leave_data_unchanged(self):
        loaders = make_loaders(sample_semicircles, train_size=20, validation_size=10, test_size=10,
                               batch_size=4, updates=3)
        original = [tensor.clone() for tensor in loaders["train"].dataset.tensors]
        model = LinearLogit()
        optimizer = RecordingRobustSGD(model.parameters(), lr=0)
        history, metrics = train(model, loaders, optimizer, eval_every=2, gradient_noise=0.5)
        self.assertEqual([row["update"] for row in history], [0, 2, 3])
        self.assertEqual(metrics["gradient_evaluations"], 12)
        self.assertEqual(metrics["zero_step_frequency"], 1)
        self.assertEqual(history[0]["train_loss"], history[-1]["train_loss"])
        for before, after in zip(original, loaders["train"].dataset.tensors):
            torch.testing.assert_close(before, after)
        clean_x, clean_y = next(iter(make_loaders(sample_semicircles, train_size=20, validation_size=10,
                                               test_size=10, batch_size=4, updates=3)["train"]))
        clean = per_example_gradients(model, clean_x, clean_y)
        generator = torch.Generator().manual_seed(1)
        expected = torch.cat([
            (gradients + 0.5 * torch.randn(gradients.shape, generator=generator)).reshape(4, -1)
            for gradients in clean.values()
        ], dim=1)
        torch.testing.assert_close(optimizer.observations[0], expected)
        torch.testing.assert_close(optimizer.mean_gradients[0], expected.mean(dim=0))

    def test_reproducible_training_and_group_learning_rate_schedule(self):
        histories = []
        for _ in range(2):
            loaders = make_loaders(sample_semicircles, train_size=20, validation_size=10, test_size=10,
                                   batch_size=4, updates=4)
            model = LinearLogit()
            optimizer = RecordingRobustSGD([
                {"params": [model.layer.weight], "lr": 0.1},
                {"params": [model.layer.bias], "lr": 0.05},
            ], lr=0.1)
            history, _ = train(model, loaders, optimizer, eval_every=2, gradient_noise=0.2)
            histories.append(history)
        self.assertEqual(histories[0], histories[1])
        self.assertEqual(optimizer.learning_rates[0], [0.1, 0.05])
        self.assertLess(optimizer.learning_rates[1][0], 0.1)
        for weight_lr, bias_lr in optimizer.learning_rates:
            self.assertAlmostEqual(weight_lr, 2 * bias_lr)

    def test_optimizer_interface_matches_mean_loss_backpropagation(self):
        loaders = make_loaders(sample_semicircles, train_size=20, validation_size=10, test_size=10,
                               batch_size=4, updates=1)
        matching = make_loaders(sample_semicircles, train_size=20, validation_size=10, test_size=10,
                                batch_size=4, updates=1)
        x, y = next(iter(matching["train"]))
        model = LinearLogit()
        gradients = torch.autograd.grad(F.binary_cross_entropy_with_logits(model(x), y),
                                        tuple(model.parameters()))
        expected = [p.detach() - 0.1 * g for p, g in zip(model.parameters(), gradients)]
        optimizer = RobustSGD(model.parameters(), lr=0.1, rho=0)
        train(model, loaders, optimizer)
        for parameter, target in zip(model.parameters(), expected):
            torch.testing.assert_close(parameter, target)

    def test_network_training_with_optimizer_interface(self):
        for rho in (0, 0.5):
            with self.subTest(rho=rho):
                loaders = make_loaders(sample_semicircles, train_size=100, validation_size=40, test_size=40,
                                       batch_size=16, updates=20)
                model = SemicircleNet(seed=0)
                optimizer = RobustSGD(model.parameters(), lr=0.1, rho=rho)
                history, metrics = train(model, loaders, optimizer, eval_every=10)
                self.assertLess(history[-1]["train_loss"], history[0]["train_loss"])
                self.assertEqual(metrics["gradient_evaluations"], 20 * 16)

    def test_shared_training_with_logistic_regression(self):
        loaders = make_loaders(sample_logreg, train_size=100, validation_size=40,
                               test_size=40, batch_size=16, updates=20)
        model = nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.zero_()
            model.bias.zero_()
        optimizer = RobustSGD(model.parameters(), lr=0.1, rho=0)
        history, metrics = train(model, loaders, optimizer, eval_every=10)
        self.assertLess(history[-1]["train_loss"], history[0]["train_loss"])
        self.assertEqual(metrics["gradient_evaluations"], 20 * 16)

    def test_evaluation_with_known_predictions_and_uneven_batches(self):
        points = torch.tensor([[2.0, 0.0], [-2.0, 0.0], [0.0, 0.0], [-1.0, 0.0]])
        labels = torch.tensor([1.0, 0.0, 0.0, 0.0])
        model = nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, 0.0]]))
            model.bias.zero_()
        loader = DataLoader(TensorDataset(points, labels), batch_size=3)
        metrics = evaluate(model, loader)
        expected_loss = (2 * math.log1p(math.exp(-2)) + math.log(2)
                         + math.log1p(math.exp(-1))) / 4
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["loss"], expected_loss, places=6)


if __name__ == "__main__":
    unittest.main()
