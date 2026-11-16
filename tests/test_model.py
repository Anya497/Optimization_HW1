import unittest

import torch
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from src.models import SemicircleNet
from src.grad_noise import per_example_gradients


class ModelTests(unittest.TestCase):
    def test_architecture_and_reproducible_initialization(self):
        state = torch.random.get_rng_state().clone()
        model = SemicircleNet(seed=7)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 33)
        self.assertEqual(model(torch.zeros(5, 2)).shape, (5,))
        self.assertEqual(model(torch.zeros(2)).shape, ())
        self.assertEqual(model.hidden.weight.shape, (8, 2))
        self.assertEqual(model.output.weight.shape, (1, 8))
        first = parameters_to_vector(model.parameters())
        second = parameters_to_vector(SemicircleNet(seed=7).parameters())
        other = parameters_to_vector(SemicircleNet(seed=8).parameters())
        torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(first, other))

    def test_all_parameter_gradients_against_finite_differences(self):
        model = SemicircleNet(seed=3).double()
        x = torch.tensor([[0.2, -0.5], [1.1, 0.3]], dtype=torch.float64)
        y = torch.tensor([0.0, 1.0], dtype=torch.float64)
        gradients = per_example_gradients(model, x, y)
        actual = torch.cat([gradients[p].reshape(len(x), -1) for p in model.parameters()], dim=1)
        original = parameters_to_vector(model.parameters()).detach().clone()
        expected = torch.empty_like(actual)
        epsilon = 1e-6
        with torch.no_grad():
            for index in range(original.numel()):
                offset = torch.zeros_like(original)
                offset[index] = epsilon
                vector_to_parameters(original + offset, model.parameters())
                plus = F.binary_cross_entropy_with_logits(model(x), y, reduction="none")
                vector_to_parameters(original - offset, model.parameters())
                minus = F.binary_cross_entropy_with_logits(model(x), y, reduction="none")
                expected[:, index] = (plus - minus) / (2 * epsilon)
            vector_to_parameters(original, model.parameters())
        torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-5)

if __name__ == "__main__":
    unittest.main()
