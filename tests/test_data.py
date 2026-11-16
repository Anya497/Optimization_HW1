import math
import unittest

import torch

from src.data.semicircles import make_loaders, points_from_angles


class DataTests(unittest.TestCase):
    def test_semicircle_formula_for_both_classes(self):
        angles = torch.tensor([0.0, math.pi / 2, 0.0, math.pi / 2])
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        points = points_from_angles(angles, labels, torch.Generator(), noise_std=0)
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.5], [1.0, -0.5]])
        torch.testing.assert_close(points, expected, atol=1e-6, rtol=0)

    def test_balanced_splits_and_reproducible_replacement_sampling(self):
        first = make_loaders(train_size=20, validation_size=10, test_size=10,
                             batch_size=8, updates=10)
        second = make_loaders(train_size=20, validation_size=10, test_size=10,
                              batch_size=8, updates=10)
        for name in ("train_eval", "validation", "test"):
            labels = first[name].dataset.tensors[1]
            self.assertEqual(labels.sum().item(), len(labels) // 2)
        self.assertTrue(first["train"].sampler.replacement)
        self.assertEqual(len(first["train"]), 10)
        for (x1, y1), (x2, y2) in zip(first["train"], second["train"]):
            torch.testing.assert_close(x1, x2)
            torch.testing.assert_close(y1, y2)


if __name__ == "__main__":
    unittest.main()
