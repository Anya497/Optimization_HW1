import math
import unittest

import torch

from src.data.common import make_dataset, make_loaders
from src.data.semicircles import points_from_angles, sample_points as sample_semicircles
from src.data.log_regression import sample_points as sample_logreg


class DataTests(unittest.TestCase):
    def test_semicircle_formula_for_both_classes(self):
        angles = torch.tensor([0.0, math.pi / 2, 0.0, math.pi / 2])
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        points = points_from_angles(angles, labels, torch.Generator(), noise_std=0)
        expected = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.5], [1.0, -0.5]])
        torch.testing.assert_close(points, expected, atol=1e-6, rtol=0)

    def test_balanced_splits_and_reproducible_replacement_sampling(self):
        for sampler in (sample_semicircles, sample_logreg):
            with self.subTest(sampler=sampler.__module__):
                first = make_loaders(sampler, train_size=20, validation_size=10, test_size=10,
                                     batch_size=8, updates=10)
                second = make_loaders(sampler, train_size=20, validation_size=10, test_size=10,
                                      batch_size=8, updates=10)
                for name in ("train_eval", "validation", "test"):
                    labels = first[name].dataset.tensors[1]
                    self.assertEqual(labels.sum().item(), len(labels) // 2)
                self.assertTrue(first["train"].sampler.replacement)
                self.assertEqual(len(first["train"]), 10)
                for (x1, y1), (x2, y2) in zip(first["train"], second["train"]):
                    torch.testing.assert_close(x1, x2)
                    torch.testing.assert_close(y1, y2)

    def test_dataset_shapes_and_reproducibility_for_both_samplers(self):
        for sampler in (sample_semicircles, sample_logreg):
            with self.subTest(sampler=sampler.__module__):
                first = make_dataset(20, sampler, seed=7)
                second = make_dataset(20, sampler, seed=7)
                points, labels = first.tensors
                self.assertEqual(points.shape, (20, 2))
                self.assertEqual(labels.shape, (20,))
                self.assertEqual(labels.sum().item(), 10)
                torch.testing.assert_close(points, second.tensors[0])
                torch.testing.assert_close(labels, second.tensors[1])

    def test_training_validation_and_test_data_are_separate(self):
        for sampler in (sample_semicircles, sample_logreg):
            with self.subTest(sampler=sampler.__module__):
                loaders = make_loaders(sampler, train_size=20, validation_size=20,
                                       test_size=20, updates=1)
                train = loaders["train_eval"].dataset.tensors[0]
                validation = loaders["validation"].dataset.tensors[0]
                test = loaders["test"].dataset.tensors[0]
                self.assertFalse(torch.equal(train, validation))
                self.assertFalse(torch.equal(train, test))
                self.assertFalse(torch.equal(validation, test))


if __name__ == "__main__":
    unittest.main()
