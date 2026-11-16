import json
import tempfile
import unittest

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.common import save_results


class ResultsTests(unittest.TestCase):
    def test_save_results_with_both_dataset_formats(self):
        points = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
        labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
        datasets = {
            "tensor": TensorDataset(points, labels),
            "list": [(point, label.reshape(1)) for point, label in zip(points, labels)],
        }
        model = nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0, -1.0]]))
            model.bias.zero_()
        history = [{"update": 0, "train_loss": 1.0, "validation_loss": 1.1}]
        metrics = {"test_loss": 0.5, "test_accuracy": 0.75}
        settings = {"seed": 0, "data_seed": 42, "gradient_noise": 0.0, "learning_rate": 0.1}

        with tempfile.TemporaryDirectory() as directory:
            for name, dataset in datasets.items():
                with self.subTest(dataset=name):
                    output = save_results(model, {"test": DataLoader(dataset, batch_size=2)},
                                          directory, name, history, metrics, settings)
                    metadata = json.loads((output / "metrics.json").read_text())
                    self.assertEqual(metadata["metrics"], metrics)
                    self.assertIn("validation_loss", (output / "losses.csv").read_text())
                    weights = torch.load(output / "model.pt", weights_only=True)
                    torch.testing.assert_close(weights["weight"], model.weight)
                    with Image.open(output / "plots.png") as plot:
                        plot.verify()


if __name__ == "__main__":
    unittest.main()
