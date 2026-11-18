import csv

import torch
from PIL import Image
from torch.nn import functional as F

from experiments.run_logreg_descent import run_descent, training_losses


def test_trajectory_losses_match_binary_cross_entropy():
    parameters = torch.tensor([[0.2, -0.4, 0.3], [1., 2., -0.1]])
    points = torch.tensor([[1., 2.], [-1., 3.], [0., 1.]])
    labels = torch.tensor([0., 1., 1.])
    actual = training_losses(parameters, points, labels)
    expected = torch.stack([
        F.binary_cross_entropy_with_logits(points @ p[:2] + p[2], labels)
        for p in parameters
    ])
    torch.testing.assert_close(actual, expected)


def test_paired_zero_radius_paths_and_saved_figures(tmp_path):
    runs = run_descent(tmp_path, train_size=20, batch_size=4, updates=4,
                       rho=0., gradient_noise=0.2)
    torch.testing.assert_close(runs['SGD']['path'], runs['R-SGD']['path'])
    for name, run in runs.items():
        assert run['path'].shape == (5, 3)
        assert torch.count_nonzero(run['path'][:, 2]) == 0
        assert abs(run['losses'][-1].item() - run['metrics']['train_loss']) < 1e-6
    for name in ('sgd', 'rsgd'):
        with (tmp_path / f'{name}_trajectory.csv').open() as file:
            rows = list(csv.DictReader(file))
        assert len(rows) == 5
        assert all(float(row['bias']) == 0 for row in rows)
        assert rows[0]['update'] == '0' and rows[-1]['update'] == '4'
    for dimension in (2, 3):
        with Image.open(tmp_path / f'descent_{dimension}d.gif') as image:
            assert image.is_animated and image.n_frames == 5
            assert image.info["loop"] == 0
            image.seek(image.n_frames - 1)
            image.load()
