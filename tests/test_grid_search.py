import csv
import json
import math

import torch
from torch import nn

from experiments import grid_search
from experiments.grid_search import (
    estimate_gradient_scale,
    run_grid_search,
    select_best_parameters,
)
from experiments.run_logreg_exp import run_experiment as run_logreg
from src.data.log_regression import sample_points as sample_logreg
from src.data.common import make_loaders
from src.models import LogReg
from src.optimizer import RobustSGD


def test_selection_uses_mean_validation_loss_and_breaks_ties_on_grid_order():
    results = {
        (0.1, 0.0): [{"validation_loss": 0.8}, {"validation_loss": 0.6}],
        (0.2, 0.0): [{"validation_loss": 0.4}, {"validation_loss": 0.5}],
        (0.1, 0.5): [{"validation_loss": 0.3}, {"validation_loss": 0.4}],
    }
    assert select_best_parameters(results) == (0.1, 0.5)

    tied = {
        (0.2, 0.0): [{"validation_loss": 0.5}],
        (0.1, 0.0): [{"validation_loss": 0.5}],
    }
    assert select_best_parameters(tied) == (0.1, 0.0)


def test_gradient_scale_uses_mean_per_example_norm_over_parameter_count(monkeypatch):
    model = nn.Linear(2, 1)
    loaders = make_loaders(
        sample_logreg, train_size=8, validation_size=4, test_size=4,
        batch_size=4, updates=2, seed=1,
    )
    optimizer = RobustSGD(model.parameters(), lr=0.01, rho=0)
    monkeypatch.setattr(
        grid_search, "per_example_gradients",
        lambda model, x, y: {
            parameter: torch.ones((len(x), *parameter.shape))
            for parameter in model.parameters()
        },
    )
    scale = estimate_gradient_scale(model, loaders, optimizer, warmup_updates=2)
    assert math.isclose(scale, math.sqrt(3) / math.sqrt(3), abs_tol=1e-6)


def test_grid_search_records_every_point_and_selects_with_fake_runner(tmp_path):
    calls = []

    def fake_runner(optimizer_cls, name="optimizer", **settings):
        calls.append((name, settings))
        return {
            "validation_loss": settings["learning_rate"] + settings["rho"],
            "test_accuracy": 0.5,
        }

    summary = run_grid_search(
        fake_runner, lambda seed: nn.Linear(2, 1), sample_logreg,
        experiment="fake", output_dir=tmp_path,
        lr_grid=(0.1, 0.2), rho_grid=(0.0, 0.5), seeds=(1, 2),
        noise_multipliers=(0.0, 1.0), warmup_updates=2,
        run_settings={"train_size": 8, "validation_size": 4, "test_size": 4,
                      "batch_size": 4, "updates": 3},
    )
    assert summary["sgd"]["learning_rate"] == 0.1
    assert summary["rsgd"]["learning_rate"] == 0.1
    assert summary["rsgd"]["rho"] == 0.0
    assert len(calls) == 20
    assert sum(settings.get("save") is False for _, settings in calls) == 12
    root = tmp_path / "fake" / "grid_search"
    with (root / "sgd_lr_grid.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 4
    with (root / "rsgd_lr_rho_grid.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 8
    with (root / "noise_sweep_summary.csv").open() as stream:
        noise_rows = list(csv.DictReader(stream))
    assert len(noise_rows) == 8
    assert {row["method"] for row in noise_rows} == {"sgd", "rsgd"}
    assert json.loads((root / "selection.json").read_text()) == summary


def test_grid_search_saves_full_artifacts_for_noise_sweep(tmp_path):
    summary = run_grid_search(
        run_logreg, LogReg, sample_logreg,
        experiment="logreg", output_dir=tmp_path,
        lr_grid=(0.1,), rho_grid=(0.0,), seeds=(1,),
        noise_multipliers=(0.0, 1.0), warmup_updates=2,
        run_settings={"train_size": 8, "validation_size": 4, "test_size": 4,
                      "batch_size": 4, "updates": 3, "eval_every": 2},
    )
    root = tmp_path / "logreg" / "grid_search"
    assert summary["sgd"]["learning_rate"] == 0.1
    assert summary["rsgd"]["rho"] == 0.0
    assert summary["gradient_scale"]["sgd"]["scale"] > 0
    assert summary["noise_grid"]["sgd"][0] == 0.0
    assert len(list((root / "noise_sweep").rglob("metrics.json"))) == 4
    assert len(list((root / "noise_sweep").rglob("plots.png"))) == 4
