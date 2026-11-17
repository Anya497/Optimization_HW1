import csv
import json
import math

import pytest
import torch
from torch import nn

from experiments import grid_search
from experiments.run_grid_search import main
from experiments.grid_search import (
    aggregate_metrics,
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


def test_aggregate_metrics_reports_sample_mean_and_std():
    rows = [
        {"method": "sgd", "validation_loss": 1.0, "test_accuracy": 0.5},
        {"method": "sgd", "validation_loss": 3.0, "test_accuracy": 0.7},
    ]
    result = aggregate_metrics(rows, ("method",))[0]
    assert result["method"] == "sgd" and result["n_runs"] == 2
    assert math.isclose(result["validation_loss_mean"], 2.0)
    assert math.isclose(result["validation_loss_std"], math.sqrt(2))
    assert math.isclose(result["test_accuracy_mean"], 0.6)
    assert math.isclose(result["test_accuracy_std"], math.sqrt(0.02))


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
    sgd_calls = [settings for name, settings in calls if name.startswith("tuning/sgd")]
    rsgd_calls = [settings for name, settings in calls if name.startswith("tuning/rsgd")]
    assert len(sgd_calls) == 4 and len(rsgd_calls) == 8
    assert {settings["rho"] for settings in sgd_calls} == {0.0}
    assert 0.0 in {settings["rho"] for settings in rsgd_calls}
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
    with (root / "tuning_aggregated.csv").open() as stream:
        tuning_groups = list(csv.DictReader(stream))
    with (root / "noise_sweep_aggregated.csv").open() as stream:
        noise_groups = list(csv.DictReader(stream))
    assert len(tuning_groups) == 6 and len(noise_groups) == 4
    assert all(group["n_runs"] == "2" for group in noise_groups)
    assert all("test_accuracy_mean" in group for group in noise_groups)
    assert json.loads((root / "selection.json").read_text()) == summary


def test_dataset_sensitivity_holds_training_seed_fixed(tmp_path):
    calls = []

    def fake_runner(optimizer_cls, name="optimizer", **settings):
        calls.append((name, settings))
        return {
            "validation_loss": settings["learning_rate"] + settings["rho"],
            "test_accuracy": 0.5,
        }

    summary = run_grid_search(
        fake_runner, lambda seed: nn.Linear(2, 1), sample_logreg,
        experiment="sensitivity", output_dir=tmp_path,
        lr_grid=(0.1,), rho_grid=(0.0, 0.5), seeds=(1, 2),
        noise_multipliers=(0.0, 1.0), warmup_updates=2,
        data_seeds=(1, 2),
        run_settings={"train_size": 8, "validation_size": 4, "test_size": 4,
                      "batch_size": 4, "updates": 3},
    )
    root = tmp_path / "sensitivity" / "grid_search"
    assert summary["data_seed"] == 1
    assert summary["dataset_sensitivity"]["enabled"] is True
    assert summary["dataset_sensitivity"]["training_seed"] == 1
    assert summary["dataset_sensitivity"]["data_seeds"] == [1, 2]
    sensitivity_calls = [
        settings for name, settings in calls
        if name.startswith("dataset_sensitivity/")
    ]
    assert len(sensitivity_calls) == 8
    assert {settings["seed"] for settings in sensitivity_calls} == {1}
    assert {settings["data_seed"] for settings in sensitivity_calls} == {1, 2}
    tuning_calls = [
        settings for name, settings in calls if name.startswith("tuning/")
    ]
    assert {settings["data_seed"] for settings in tuning_calls} == {1}
    with (root / "dataset_sensitivity.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 8
    with (root / "dataset_sensitivity_aggregated.csv").open() as stream:
        groups = list(csv.DictReader(stream))
    assert len(groups) == 4
    assert all(group["n_runs"] == "2" for group in groups)


def test_failed_tuning_seed_is_recorded_and_point_is_excluded(tmp_path):
    def flaky_runner(optimizer_cls, name="optimizer", **settings):
        if settings["rho"] == 0.5 and settings["seed"] == 1:
            raise RuntimeError("boom")
        return {
            "validation_loss": settings["learning_rate"] + settings["rho"],
            "test_accuracy": 0.5,
        }

    summary = run_grid_search(
        flaky_runner, lambda seed: nn.Linear(2, 1), sample_logreg,
        experiment="flaky", output_dir=tmp_path,
        lr_grid=(0.1,), rho_grid=(0.0, 0.5), seeds=(1, 2),
        noise_multipliers=(0.0, 1.0), warmup_updates=2,
        run_settings={"train_size": 8, "validation_size": 4, "test_size": 4,
                      "batch_size": 4, "updates": 3},
    )
    root = tmp_path / "flaky" / "grid_search"
    assert summary["rsgd"]["rho"] == 0.0
    assert summary["variability"]["failed_runs"] == 1
    with (root / "failed_runs.csv").open() as stream:
        failures = list(csv.DictReader(stream))
    assert len(failures) == 1
    assert failures[0]["phase"] == "tuning"
    assert failures[0]["method"] == "rsgd"
    assert failures[0]["seed"] == "1"
    assert failures[0]["error_type"] == "RuntimeError"
    with (root / "rsgd_lr_rho_grid.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 2


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


@pytest.mark.parametrize("experiment", ["logreg", "semicircle"])
def test_cli_runs_both_experiments(experiment, tmp_path, capsys):
    summary = main([
        "--experiment", experiment,
        "--output-dir", str(tmp_path),
        "--seeds", "1",
        "--lr-grid", "0.1",
        "--rho-grid", "0",
        "--noise-multipliers", "0,1",
        "--warmup-updates", "2",
        "--train-size", "8",
        "--validation-size", "4",
        "--test-size", "4",
        "--batch-size", "4",
        "--updates", "3",
        "--eval-every", "2",
    ])
    assert summary["sgd"]["learning_rate"] == 0.1
    assert summary["rsgd"]["rho"] == 0.0
    assert summary["grids"]["noise_multipliers"] == [0.0, 1.0]
    assert (tmp_path / experiment / "grid_search" / "selection.json").is_file()
    assert experiment in capsys.readouterr().out


def test_cli_accepts_dataset_sensitivity_flags(tmp_path):
    summary = main([
        "--experiment", "logreg",
        "--output-dir", str(tmp_path),
        "--seeds", "1",
        "--lr-grid", "0.1",
        "--rho-grid", "0",
        "--noise-multipliers", "0,1",
        "--warmup-updates", "2",
        "--train-size", "8",
        "--validation-size", "4",
        "--test-size", "4",
        "--batch-size", "4",
        "--updates", "3",
        "--eval-every", "2",
        "--data-seeds", "1,2",
        "--data-sensitivity-seed", "2",
    ])
    assert summary["dataset_sensitivity"]["training_seed"] == 2
    assert summary["dataset_sensitivity"]["data_seeds"] == [1, 2]
    root = tmp_path / "logreg" / "grid_search"
    assert (root / "dataset_sensitivity.csv").is_file()
    assert (root / "dataset_sensitivity_aggregated.csv").is_file()


def test_cli_rejects_both_data_seed_flags():
    with pytest.raises(SystemExit) as error:
        main(["--experiment", "logreg", "--data-seed", "1", "--data-seeds", "1,2"])
    assert error.value.code == 2
