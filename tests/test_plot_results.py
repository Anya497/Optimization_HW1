import csv
import json

import numpy as np
import pytest
from PIL import Image

from experiments.grid_search import run_grid_search
from experiments.plot_results import mean_std, plot_results, selected_runs
from experiments.run_logreg_exp import run_experiment
from src.data.log_regression import sample_points
from src.models import LogReg


def test_selected_runs_freezes_validation_choice_instead_of_best_test_score():
    selection = {"sgd": {"learning_rate": 0.1, "rho": 0.0},
                 "rsgd": {"learning_rate": 0.3, "rho": 0.5}}
    selected = {"method": "rsgd", "settings": {"learning_rate": 0.3, "rho": 0.5},
                "metrics": {"test_accuracy": 0.7}}
    other = {"method": "rsgd", "settings": {"learning_rate": 0.3, "rho": 1.0},
             "metrics": {"test_accuracy": 0.9}}
    assert selected_runs([selected, other], selection) == [selected]


def test_error_bands_use_sample_standard_deviation_across_seeds():
    mean, std = mean_std([[1., 2.], [3., 6.]])
    np.testing.assert_allclose(mean, [2., 4.])
    np.testing.assert_allclose(std, np.sqrt([2., 8.]))


@pytest.mark.parametrize("legacy", [False, True])
def test_plot_saved_grid_search_uses_existing_metric_summary(tmp_path, legacy):
    run_grid_search(
        run_experiment, LogReg, sample_points, experiment="logreg", output_dir=tmp_path,
        lr_grid=(0.1,), rho_grid=(0.0,), seeds=(1,), noise_multipliers=(0., 1.),
        warmup_updates=2, data_seed=42,
        run_settings={"train_size": 8, "validation_size": 4, "test_size": 4,
                      "batch_size": 4, "updates": 3, "eval_every": 2},
    )
    root = tmp_path / "logreg" / "grid_search"
    if legacy:
        for path in (root / "noise_sweep").rglob("losses.csv"):
            with path.open() as file:
                history = list(csv.DictReader(file))
            for row in history:
                row.pop("validation_accuracy")
            with path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
    output = plot_results(root, seed=1)
    figures = list(output.glob("*.png"))
    assert len(figures) == (5 if legacy else 6)
    assert (output / "validation_accuracy.png").is_file() == (not legacy)
    for path in figures:
        with Image.open(path) as image:
            image.verify()
    assert not (output / "metrics_summary.csv").exists()
    manifest = json.loads((output / "plot_manifest.json").read_text())
    assert manifest["metric_source"] == "noise_sweep_aggregated.csv"
    assert len(manifest["counts"]) == 4
    with (root / "noise_sweep_aggregated.csv").open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4
    assert {row["method"] for row in rows} == {"sgd", "rsgd"}
    for row in rows:
        assert row["n_runs"] == "1"
        assert float(row["gradient_evaluations_mean"]) == 12
        assert float(row["gradient_evaluations_std"]) == 0
