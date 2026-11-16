import csv
import json
import math

from experiments.run_semicircle_exp import run_experiment
from src.optimizer import RobustSGD


def test_small_experiment_saves_its_results(tmp_path):
    metrics = run_experiment(
        RobustSGD, name="smoke", rho=0.5,
        train_size=20, validation_size=10, test_size=10,
        batch_size=4, updates=6, eval_every=4, output_dir=str(tmp_path),
    )
    assert all(math.isfinite(value) for value in metrics.values())
    assert metrics["gradient_evaluations"] == 24
    assert 0 <= metrics["test_accuracy"] <= 1

    metadata_path = next((tmp_path / "smoke").glob("*/metrics.json"))
    metadata = json.loads(metadata_path.read_text())
    assert metadata["metrics"] == metrics
    assert metadata["settings"]["rho"] == 0.5
    with (metadata_path.parent / "losses.csv").open() as stream:
        updates = [int(row["update"]) for row in csv.DictReader(stream)]
    assert updates == [0, 4, 6]
    assert (metadata_path.parent / "model.pt").is_file()
    assert (metadata_path.parent / "plots.png").is_file()
