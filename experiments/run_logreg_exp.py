"""Simulation 1 entry point using RobustSGD."""

from src.data.log_regression import sample_points
from src.models import LogReg

from .common import run_experiment as _run_experiment


def run_experiment(optimizer_cls, name="optimizer", **settings) -> dict:
    """Run the Gaussian/logistic-regression simulation with the shared pipeline.

    Accepts the same keyword settings as :func:`experiments.common.run_experiment`
    and defaults the output directory to ``results/logreg``.
    """
    settings.setdefault("output_dir", "results/logreg")
    return _run_experiment(LogReg, sample_points, optimizer_cls, name, **settings)
