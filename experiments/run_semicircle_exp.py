"""Simulation 2 entry point using RobustSGD."""

from src.data.semicircles import sample_points
from src.models import SemicircleNet

from .common import run_experiment as _run_experiment


def run_experiment(optimizer_cls, name="optimizer", **settings) -> dict:
    """Run the semicircle simulation with the shared pipeline.

    Accepts the same keyword settings as :func:`experiments.common.run_experiment`
    and defaults the output directory to ``results/simulation2``.
    """
    settings.setdefault("output_dir", "results/simulation2")
    return _run_experiment(SemicircleNet, sample_points, optimizer_cls, name, **settings)
