"""Simulation 2 entry point using the incoming RobustSGD optimizer interface."""

import torch

from src.data.semicircles import make_loaders
from src.models import SemicircleNet
from src.train import train

from .common import save_results


def run_experiment(
    optimizer_cls: type[torch.optim.Optimizer],
    name: str = "optimizer",
    *,
    train_size: int = 1000,
    validation_size: int = 400,
    test_size: int = 400,
    batch_size: int = 32,
    updates: int = 1000,
    data_seed: int = 42,
    seed: int = 0,
    learning_rate: float = 0.1,
    rho: float = 0.0,
    decay_scale: float = 50.0,
    gradient_noise: float = 0.0,
    eval_every: int = 50,
    output_dir: str = "results/simulation2",
) -> dict:
    """Create fresh data/model for a paired run, train, and save its outputs."""
    model = SemicircleNet(seed=seed)
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate, rho=rho)
    loaders = make_loaders(train_size, validation_size, test_size, batch_size,
                           updates, data_seed, seed)
    history, metrics = train(model, loaders, optimizer, decay_scale=decay_scale,
                             gradient_noise=gradient_noise, seed=seed, eval_every=eval_every)
    settings = {
        "train_size": train_size, "validation_size": validation_size,
        "test_size": test_size, "batch_size": batch_size, "updates": updates,
        "data_seed": data_seed, "seed": seed, "learning_rate": learning_rate,
        "decay_scale": decay_scale, "gradient_noise": gradient_noise,
        "eval_every": eval_every,
        "rho": rho,
    }
    save_results(model, loaders, output_dir, name, history, metrics, settings)
    return metrics
