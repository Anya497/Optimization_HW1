import csv
import json
import platform
from pathlib import Path
from typing import Callable

import torch
import random
from torch import nn

from src.data.common import make_loaders
from src.train import train


def run_experiment(
    model_factory: Callable[..., nn.Module],
    sampler: Callable,
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
    output_dir: str = "results",
    save: bool = True,
) -> dict:
    """Create fresh data/model for a paired run, train, and save its outputs.

    Set ``save=False`` for tuning points whose metrics are recorded by the
    caller; full artifacts are only written for the final runs.
    """
    model = model_factory(seed=seed)
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate, rho=rho)
    loaders = make_loaders(
        sampler, train_size, validation_size, test_size, batch_size, updates,
        data_seed, seed,
    )
    history, metrics = train(
        model, loaders, optimizer, decay_scale=decay_scale,
        gradient_noise=gradient_noise, seed=seed, eval_every=eval_every,
    )
    settings = {
        "train_size": train_size, "validation_size": validation_size,
        "test_size": test_size, "batch_size": batch_size, "updates": updates,
        "data_seed": data_seed, "seed": seed, "learning_rate": learning_rate,
        "decay_scale": decay_scale, "gradient_noise": gradient_noise,
        "eval_every": eval_every, "rho": rho,
    }
    if save:
        save_results(model, loaders, output_dir, name, history, metrics, settings)
    return metrics


def setup_seed(seed: int):
    torch.random.manual_seed(seed)
    random.seed(seed)


def save_results(model, loaders, output_dir, name, history, metrics, settings) -> Path:
    """Save one run's settings, metrics, losses, weights, and decision boundary."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    run_name = (f"seed_{settings['seed']}_sigma_{settings['gradient_noise']}"
                f"_lr_{settings['learning_rate']}")
    output = Path(output_dir) / name / run_name
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"settings": settings, "optimizer": name, "metrics": metrics,
                "python": platform.python_version(), "torch": torch.__version__}
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with (output / "losses.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    torch.save(model.state_dict(), output / "model.pt")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for key, label in (("train_loss", "Training"), ("validation_loss", "Validation")):
        axes[0].plot([row["update"] for row in history], [row[key] for row in history], label=label)
    axes[0].set(xlabel="Updates", ylabel="BCE loss")
    axes[0].legend()

    point_batches, label_batches = zip(*loaders["test"])
    points = torch.cat(point_batches)
    labels = torch.cat(label_batches).reshape(-1)
    lower, upper = points.min(dim=0).values - 0.3, points.max(dim=0).values + 0.3
    xx, yy = torch.meshgrid(torch.linspace(lower[0], upper[0], 150),
                           torch.linspace(lower[1], upper[1], 150), indexing="xy")
    model.eval()
    with torch.no_grad():
        logits = model(torch.stack((xx.flatten(), yy.flatten()), dim=1)).reshape(xx.shape)
    axes[1].contourf(xx.numpy(), yy.numpy(), logits.sigmoid().numpy(), levels=20,
                     vmin=0, vmax=1, cmap="coolwarm", alpha=0.4)
    axes[1].contour(xx.numpy(), yy.numpy(), logits.numpy(), levels=[0], colors="black")
    axes[1].scatter(points[:, 0], points[:, 1], c=labels, cmap="coolwarm", vmin=0, vmax=1, s=8)
    axes[1].set(xlabel="a1", ylabel="a2", title="Test set")
    figure.suptitle(f"{name}: seed={settings['seed']}, sigma={settings['gradient_noise']}")
    figure.tight_layout()
    figure.savefig(output / "plots.png", dpi=150)
    plt.close(figure)
    return output
