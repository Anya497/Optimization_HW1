import csv
import json
import platform
from pathlib import Path

import torch
import random


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

    points, labels = loaders["test"].dataset.tensors
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
    axes[1].set(xlabel="a1", ylabel="a2", title=f"Test data seed {settings['data_seed'] + 2}")
    figure.suptitle(f"{name}: seed={settings['seed']}, sigma={settings['gradient_noise']}")
    figure.tight_layout()
    figure.savefig(output / "plots.png", dpi=150)
    plt.close(figure)
    return output
