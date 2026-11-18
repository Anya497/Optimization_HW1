"""Visualize SGD/R-SGD on a logistic loss surface with bias fixed at zero."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import torch
from torch.nn import functional as F

from src.data.common import make_loaders
from src.data.log_regression import sample_points
from src.models import LogReg
from src.optimizer import RobustSGD
from src.train import train


class RecordingSGD(RobustSGD):
    """Record the existing optimizer's updates without changing its algorithm."""

    def __init__(self, params, **settings):
        super().__init__(params, **settings)
        self.trajectory = [self.snapshot()]
        self.zero_steps = []

    def snapshot(self):
        return torch.cat([p.detach().flatten().clone()
                          for group in self.param_groups for p in group["params"]])

    def step(self, closure=None, noisy_grads=None):
        loss = super().step(closure=closure, noisy_grads=noisy_grads)
        self.trajectory.append(self.snapshot())
        self.zero_steps.append(self.zero_step)
        return loss


@torch.no_grad()
def training_losses(parameters, points, labels):
    """Mean uncorrupted BCE for each (w1, w2, bias) parameter vector."""
    logits = points @ parameters[:, :2].T + parameters[:, 2]
    return F.binary_cross_entropy_with_logits(
        logits, labels[:, None].expand_as(logits), reduction="none"
    ).mean(dim=0)


def run_descent(
    output_dir="results/logreg/descent", *, seed=0, data_seed=42,
    train_size=500, batch_size=32, updates=200, learning_rate=0.3,
    rho=1.0, gradient_noise=0.5, decay_scale=50.0,
):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runs = {}
    for name, radius in (("SGD", 0.0), ("R-SGD", rho)):
        model = LogReg(seed=seed)
        model.lin.bias = None  # No bias term: only w1 and w2 are trained.
        optimizer = RecordingSGD(model.parameters(), lr=learning_rate, rho=radius)
        loaders = make_loaders(sample_points, train_size=train_size,
                               validation_size=100, test_size=100,
                               batch_size=batch_size, updates=updates,
                               data_seed=data_seed, seed=seed)
        _, metrics = train(model, loaders, optimizer, decay_scale=decay_scale,
                           gradient_noise=gradient_noise, seed=seed,
                           eval_every=max(1, updates // 10))
        points, labels = loaders["train"].dataset.tensors
        path = F.pad(torch.stack(optimizer.trajectory), (0, 1))  # Save the fixed zero bias too.
        losses = training_losses(path, points, labels)
        runs[name] = {"path": path, "losses": losses, "metrics": metrics,
                      "zero_steps": [False, *optimizer.zero_steps]}
        with (output / f"{'sgd' if name == 'SGD' else 'rsgd'}_trajectory.csv").open("w", newline="") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerow(["update", "w1", "w2", "bias", "train_loss", "zero_step"])
            for update, (parameters, loss) in enumerate(zip(path, losses)):
                writer.writerow([update, *parameters.tolist(), loss.item(),
                                 False if update == 0 else optimizer.zero_steps[update - 1]])

    paths = torch.cat([run["path"] for run in runs.values()])
    lower, upper = paths[:, :2].min(dim=0).values, paths[:, :2].max(dim=0).values
    padding = 0.15 * (upper - lower).clamp_min(0.5)
    xx, yy = torch.meshgrid(torch.linspace(lower[0] - padding[0], upper[0] + padding[0], 100),
                           torch.linspace(lower[1] - padding[1], upper[1] + padding[1], 100), indexing="xy")
    initial = runs["SGD"]["path"][0]
    slice_parameters = torch.stack((xx.flatten(), yy.flatten(), initial[2].expand(xx.numel())), dim=1)
    surface = training_losses(slice_parameters, points, labels).reshape(xx.shape)
    colors = {"SGD": "#2563eb", "R-SGD": "#ea580c"}
    fps = 20

    for dimension in (2, 3):
        fig = plt.figure(figsize=(10, 4.5), layout="constrained")
        ax = fig.add_subplot(1, 2, 1, projection="3d" if dimension == 3 else None)
        loss_ax = fig.add_subplot(1, 2, 2)
        ax.set(xlabel="w1", ylabel="w2", xlim=(float(xx.min()), float(xx.max())), ylim=(float(yy.min()), float(yy.max())))
        if dimension == 2:
            contours = ax.contour(xx, yy, surface, levels=18, cmap="Greys", linewidths=0.7)
            ax.clabel(contours, inline=True, fontsize=7, fmt="%.2f")
        else:
            ax.plot_surface(xx.numpy(), yy.numpy(), surface.numpy(), cmap="viridis",
                            alpha=0.2, rcount=20, ccount=20, linewidth=0)
            ax.contour(xx, yy, surface, levels=14, zdir="z", offset=0,
                       cmap="viridis", linewidths=0.7)
            ax.set(zlabel="Loss", zlim=(0, float(surface.max()) * 1.05))
            ax.view_init(elev=25, azim=-55)
        ax.set_title(f"Loss {'contours' if dimension == 2 else 'surface'} | bias fixed at 0", fontsize=10)
        artists = []
        for name, run in runs.items():
            coordinates = [run["path"][:, 0].numpy(), run["path"][:, 1].numpy()]
            if dimension == 3:
                coordinates.append(run["losses"].numpy())
            style = {"color": colors[name], "linestyle": "--" if name == "R-SGD" else "-"}
            line, = ax.plot(*[[]] * dimension, label=name, linewidth=2, zorder=10, **style)
            head, = ax.plot(*[[]] * dimension, marker="o", markersize=7, color=colors[name], zorder=11)
            curve, = loss_ax.plot([], [], label=name, **style)
            dot, = loss_ax.plot([], [], marker="o", color=colors[name], markersize=5)
            ax.scatter(*[coordinate[0] for coordinate in coordinates], color="black", marker="*", s=80)
            artists.append((run, coordinates, line, head, curve, dot, colors[name]))
        all_losses = torch.cat([run["losses"] for run in runs.values()])
        loss_ax.set(xlabel="Updates", ylabel="Training BCE", yscale="log",
                    xlim=(0, updates), ylim=(float(all_losses.min()) * 0.9, float(all_losses.max()) * 1.1),
                    title="Training loss")
        if dimension == 3:
            loss_ax.yaxis.tick_right()
            loss_ax.yaxis.set_label_position("right")
        ax.legend(fontsize=8)
        loss_ax.legend(fontsize=8)
        title = fig.suptitle("")

        def draw(step):
            for run, coordinates, line, head, curve, dot, color in artists:
                history = [coordinate[:step + 1] for coordinate in coordinates]
                point = [coordinate[step:step + 1] for coordinate in coordinates]
                if dimension == 3:
                    line.set_data_3d(*history)
                    head.set_data_3d(*point)
                else:
                    line.set_data(*history)
                    head.set_data(*point)
                head.set_markerfacecolor("white" if run["zero_steps"][step] else color)
                curve.set_data(range(step + 1), run["losses"][:step + 1])
                dot.set_data([step], [run["losses"][step]])
            paused = " | R-SGD zero step" if runs["R-SGD"]["zero_steps"][step] else ""
            title.set_text(f"σ={gradient_noise:g}, ρ={rho:g} | Update {step}/{updates}{paused}")

        animation = FuncAnimation(fig, draw, frames=[*range(updates + 1), *([updates] * (fps // 2))],
                                  interval=1000 / fps, repeat=True)
        animation.save(output / f"descent_{dimension}d.gif", writer=PillowWriter(fps=fps), dpi=100)
        plt.close(fig)

    settings = {
        "seed": seed, "data_seed": data_seed, "train_size": train_size,
        "validation_size": 100, "test_size": 100, "batch_size": batch_size,
        "updates": updates, "learning_rate": learning_rate, "decay_scale": decay_scale,
        "rho": rho, "gradient_noise": gradient_noise, "parameter_order": ["w1", "w2", "bias"],
        "trainable_parameters": ["w1", "w2"], "fixed_bias": 0.0,
        "contour_bias": initial[2].item(), "animation_fps": fps, "torch_version": torch.__version__,
        "metrics": {name: run["metrics"] for name, run in runs.items()},
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    for name, run in runs.items():
        print(f"{name}: loss {run['losses'][0]:.4f} → {run['losses'][-1]:.4f}; "
              f"zero steps {run['metrics']['zero_steps']}/{updates}")
    print(f"Saved 2D/3D GIFs and parameter paths to {output}")
    return runs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/logreg/descent")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=0.5)
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--updates", type=int, default=200)
    args = parser.parse_args()
    run_descent(args.output_dir, seed=args.seed, gradient_noise=args.noise,
                rho=args.rho, updates=args.updates)
