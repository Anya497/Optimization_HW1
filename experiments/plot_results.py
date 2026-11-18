"""Plot a saved grid search: python -m experiments.plot_results <grid_search_dir>."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle

from src.data.common import make_dataset
from .run_grid_search import EXPERIMENTS


METHODS = {"sgd": ("SGD", "#2563eb"), "rsgd": ("R-SGD", "#ea580c")}
METRICS = {
    "train_loss": "Training loss", "test_loss": "Test loss",
    "test_accuracy": "Test accuracy", "runtime_seconds": "Runtime (s)",
    "gradient_evaluations": "Gradient evaluations", "zero_step_frequency": "Zero-step frequency",
}


def read_csv(path):
    with Path(path).open() as file:
        return list(csv.DictReader(file))


def mean_std(values):
    values = np.asarray(values)
    return values.mean(axis=0), values.std(axis=0, ddof=1) if len(values) > 1 else np.zeros_like(values[0])


def read_runs(root):
    runs = []
    for path in sorted((root / "noise_sweep").rglob("metrics.json")):
        run = json.loads(path.read_text())
        run["method"] = run["optimizer"].split("/")[-1].split("_lr_")[0]
        run["path"] = path.parent
        run["history"] = read_csv(path.parent / "losses.csv")
        runs.append(run)
    return runs


def selected_runs(runs, selection):
    # Freeze validation-selected settings; never select a radius using test results.
    return [run for run in runs if all(
        run["settings"][key] == selection[run["method"]][key]
        for key in ("learning_rate", "rho")
    )]


def save_figure(figure, output, name):
    figure.savefig(output / f"{name}.png", dpi=160)
    plt.close(figure)


def plot_results(root, seed=None):
    root = Path(root)
    selection = json.loads((root / "selection.json").read_text())
    runs = read_runs(root)
    chosen = selected_runs(runs, selection)
    aggregated = [
        {key: value if key == "method" else int(value) if key == "n_runs" else float(value)
         for key, value in row.items()}
        for row in read_csv(root / "noise_sweep_aggregated.csv")
    ]
    noises = selection["noise_grid"]["sgd"]
    seed = selection["seeds"][0] if seed is None else seed
    output = root / "plots"
    output.mkdir(exist_ok=True)
    variability = ("training and dataset randomness" if selection["data_seed"] is None
                   else "training randomness on one fixed dataset")

    def group(method, noise):
        return [run for run in chosen if run["method"] == method
                and run["settings"]["gradient_noise"] == noise]

    curve_figures = []
    for filename, metrics in (("loss_curves", ("train_loss", "validation_loss")),
                              ("validation_accuracy", ("validation_accuracy",))):
        if filename == "validation_accuracy" and any(
            "validation_accuracy" not in point for run in chosen for point in run["history"]
        ):
            print("Validation accuracy was not recorded in these histories; rerun the selected runs to plot it.")
            continue
        fig, axes = plt.subplots(len(noises), len(metrics), squeeze=False,
                                 figsize=(5 * len(metrics), 2.8 * len(noises)), layout="constrained")
        for row, noise in enumerate(noises):
            for column, metric in enumerate(metrics):
                ax = axes[row, column]
                accuracy = metric == "validation_accuracy"
                for method, (label, color) in METHODS.items():
                    samples = group(method, noise)
                    if not samples:
                        continue
                    updates = [int(point["update"]) for point in samples[0]["history"]]
                    mean, std = mean_std([[float(point[metric]) for point in run["history"]]
                                          for run in samples])
                    if accuracy:
                        mean, std = 100 * mean, 100 * std
                    ax.plot(updates, mean, color=color, linestyle="--" if method == "rsgd" else "-",
                            label=f"{label} (n={len(samples)})")
                    ax.fill_between(updates, mean - std, mean + std, color=color, alpha=0.15)
                label = "Validation accuracy" if accuracy else "Training" if metric == "train_loss" else "Validation"
                ax.set(xlabel="Updates", ylabel="Accuracy (%)" if accuracy else "BCE loss",
                       title=f"{label} | σ={noise:.3g}")
                if accuracy:
                    ax.set_ylim(0, 100)
                ax.legend(fontsize=8)
        fig.suptitle("Frozen settings | mean ± 1 sample SD", fontsize=14)
        save_figure(fig, output, filename)
        curve_figures.append(filename)

    # Reuse metric summaries; only loss histories need new statistics.
    rows = [row for row in aggregated if all(
        row[key] == selection[row["method"]][key] for key in ("learning_rate", "rho")
    )]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), layout="constrained")
    for ax, (metric, label) in zip(axes.flat, METRICS.items()):
        for method, (name, color) in METHODS.items():
            samples = [row for row in rows if row["method"] == method]
            ax.errorbar([row["gradient_noise"] for row in samples],
                        [row[f"{metric}_mean"] for row in samples],
                        yerr=[row[f"{metric}_std"] for row in samples],
                        fmt="s--" if method == "rsgd" else "o-", capsize=3, color=color, label=name)
        ax.set(xlabel="Gradient noise σ", ylabel=label)
        if metric in ("test_accuracy", "zero_step_frequency"):
            ax.set_ylim(0, 1)
    axes[0, 0].legend()
    fig.suptitle("Frozen settings | mean ± 1 sample SD", fontsize=14)
    save_figure(fig, output, "noise_summary")

    # Section 4.4: all radii at the selected R-SGD learning rate, without retuning per σ.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    robust = [row for row in aggregated if row["method"] == "rsgd"
              and row["learning_rate"] == selection["rsgd"]["learning_rate"]]
    for noise, color in zip(noises, plt.cm.viridis(np.linspace(0.1, 0.9, len(noises)))):
        samples = sorted((row for row in robust if row["gradient_noise"] == noise),
                         key=lambda row: row["rho"])
        radii = [row["rho"] for row in samples]
        for ax, metric, label in zip(axes, ("train_loss", "validation_loss", "zero_step_frequency"),
                                     ("Training loss", "Validation loss", "Zero-step frequency")):
            ax.errorbar(radii, [row[f"{metric}_mean"] for row in samples],
                        yerr=[row[f"{metric}_std"] for row in samples], fmt="o-", capsize=3,
                        color=color, label=f"σ={noise:.3g}")
            ax.set(xlabel="Radius ρ", ylabel=label)
    axes[0].legend(fontsize=8)
    axes[-1].set_ylim(0, 1)
    fig.suptitle(f"Radius sensitivity | lr={selection['rsgd']['learning_rate']:g} | mean ± 1 SD")
    save_figure(fig, output, "radius_sensitivity")

    # Show validation-based tuning, including missing points and grid-edge selections.
    tuning = read_csv(root / "tuning_aggregated.csv")
    lrs = selection["grids"]["learning_rate"]
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.5), layout="constrained",
                             gridspec_kw={"height_ratios": [1, len(selection["grids"]["rho"])]})
    minimum = min(float(row["validation_loss_mean"]) for row in tuning)
    maximum = max(float(row["validation_loss_mean"]) for row in tuning)
    for ax, method in zip(axes, METHODS):
        radii = [0.0] if method == "sgd" else selection["grids"]["rho"]
        matrix = np.full((len(radii), len(lrs)), np.nan)
        for point in tuning:
            if point["method"] == method:
                matrix[radii.index(float(point["rho"])), lrs.index(float(point["learning_rate"]))] = float(point["validation_loss_mean"])
        image = ax.imshow(matrix, aspect="auto", cmap="viridis_r", vmin=minimum, vmax=maximum)
        ax.set_facecolor("#dddddd")
        ax.set_xticks(range(len(lrs)), [f"{lr:g}" for lr in lrs])
        ax.set_yticks(range(len(radii)), [f"{rho:g}" for rho in radii])
        ax.set(xlabel="Learning rate", ylabel="Radius ρ", title=METHODS[method][0])
        best = selection[method]
        ax.add_patch(Rectangle((lrs.index(best["learning_rate"]) - 0.5,
                               radii.index(best["rho"]) - 0.5), 1, 1,
                              fill=False, edgecolor="#ef4444", linewidth=2.5))
    fig.colorbar(image, ax=list(axes), label="Mean validation loss", shrink=0.8)
    fig.suptitle("Tuning | outlined cell = selected settings")
    save_figure(fig, output, "tuning")

    # Section 3: paired decision boundaries on exactly the saved run's test data.
    _, model_factory, sampler = EXPERIMENTS[selection["experiment"]]
    fig, axes = plt.subplots(len(noises), 2, squeeze=False,
                             figsize=(8, 3 * len(noises)), layout="constrained")
    for row, noise in enumerate(noises):
        for column, (method, (name, _)) in enumerate(METHODS.items()):
            ax = axes[row, column]
            matches = [run for run in group(method, noise) if run["settings"]["seed"] == seed]
            if not matches:
                ax.text(0.5, 0.5, "Run unavailable", transform=ax.transAxes, ha="center")
                continue
            run = matches[0]
            settings = run["settings"]
            points, labels = make_dataset(settings["test_size"], sampler, settings["data_seed"] + 2).tensors
            lower, upper = points.min(dim=0).values - 0.3, points.max(dim=0).values + 0.3
            xx, yy = torch.meshgrid(torch.linspace(lower[0], upper[0], 150),
                                   torch.linspace(lower[1], upper[1], 150), indexing="xy")
            model = model_factory(seed=seed)
            model.load_state_dict(torch.load(run["path"] / "model.pt", weights_only=True))
            model.eval()
            with torch.no_grad():
                logits = model(torch.stack((xx.flatten(), yy.flatten()), dim=1)).reshape(xx.shape)
            ax.contourf(xx, yy, logits.sigmoid(), levels=np.linspace(0, 1, 11), cmap="coolwarm", alpha=0.35)
            if logits.min() < 0 < logits.max():
                ax.contour(xx, yy, logits, levels=[0], colors="black", linewidths=1)
            ax.scatter(points[:, 0], points[:, 1], c=labels, cmap="coolwarm", vmin=0, vmax=1, s=9)
            ax.set(xlabel="a1", ylabel="a2", title=f"{name} | σ={noise:.3g}", aspect="equal")
    fig.suptitle(f"Decision boundaries | evaluation seed {seed} | R-SGD ρ={selection['rsgd']['rho']:g}")
    save_figure(fig, output, "decision_boundaries")

    manifest = {
        "evaluation_seed": seed, "variability": variability,
        "error_bars": "Mean ± one sample standard deviation across successful seeds; no spread estimated for n=1",
        "selected_settings": {method: selection[method] for method in METHODS},
        "metric_source": "noise_sweep_aggregated.csv",
        "counts": [{key: row[key] for key in ("method", "gradient_noise", "n_runs")} for row in rows],
        "failed_runs": selection["variability"]["failed_runs"],
        "figures": curve_figures + ["noise_summary", "radius_sensitivity", "tuning", "decision_boundaries"],
    }
    (output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(manifest['figures'])} figures to {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    plot_results(args.root, args.seed)
