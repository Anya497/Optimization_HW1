"""Check Equation (4) at fixed logistic-regression parameters; no training."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from torch import nn

from src.data.common import make_dataset
from src.data.log_regression import sample_points
from src.grad_noise import per_example_gradients


def main():
    batch_sizes = [8, 16, 32, 64, 128]
    repetitions = 200
    data_size = 1000
    data_seed = 42
    model_seed = 0
    sampling_seed = 1
    output_dir = Path(__file__).resolve().parent / "batch_size_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = make_dataset(data_size, sample_points, seed=data_seed)
    x, y = [tensor.double() for tensor in dataset.tensors]
    torch.manual_seed(model_seed)
    model = nn.Linear(2, 1, dtype=torch.float64)

    # The model stays fixed, so its gradients can be computed once and sampled.
    gradients = per_example_gradients(model, x, y)
    observations = torch.cat([
        values.reshape(data_size, -1) for values in gradients.values()
    ], dim=1)
    generator = torch.Generator().manual_seed(sampling_seed)
    parameter_names = ["w1", "w2", "bias"]
    population_variance = observations.var(dim=0, unbiased=False)
    summaries, rows, matrices = [], [], []

    for batch_size in batch_sizes:
        indices = torch.randint(data_size, (repetitions, batch_size), generator=generator)
        batches = observations[indices]
        means = batches.mean(dim=1, keepdim=True)
        # Equation (4): covariance of the batch mean, then the shared ridge.
        raw_diagonal = (batches - means).square().sum(dim=1) / (batch_size * (batch_size - 1))
        ridge = 1e-3 * raw_diagonal.mean(dim=1, keepdim=True).clamp_min(1e-8)
        diagonal = raw_diagonal + ridge
        mean_matrix = torch.diag(diagonal.mean(dim=0))
        matrices.append(mean_matrix.numpy())
        summaries.append({
            "batch_size": batch_size,
            "sampled_observations": repetitions * batch_size,
            "mean_S": mean_matrix.tolist(),
            "std_diagonal": diagonal.std(dim=0).tolist(),
            "standard_error_diagonal": (diagonal.std(dim=0) / repetitions ** 0.5).tolist(),
            "mean_raw_diagonal": raw_diagonal.mean(dim=0).tolist(),
            "expected_raw_diagonal": (population_variance / batch_size).tolist(),
        })
        for repeat in range(repetitions):
            rows.append([batch_size, repeat, *diagonal[repeat].tolist(), ridge[repeat].item()])

    settings = {
        "batch_sizes": batch_sizes, "repetitions": repetitions, "data_size": data_size,
        "data_seed": data_seed, "model_seed": model_seed, "sampling_seed": sampling_seed,
        "parameters": {name: value.detach().tolist() for name, value in model.named_parameters()},
        "parameter_order": parameter_names, "gradient_noise": 0.0,
        "sampling": "Independent draws with replacement from one fixed dataset",
        "training_updates": 0,
        "torch_version": torch.__version__,
        "population_gradient_variance": population_variance.tolist(),
    }
    (output_dir / "summary.json").write_text(json.dumps({"settings": settings, "results": summaries}, indent=2) + "\n")
    with (output_dir / "matrices.csv").open("w", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["batch_size", "repetition", "S_w1_w1", "S_w2_w2", "S_bias_bias", "ridge"])
        writer.writerows(rows)

    fig, axes = plt.subplots(2, len(batch_sizes), figsize=(13, 5.5), layout="constrained")
    fig.suptitle("Batch size and gradient uncertainty", fontsize=18)
    top_max = max(matrix.max() for matrix in matrices)
    scaled_max = max((batch_size * matrix).max() for batch_size, matrix in zip(batch_sizes, matrices))
    for column, (batch_size, matrix) in enumerate(zip(batch_sizes, matrices)):
        for row, (values, maximum) in enumerate(((matrix, top_max), (batch_size * matrix, scaled_max))):
            ax = axes[row, column]
            ax.imshow(values, cmap="magma", norm=Normalize(0, maximum))
            ax.set_xticks(range(3), parameter_names if row == 1 else [""] * 3)
            ax.set_yticks(range(3), parameter_names if column == 0 else [""] * 3)
            ax.tick_params(length=0, labelsize=10)
            if row == 0:
                ax.set_title(f"b = {batch_size}", fontsize=12, pad=10)
            for i in range(3):
                value = values[i, i]
                ax.text(i, i, f"{value:.3g}", ha="center", va="center",
                        color="black" if value > maximum * 0.65 else "white", fontsize=10)
            for spine in ax.spines.values():
                spine.set_visible(False)
        if column == 0:
            axes[0, column].set_ylabel("Mean S", fontsize=13, labelpad=12)
            axes[1, column].set_ylabel("b × mean S", fontsize=13, labelpad=12)
    fig.colorbar(axes[0, 0].images[0], ax=axes[0, :], shrink=0.75)
    fig.colorbar(axes[1, 0].images[0], ax=axes[1, :], shrink=0.75)
    fig.savefig(output_dir / "uncertainty_heatmaps.png", dpi=180)
    fig.savefig(output_dir / "uncertainty_heatmaps.pdf")
    plt.close(fig)

    print("batch size | mean diagonal S (w1, w2, bias) | batch size × mean diagonal S")
    for batch_size, matrix in zip(batch_sizes, matrices):
        print(f"{batch_size:10} | {np.diag(matrix)} | {batch_size * np.diag(matrix)}")
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
