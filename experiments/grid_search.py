"""Hyperparameter grid search for SGD and Robust SGD.

Both simulations share the driver: the caller supplies the simulation runner,
model factory, and data sampler. Selection uses the mean validation loss across
seeds; the test split is never used for selection.
"""

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch import nn

from src.data.common import make_loaders
from src.grad_noise import per_example_gradients
from src.optimizer import RobustSGD

from .common import setup_seed

LEARNING_RATES = (1e-3, 1e-2, 1e-1, 1.0)
RHOS = (0.0, 0.1, 0.3, 1.0, 3.0)
SEEDS = (1, 2, 3, 4, 5)
NOISE_MULTIPLIERS = (0.0, 0.1, 0.3, 1.0, 2.5)
WARMUP_UPDATES = 50
METRIC_KEYS = (
    "train_loss", "validation_loss", "validation_accuracy", "test_loss",
    "test_accuracy", "runtime_seconds", "update_seconds",
    "gradient_evaluations", "zero_steps", "zero_step_frequency",
)
FAILURE_FIELDS = (
    "phase", "method", "learning_rate", "rho", "seed", "gradient_noise",
    "data_seed", "error_type", "message",
)
DEFAULT_RUN_SETTINGS = {
    "train_size": 1000,
    "validation_size": 400,
    "test_size": 400,
    "batch_size": 32,
    "updates": 1000,
    "decay_scale": 50.0,
    "eval_every": 50,
}


def record_failure(
    failures: list[dict], *, phase: str, method: str, learning_rate: float,
    rho: float, seed: int, gradient_noise: float | None,
    data_seed: int | None, error: Exception,
) -> None:
    """Append one failed run's context to the shared failure table."""
    failures.append({
        "phase": phase, "method": method, "learning_rate": learning_rate,
        "rho": rho, "seed": seed, "gradient_noise": gradient_noise,
        "data_seed": data_seed, "error_type": type(error).__name__,
        "message": str(error),
    })


def sweep_grid(
    run_one: Callable[[float, float, int], dict],
    lr_grid: Sequence[float],
    rho_grid: Sequence[float],
    seeds: Sequence[int],
    *,
    method: str = "tuning",
    failures: list[dict] | None = None,
    data_seed: int | None = None,
) -> dict[tuple[float, float], list[dict]]:
    """Run every (learning rate, rho) point for every seed.

    A point is kept only if all its seeds succeed, so selection always compares
    the same number of runs per point; failed seeds go to ``failures``.
    """
    results = {}
    for learning_rate in lr_grid:
        for rho in rho_grid:
            rows = []
            complete = True
            for seed in seeds:
                setup_seed(seed)
                try:
                    metrics = run_one(learning_rate, rho, seed)
                except Exception as error:
                    complete = False
                    if failures is not None:
                        record_failure(
                            failures, phase="tuning", method=method,
                            learning_rate=learning_rate, rho=rho, seed=seed,
                            gradient_noise=None, data_seed=data_seed, error=error,
                        )
                    continue
                rows.append({
                    "learning_rate": learning_rate,
                    "rho": rho,
                    "seed": seed,
                    **metrics,
                })
            if complete:
                results[(learning_rate, rho)] = rows
    return results


def mean_validation_loss(rows: Sequence[dict]) -> float:
    """Mean final validation loss of one grid point across its seeds."""
    return sum(row["validation_loss"] for row in rows) / len(rows)


def aggregate_metrics(
    rows: Sequence[dict], group_keys: Sequence[str],
) -> list[dict]:
    """Compute sample mean/std of numeric metrics over runs in each group."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[column] for column in group_keys)
        groups.setdefault(key, []).append(row)
    aggregated = []
    for key, group in groups.items():
        result = dict(zip(group_keys, key))
        result["n_runs"] = len(group)
        for metric in METRIC_KEYS:
            values = [float(row[metric]) for row in group if metric in row]
            if not values:
                continue
            result[f"{metric}_mean"] = sum(values) / len(values)
            result[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        aggregated.append(result)
    return aggregated


def select_best_parameters(
    results: dict[tuple[float, float], list[dict]],
) -> tuple[float, float]:
    """Return the point with the smallest mean validation loss."""
    if not results:
        raise ValueError("cannot select from an empty grid")
    return min(
        results,
        key=lambda point: (mean_validation_loss(results[point]), point[0], point[1]),
    )


def estimate_gradient_scale(
    model: nn.Module,
    loaders: dict,
    optimizer: torch.optim.Optimizer,
    warmup_updates: int = WARMUP_UPDATES,
) -> float:
    """Estimate the per-coordinate gradient-noise scale from warmup updates.

    For each update, average the L2 norm of the flattened per-example gradient
    vectors, then divide by the square root of the parameter count. The supplied
    loaders must contain at least ``warmup_updates`` batches.
    """
    parameters = list(model.parameters())
    num_parameters = sum(parameter.numel() for parameter in parameters)
    if num_parameters == 0:
        raise ValueError("model has no trainable parameters")
    norm_sums = []
    for update, (x, y) in enumerate(loaders["train"]):
        if update >= warmup_updates:
            break
        model.train()
        optimizer.zero_grad()
        observations = per_example_gradients(model, x, y)
        with torch.no_grad():
            flat = torch.cat(
                [observations[p].reshape(len(x), -1) for p in parameters], dim=1
            )
            norm_sums.append(flat.norm(dim=1).mean().item())
        optimizer.step(noisy_grads=observations)
    if not norm_sums:
        raise ValueError("no warmup updates available")
    return (sum(norm_sums) / len(norm_sums)) / math.sqrt(num_parameters)


def write_csv(
    path: Path, rows: Sequence[dict], fieldnames: Sequence[str] | None = None,
) -> None:
    """Write result rows with stable column order, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = {key for row in rows for key in row}
        leading = [key for key in ("method", "learning_rate", "rho", "seed",
                                   "gradient_noise") if key in keys]
        fieldnames = leading + sorted(keys - set(leading))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def run_grid_search(
    runner: Callable[..., dict],
    model_factory: Callable[..., nn.Module],
    sampler: Callable,
    *,
    experiment: str = "experiment",
    output_dir: str | Path = "results",
    optimizer_cls: type[torch.optim.Optimizer] = RobustSGD,
    lr_grid: Sequence[float] = LEARNING_RATES,
    rho_grid: Sequence[float] = RHOS,
    seeds: Sequence[int] = SEEDS,
    noise_multipliers: Sequence[float] = NOISE_MULTIPLIERS,
    warmup_updates: int = WARMUP_UPDATES,
    data_seed: int | None = None,
    run_settings: dict | None = None,
) -> dict:
    """Sweep hyperparameters, select on validation loss, and sweep noise.

    ``runner`` is either simulation wrapper, ``model_factory`` creates the
    simulation model from a seed, and ``sampler`` generates its data. With
    ``data_seed=None`` each seed gets its own dataset; otherwise all runs at a
    given seed share the override, keeping SGD/RSGD paired.
    """
    if not seeds or not lr_grid or not rho_grid or warmup_updates < 1:
        raise ValueError("seeds, grids, and warmup_updates must be non-empty")

    settings = dict(DEFAULT_RUN_SETTINGS)
    if run_settings:
        settings.update(run_settings)
    root = Path(output_dir) / experiment / "grid_search"
    root.mkdir(parents=True, exist_ok=True)
    failures: list[dict] = []

    def seed_data(seed: int) -> int:
        return seed if data_seed is None else data_seed

    def tune(learning_rate: float, rho: float, seed: int, name: str) -> dict:
        return runner(
            optimizer_cls, name=f"tuning/{name}", rho=rho, seed=seed,
            learning_rate=learning_rate, data_seed=seed_data(seed),
            output_dir=str(root), save=False, **settings,
        )

    sgd_grid = sweep_grid(
        lambda lr, rho, seed: tune(lr, rho, seed, f"sgd_lr_{lr}"),
        lr_grid, (0.0,), seeds,
        method="sgd", failures=failures, data_seed=data_seed,
    )
    rsgd_grid = sweep_grid(
        lambda lr, rho, seed: tune(lr, rho, seed, f"rsgd_lr_{lr}_rho_{rho}"),
        lr_grid, rho_grid, seeds,
        method="rsgd", failures=failures, data_seed=data_seed,
    )
    sgd_lr = select_best_parameters(sgd_grid)[0]
    rsgd_lr, rsgd_rho = select_best_parameters(rsgd_grid)

    def gradient_scale(learning_rate: float, rho: float) -> dict:
        per_seed = []
        for seed in seeds:
            setup_seed(seed)
            model = model_factory(seed=seed)
            optimizer = optimizer_cls(model.parameters(), lr=learning_rate, rho=rho)
            loaders = make_loaders(
                sampler,
                train_size=settings["train_size"],
                validation_size=settings["validation_size"],
                test_size=settings["test_size"],
                batch_size=settings["batch_size"],
                updates=warmup_updates,
                data_seed=seed_data(seed),
                seed=seed,
            )
            per_seed.append(
                estimate_gradient_scale(model, loaders, optimizer, warmup_updates)
            )
        return {"scale": sum(per_seed) / len(per_seed), "per_seed": per_seed}

    scales = {
        "sgd": gradient_scale(sgd_lr, 0.0),
        "rsgd": gradient_scale(rsgd_lr, rsgd_rho),
    }
    noise_grids = {
        method: [scales[method]["scale"] * multiplier for multiplier in noise_multipliers]
        for method in scales
    }

    def sweep_noise(method: str, lr: float, rho: float) -> list[dict]:
        rows = []
        for seed in seeds:
            for noise in noise_grids[method]:
                setup_seed(seed)
                try:
                    metrics = runner(
                        optimizer_cls,
                        name=f"noise_sweep/{method}_lr_{lr}_rho_{rho}",
                        rho=rho, seed=seed, learning_rate=lr, gradient_noise=noise,
                        data_seed=seed_data(seed), output_dir=str(root), **settings,
                    )
                except Exception as error:
                    record_failure(
                        failures, phase="noise", method=method, learning_rate=lr,
                        rho=rho, seed=seed, gradient_noise=noise,
                        data_seed=seed_data(seed), error=error,
                    )
                    continue
                rows.append({
                    "method": method, "learning_rate": lr, "rho": rho,
                    "seed": seed, "gradient_noise": noise, **metrics,
                })
        return rows

    noise_rows = (
        sweep_noise("sgd", sgd_lr, 0.0)
        + sweep_noise("rsgd", rsgd_lr, rsgd_rho)
    )
    sgd_rows = [
        {"method": "sgd", **row} for rows in sgd_grid.values() for row in rows
    ]
    rsgd_rows = [
        {"method": "rsgd", **row} for rows in rsgd_grid.values() for row in rows
    ]
    write_csv(root / "sgd_lr_grid.csv", sgd_rows)
    write_csv(root / "rsgd_lr_rho_grid.csv", rsgd_rows)
    write_csv(root / "tuning_aggregated.csv", aggregate_metrics(
        sgd_rows + rsgd_rows, ("method", "learning_rate", "rho"),
    ))
    write_csv(root / "noise_sweep_summary.csv", noise_rows)
    write_csv(root / "noise_sweep_aggregated.csv", aggregate_metrics(
        noise_rows, ("method", "learning_rate", "rho", "gradient_noise"),
    ))
    write_csv(root / "failed_runs.csv", failures, fieldnames=FAILURE_FIELDS)
    summary = {
        "experiment": experiment,
        "selection_rule": "minimum mean validation_loss across seeds",
        "seeds": list(seeds),
        "grids": {
            "learning_rate": list(lr_grid),
            "rho": list(rho_grid),
            "noise_multipliers": list(noise_multipliers),
        },
        "run_settings": settings,
        "data_seed": data_seed,
        "sgd": {
            "learning_rate": sgd_lr,
            "rho": 0.0,
            "mean_validation_loss": mean_validation_loss(sgd_grid[(sgd_lr, 0.0)]),
        },
        "rsgd": {
            "learning_rate": rsgd_lr,
            "rho": rsgd_rho,
            "mean_validation_loss": mean_validation_loss(rsgd_grid[(rsgd_lr, rsgd_rho)]),
        },
        "gradient_scale": scales,
        "noise_grid": noise_grids,
        "variability": {
            "std_definition": "sample std across successful evaluation seeds (n-1)",
            "tuning_file": "tuning_aggregated.csv",
            "noise_sweep_file": "noise_sweep_aggregated.csv",
            "failed_runs_file": "failed_runs.csv",
            "failed_runs": len(failures),
        },
    }
    (root / "selection.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
