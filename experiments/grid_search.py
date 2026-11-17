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

LEARNING_RATES = (1e-2, 1e-1, 1.0, 3.0, 10.0)
RHOS = (0.0, 0.1, 0.3, 1.0, 3.0)
SEEDS = (1, 2, 3, 4, 5)
NOISE_MULTIPLIERS = (0.0, 0.1, 0.3, 1.0, 2.5)
WARMUP_UPDATES = 50
IDENTITY_KEYS = frozenset(
    {
        "method",
        "learning_rate",
        "rho",
        "seed",
        "data_seed",
        "gradient_noise",
        "n_runs",
    }
)
FAILURE_FIELDS = (
    "phase",
    "method",
    "learning_rate",
    "rho",
    "seed",
    "gradient_noise",
    "data_seed",
    "error_type",
    "message",
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
    failures: list[dict],
    *,
    phase: str,
    method: str,
    learning_rate: float,
    rho: float,
    seed: int,
    gradient_noise: float | None,
    data_seed: int | None,
    error: Exception,
) -> None:
    """Append one failed run's context to the shared failure table."""
    failures.append(
        {
            "phase": phase,
            "method": method,
            "learning_rate": learning_rate,
            "rho": rho,
            "seed": seed,
            "gradient_noise": gradient_noise,
            "data_seed": data_seed,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    )


def sweep_grid(
    run_one: Callable[[float, float, int], dict],
    lr_grid: Sequence[float],
    rho_grid: Sequence[float],
    seeds: Sequence[int],
    *,
    method: str = "tuning",
    failures: list[dict] | None = None,
    data_seed: int | None = None,
    resolve_data_seed: Callable[[int], int] | None = None,
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
                            failures,
                            phase="tuning",
                            method=method,
                            learning_rate=learning_rate,
                            rho=rho,
                            seed=seed,
                            gradient_noise=None,
                            data_seed=(
                                resolve_data_seed(seed)
                                if resolve_data_seed is not None
                                else data_seed
                            ),
                            error=error,
                        )
                    continue
                rows.append(
                    {
                        "learning_rate": learning_rate,
                        "rho": rho,
                        "seed": seed,
                        **metrics,
                    }
                )
            if complete:
                results[(learning_rate, rho)] = rows
    return results


def mean_validation_loss(rows: Sequence[dict]) -> float:
    """Mean final validation loss of one grid point across its seeds."""
    return sum(row["validation_loss"] for row in rows) / len(rows)


def aggregate_metrics(
    rows: Sequence[dict],
    group_keys: Sequence[str],
) -> list[dict]:
    """Compute sample mean/std of numeric metrics over runs in each group.

    Numeric columns other than run identifiers are treated as metrics, so new
    metrics from the runner are aggregated without changing this function.
    """
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(row[column] for column in group_keys)
        groups.setdefault(key, []).append(row)
    metric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in IDENTITY_KEYS
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        }
    )
    aggregated = []
    for key, group in groups.items():
        result = dict(zip(group_keys, key))
        result["n_runs"] = len(group)
        for metric in metric_keys:
            values = [float(row[metric]) for row in group if metric in row]
            if not values:
                continue
            result[f"{metric}_mean"] = sum(values) / len(values)
            result[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        aggregated.append(result)
    return aggregated


def search_budget(
    *,
    tuning_rows: Sequence[dict],
    noise_rows: Sequence[dict],
    sensitivity_rows: Sequence[dict],
    settings: dict,
    seeds: Sequence[int],
    warmup_updates: int,
    failed_runs: int,
) -> dict:
    """Count completed runs, optimizer updates, and gradient evaluations.

    Warmup updates belong to one scale estimation per seed; they
    are optimizer updates but not full training runs, so ``total_runs`` excludes
    them while the update/evaluation totals include them.
    """
    updates = settings["updates"]
    batch_size = settings["batch_size"]
    warmup_updates_total = len(seeds) * warmup_updates

    def counts(rows: Sequence[dict]) -> dict:
        return {
            "runs": len(rows),
            "updates": len(rows) * updates,
            "gradient_evaluations": sum(
                row.get("gradient_evaluations", 0) for row in rows
            ),
        }

    parts = {
        "tuning": counts(tuning_rows),
        "noise_sweep": counts(noise_rows),
        "dataset_sensitivity": counts(sensitivity_rows),
        "warmup": {
            "runs": 0,
            "updates": warmup_updates_total,
            "gradient_evaluations": warmup_updates_total * batch_size,
        },
    }
    return {
        **parts,
        "failed_runs": failed_runs,
        "total_runs": sum(part["runs"] for part in parts.values()),
        "total_updates": sum(part["updates"] for part in parts.values()),
        "total_gradient_evaluations": sum(
            part["gradient_evaluations"] for part in parts.values()
        ),
    }


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
    path: Path,
    rows: Sequence[dict],
    fieldnames: Sequence[str] | None = None,
) -> None:
    """Write result rows with stable column order, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = {key for row in rows for key in row}
        leading = [
            key
            for key in ("method", "learning_rate", "rho", "seed", "gradient_noise")
            if key in keys
        ]
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
    data_seeds: Sequence[int] | None = None,
    data_sensitivity_seed: int | None = None,
    run_settings: dict | None = None,
    plot_runs: bool = False,
) -> dict:
    """Sweep hyperparameters, select on validation loss, and sweep noise.

    ``runner`` is either simulation wrapper, ``model_factory`` creates the
    simulation model from a seed, and ``sampler`` generates its data. With
    ``data_seed=None`` each seed gets its own dataset; otherwise all runs at a
    given seed share the override, keeping SGD/RSGD paired.

    ``data_seeds`` enables a separate study that holds the training seed fixed
    while varying the dataset; its first value becomes the fixed dataset of the
    main grid so training-seed spread is measured on one dataset.
    """
    if not seeds or not lr_grid or not rho_grid or warmup_updates < 1:
        raise ValueError("seeds, grids, and warmup_updates must be non-empty")
    if data_seeds is not None:
        if not data_seeds:
            raise ValueError("data_seeds must be non-empty when supplied")
        if data_seed is not None:
            raise ValueError("use either data_seed or data_seeds, not both")
        data_seed = data_seeds[0]
    if data_sensitivity_seed is not None and data_seeds is None:
        raise ValueError("data_sensitivity_seed requires data_seeds")

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
            optimizer_cls,
            name=f"tuning/{name}",
            rho=rho,
            seed=seed,
            learning_rate=learning_rate,
            data_seed=seed_data(seed),
            output_dir=str(root),
            save=False,
            **settings,
        )

    sgd_grid = sweep_grid(
        lambda lr, rho, seed: tune(lr, rho, seed, f"sgd_lr_{lr}"),
        lr_grid,
        [0.0],
        seeds,
        method="sgd",
        failures=failures,
        data_seed=data_seed,
        resolve_data_seed=seed_data,
    )
    sgd_lr, _ = select_best_parameters(sgd_grid)
    rsgd_grid = sweep_grid(
        lambda lr, rho, seed: tune(lr, rho, seed, f"rsgd_lr_{lr}_rho_{rho}"),
        lr_grid,
        rho_grid,
        seeds,
        method="rsgd",
        failures=failures,
        data_seed=data_seed,
        resolve_data_seed=seed_data,
    )
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

    scales = {"rsgd": gradient_scale(rsgd_lr, rsgd_rho)}
    shared_noise_grid = [scales["rsgd"]["scale"] * m for m in noise_multipliers]
    noise_grids = {"sgd": shared_noise_grid, "rsgd": shared_noise_grid}

    def sweep_noise(method: str, lr: float, rho_values: Sequence[float]) -> list[dict]:
        rows = []
        for seed in seeds:
            for noise in noise_grids[method]:
                for rho in rho_values:
                    setup_seed(seed)
                    try:
                        metrics = runner(
                            optimizer_cls,
                            name=f"noise_sweep/{method}_lr_{lr}_rho_{rho}",
                            rho=rho,
                            seed=seed,
                            learning_rate=lr,
                            gradient_noise=noise,
                            data_seed=seed_data(seed),
                            output_dir=str(root),
                            plot=plot_runs,
                            **settings,
                        )
                    except Exception as error:
                        record_failure(
                            failures,
                            phase="noise",
                            method=method,
                            learning_rate=lr,
                            rho=rho,
                            seed=seed,
                            gradient_noise=noise,
                            data_seed=seed_data(seed),
                            error=error,
                        )
                        continue
                    rows.append(
                        {
                            "method": method,
                            "learning_rate": lr,
                            "rho": rho,
                            "seed": seed,
                            "gradient_noise": noise,
                            **metrics,
                        }
                    )
        return rows

    noise_rows = sweep_noise("sgd", sgd_lr, [0.0]) + sweep_noise(
        "rsgd", rsgd_lr, rho_grid
    )

    sensitivity_rows: list[dict] = []
    sensitivity_seed = (
        seeds[0] if data_sensitivity_seed is None else data_sensitivity_seed
    )
    if data_seeds is not None:
        for sensitivity_data_seed in data_seeds:
            for method, lr, rho in (("rsgd", rsgd_lr, rsgd_rho),):
                for noise in noise_grids[method]:
                    setup_seed(sensitivity_seed)
                    try:
                        metrics = runner(
                            optimizer_cls,
                            name=f"dataset_sensitivity/{method}_lr_{lr}_rho_{rho}",
                            rho=rho,
                            seed=sensitivity_seed,
                            learning_rate=lr,
                            gradient_noise=noise,
                            data_seed=sensitivity_data_seed,
                            output_dir=str(root),
                            save=False,
                            **settings,
                        )
                    except Exception as error:
                        record_failure(
                            failures,
                            phase="dataset_sensitivity",
                            method=method,
                            learning_rate=lr,
                            rho=rho,
                            seed=sensitivity_seed,
                            gradient_noise=noise,
                            data_seed=sensitivity_data_seed,
                            error=error,
                        )
                        continue
                    sensitivity_rows.append(
                        {
                            "method": method,
                            "learning_rate": lr,
                            "rho": rho,
                            "seed": sensitivity_seed,
                            "data_seed": sensitivity_data_seed,
                            "gradient_noise": noise,
                            **metrics,
                        }
                    )
        write_csv(root / "dataset_sensitivity.csv", sensitivity_rows)
        write_csv(
            root / "dataset_sensitivity_aggregated.csv",
            aggregate_metrics(
                sensitivity_rows,
                ("method", "gradient_noise"),
            ),
        )

    sgd_rows = [{"method": "sgd", **row} for rows in sgd_grid.values() for row in rows]
    rsgd_rows = [
        {"method": "rsgd", **row} for rows in rsgd_grid.values() for row in rows
    ]
    tuning_rows = sgd_rows + rsgd_rows
    write_csv(root / "sgd_lr_grid.csv", sgd_rows) 
    write_csv(root / "rsgd_lr_rho_grid.csv", rsgd_rows)
    write_csv(
        root / "tuning_aggregated.csv",
        aggregate_metrics(
            tuning_rows,
            ("method", "learning_rate", "rho"),
        ),
    )
    write_csv(root / "noise_sweep_summary.csv", noise_rows)
    write_csv(
        root / "noise_sweep_aggregated.csv",
        aggregate_metrics(
            noise_rows,
            ("method", "learning_rate", "rho", "gradient_noise"),
        ),
    )
    write_csv(root / "failed_runs.csv", failures, fieldnames=FAILURE_FIELDS)
    budget = search_budget(
        tuning_rows=tuning_rows,
        noise_rows=noise_rows,
        sensitivity_rows=sensitivity_rows,
        settings=settings,
        seeds=seeds,
        warmup_updates=warmup_updates,
        failed_runs=len(failures),
    )
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
            "mean_validation_loss": mean_validation_loss(
                rsgd_grid[(rsgd_lr, rsgd_rho)]
            ),
        },
        "gradient_scale": scales,
        "noise_grid": noise_grids,
        "search_budget": budget,
        "dataset_sensitivity": {
            "enabled": data_seeds is not None,
            "training_seed": sensitivity_seed if data_seeds is not None else None,
            "data_seeds": list(data_seeds) if data_seeds is not None else [],
            "runs": len(sensitivity_rows),
            "file": "dataset_sensitivity.csv" if data_seeds is not None else None,
            "aggregated_file": (
                "dataset_sensitivity_aggregated.csv" if data_seeds is not None else None
            ),
        },
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
