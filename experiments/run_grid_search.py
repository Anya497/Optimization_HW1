"""Command-line entry point for the SGD/RSGD grid search."""

import argparse
from pathlib import Path

from src.data.log_regression import sample_points as sample_logreg
from src.data.semicircles import sample_points as sample_semicircle
from src.models import LogReg, SemicircleNet

from .grid_search import (
    DEFAULT_RUN_SETTINGS,
    LEARNING_RATES,
    NOISE_MULTIPLIERS,
    RHOS,
    SEEDS,
    WARMUP_UPDATES,
    run_grid_search,
)
from .run_logreg_exp import run_experiment as run_logreg
from .run_semicircle_exp import run_experiment as run_semicircle

EXPERIMENTS = {
    "logreg": (run_logreg, LogReg, sample_logreg),
    "semicircle": (run_semicircle, SemicircleNet, sample_semicircle),
}


def _numbers(raw: str, converter, label: str) -> tuple:
    """Parse a comma-separated CLI grid and reject empty or malformed values."""
    try:
        values = tuple(converter(item) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid {label} grid: {raw}") from error
    if not values:
        raise argparse.ArgumentTypeError(f"empty {label} grid")
    return values


def _floats(raw: str) -> tuple[float, ...]:
    return _numbers(raw, float, "float")


def _ints(raw: str) -> tuple[int, ...]:
    return _numbers(raw, int, "integer")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser with all grid and training overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--plot-runs", action="store_true", help="Also save a figure for every noise-sweep run")
    parser.add_argument("--seeds", type=_ints, default=SEEDS)
    parser.add_argument("--lr-grid", type=_floats, default=LEARNING_RATES)
    parser.add_argument("--rho-grid", type=_floats, default=RHOS)
    parser.add_argument("--noise-multipliers", type=_floats, default=NOISE_MULTIPLIERS)
    parser.add_argument("--warmup-updates", type=int, default=WARMUP_UPDATES)
    parser.add_argument("--train-size", type=int)
    parser.add_argument("--validation-size", type=int)
    parser.add_argument("--test-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--decay-scale", type=float)
    parser.add_argument("--eval-every", type=int)
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--data-seed", type=int)
    seed_group.add_argument("--data-seeds", type=_ints)
    parser.add_argument("--data-sensitivity-seed", type=int)
    return parser


def main(argv: list[str] | None = None) -> dict:
    """Run the requested simulation's grid search and return its summary."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.data_sensitivity_seed is not None and args.data_seeds is None:
        parser.error("--data-sensitivity-seed requires --data-seeds")
    runner, model_factory, sampler = EXPERIMENTS[args.experiment]
    run_settings = {
        key: value for key, value in vars(args).items()
        if key in DEFAULT_RUN_SETTINGS and value is not None
    }
    summary = run_grid_search(
        runner, model_factory, sampler,
        experiment=args.experiment,
        output_dir=args.output_dir,
        seeds=args.seeds,
        lr_grid=args.lr_grid,
        rho_grid=args.rho_grid,
        noise_multipliers=args.noise_multipliers,
        warmup_updates=args.warmup_updates,
        data_seed=args.data_seed,
        data_seeds=args.data_seeds,
        data_sensitivity_seed=args.data_sensitivity_seed,
        run_settings=run_settings,
        plot_runs=args.plot_runs,
    )
    destination = Path(args.output_dir) / args.experiment / "grid_search"
    print(
        f"{args.experiment}: SGD lr={summary['sgd']['learning_rate']},"
        f"RSGD lr={summary['rsgd']['learning_rate']} rho={summary['rsgd']['rho']}; "
        f"results in {destination}"
    )
    return summary


if __name__ == "__main__":
    main()
