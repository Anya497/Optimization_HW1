# Global plan

Integration branch: `anya` by explicit user request on 2026-11-17
("нужно, чтобы все было в ветке Anya"). `git config --get agent.integrationBranch`
is unset, so the default would otherwise be `main`.

## Grid search for SGD/RSGD hyperparameters

Goal: add a grid-search entry point for both simulations, select hyperparameters
on validation data, sweep gradient noise with a data-driven scale, and persist
all results.

- **T1** — Grid-search entry point for `logreg` and `semicircle`, validation-loss
  selection over seeds, warmup-based gradient-noise scale, noise sweep and result
  persistence. Depends on: nothing. Status: `[done]` (91c55e4). Outcome:
  one CLI for both simulations; six commits merged into `anya`; two quick
  verification runs produced grid CSVs, selection JSON, and 16 saved noise-sweep
  runs under `/tmp/opencode/grid_smoke`.

Conflicts/overlaps: T1 touches `src/train.py`, `src/optimizer.py`,
`experiments/common.py`, `experiments/run_logreg_exp.py`,
`experiments/run_semicircle_exp.py`, `README.md`, and adds new files under
`experiments/`, `tests/`, `tasks/`. The working tree already contains uncommitted
logistic-regression plumbing (`LogReg` model and `run_logreg_exp.run_experiment`)
that T1 needs; it is committed with T1 rather than reimplemented.

Reuse: `experiments/common.py` (`setup_seed`, `save_results`), `src/data/common.py`
(`make_loaders`), `src/models.py` (`SemicircleNet`, `LogReg`), `src/train.py`
(`train`, `evaluate`), `src/optimizer.py` (`RobustSGD`),
`src/grad_noise.py` (`per_example_gradients`).

## Reporting data for tasks 3–5

- **T2** — Add (2) per-seed variability aggregation and failed-run tracking,
  (3) a dataset-vs-training randomness study with CLI flags, and (4) an explicit
  search budget in `selection.json`. Depends on: T1. Status: `[in progress]`
  (feature branch `T2-report-data`, review fixes pending).

Overlap: T2 and T1 both touch `experiments/grid_search.py`,
`experiments/run_grid_search.py`, `tests/test_grid_search.py`, and `README.md`;
T1 is already integrated, so no branch conflict is expected. Item 1 (semicircle
generator mismatch) and item 6 (uncertainty-ridge formula) were explicitly left
out by the user.
