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
  persistence. Depends on: nothing. Status: `[open]`.

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
