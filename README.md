# Optimization_HW1
SGD versus Robust SGD under Gradient Uncertainty
## Requirements

* python==3.13
* torch==2.15

## Project layout

- `src/data/log_regression.py`: Simulation 1 Gaussian sampler with Bernoulli labels.
- `src/data/semicircles.py`: Simulation 2 semicircle sampler.
- `src/data/common.py`: shared `make_dataset`/`make_loaders`, splitting, and batch sampling.
- `src/models.py`: the two-layer network; Simulation 1 uses `torch.nn.Linear`.
- `src/grad_noise.py`: per-example gradients with optional Gaussian noise.
- `src/train.py`: shared training and evaluation.
- `experiments/run_logreg_exp.py`: Simulation 1 setup using the shared data pipeline.
- `experiments/run_semicircle_exp.py`: Simulation 2 setup using the shared data pipeline.
- `experiments/common.py`: shared `run_experiment`, seed setup, saving results, and plotting.
- `experiments/grid_search.py`: validation-loss selection and gradient-noise sweep.
- `experiments/run_grid_search.py`: CLI entry point choosing the simulation.
- `tests/`: data, gradient, model, and training checks.

## Shared data pipeline

Pass a sampler function to choose which distribution generates the points:

```python
from src.data.common import make_dataset, make_loaders
from src.data.log_regression import sample_points as sample_logreg
from src.data.semicircles import sample_points as sample_semicircles

dataset = make_dataset(1000, sample_logreg, seed=42)
loaders = make_loaders(sample_logreg, batch_size=32, updates=1000)
# For Simulation 2, use the same function with the other sampler:
loaders = make_loaders(sample_semicircles, batch_size=32, updates=1000)
```

Each sampler accepts `(size, generator)` and returns `(points, labels)` shaped
`(n, 2)` and `(n,)`. The Gaussian sampler draws half its labels with Bernoulli(0.5)
and pairs them with their complements to keep equal class counts. Both samplers
produce balanced labels; use even split sizes.

The shared factory seeds the generator, shuffles the examples, and returns a
`TensorDataset` with scalar labels. The loader factory creates separate
training/validation/test datasets and samples training batches with replacement.
Plotting also supports list-based datasets with one-element labels.

## Simulation 2

Settings are ordinary function arguments with defaults; there is no config file.
`src/data/semicircles.py` includes the full generator: `sample_angles` samples t
uniformly from [0, pi], and `points_from_angles` implements the two
semicircle formulas with independent Gaussian data noise of standard deviation
0.1. Shared dataset/loader functions are in `src/data/common.py`. Each split is
balanced and independently generated; no preprocessing is used.
Training batches are sampled independently with replacement, as the statement
requires. Reusing the same seeds reproduces datasets and batch sequences.

Run the data checks with `python -m pytest -q tests/test_data.py`.

### Training and optimizers

Install the dependencies with `python -m pip install torch==2.15.0 matplotlib pytest`.
The training code runs on CPU and uses stable binary cross-entropy from logits,
without weight decay or preprocessing. It uses per-example automatic gradients,
adds independent Gaussian gradient noise, and evaluates only uncorrupted data.
Both methods use the schedule `lr / sqrt(1 + k / decay_scale)` with zero-based k.
The returned metrics include `validation_loss` and `validation_accuracy` from the
final uncorrupted validation set; hyperparameter selection uses the loss.

`train(model, loaders, optimizer)` accepts a `torch.optim.Optimizer` instance.
It uses the `RobustSGD` implementation in `src/optimizer.py`:

```python
optimizer.zero_grad()
# noisy_grads maps each model parameter to a tensor shaped (batch_size, *parameter.shape).
optimizer.step(noisy_grads=noisy_grads)
```

The optimizer computes the mean gradient, uncertainty, and parameter updates.
Learning-rate decay is applied through each parameter group's `lr`, preserving
any differences between groups. Use `RobustSGD` for both methods: `rho=0` is the
ordinary SGD baseline, and `rho>0` enables robustness.
After each step, `optimizer.zero_step` reports whether `g^T S^-1 g <= rho^2`.
Zero-step frequency counts this condition, rather than unchanged parameters due
to rounding or a zero learning rate. Uncertainty estimation needs at least two
per-example observations per update; a single observation raises `ValueError`.

```python
from experiments.run_semicircle_exp import run_experiment
from src.optimizer import RobustSGD

seed = 0
gradient_noise = 0.1
learning_rate = 0.1
metrics = run_experiment(RobustSGD, name="sgd", rho=0, seed=seed,
                         gradient_noise=gradient_noise, learning_rate=learning_rate)
print(metrics)
```

`experiments.run_logreg_exp.run_experiment` has the same signature and settings,
but uses the Gaussian sampler and the logistic model, and writes to `results/logreg`.
Both runners delegate to `experiments.common.run_experiment`.

For R-SGD, use the same class with, for example, `rho=0.5` and
`name="rsgd_rho_0.5"`. The radius is passed to its constructor and saved with
the run settings.

Reuse the same settings and seeds for paired SGD/R-SGD comparisons; each call
recreates its model, data, batches, and gradient-noise stream. `data_seed` fixes
the dataset independently of `seed`, which controls initialization and training
randomness. Each experiment creates a fresh optimizer instance for its model.
Repeat over at least five seeds and noise levels when conducting the full study;
the grid-search driver below performs that tuning and aggregation. Freeze tuned
settings before comparing noise levels and keep test data out of selection.

Each run saves settings/environment metadata, final training/test losses, test
accuracy, runtime, per-example gradient-evaluation count, zero-step frequency,
loss history, weights, loss curves, and a test decision boundary under `results/`.
Plotting reads batches from the test loader, so it supports both the shared
`TensorDataset` format with scalar labels and legacy list-based data with
one-element labels.
`runtime_seconds` includes intermediate loss evaluation, but excludes initial/final
evaluation and saving/plotting; `update_seconds` measures update computation only.
The folder identifies optimizer/run name, seed, gradient noise, and initial LR;
use a different `output_dir` for studies changing other settings to avoid overwrites.

### Model

`src/models.py` implements exactly `Linear(2, 8) -> tanh -> Linear(8, 1)`
with 33 trainable parameters. It returns raw logits, with no final sigmoid;
evaluation classifies at logit zero. Initialization uses Xavier-uniform weights
(tanh gain for the hidden layer, unit gain for the output) and zero biases.
Passing the same seed produces identical starting parameters without changing
the caller's CPU random stream. No dropout, normalization, or regularization is used.

Run all checks with `python -m pytest -q`. These include
finite differences for all 33 parameters, per-example gradients, paired training,
the real optimizer at `rho=0` and `rho=0.5`, zero-step frequency, the separation of
gradient noise from data, and result saving for both dataset formats. Tests that
record optimizer inputs subclass `RobustSGD` and delegate updates to its real
implementation.

Simple checks also cover both data samplers, separate dataset splits, logistic
regression through the shared trainer, known evaluation scores, and a six-update
experiment that saves its settings, metrics, weights, and plots.

Training budgets are counted in optimizer updates. With replacement sampling, an
epoch can be defined as a chosen number of sampled examples, but does not
guarantee that every stored example is visited. The assignment does not require
epochs; it requires comparable budgets and loss plots against updates.
The local checks and output-generation smoke run used Python 3.12.14 and PyTorch
2.15.0; Python 3.13 remains the repository's declared target.

## Hyperparameter grid search

`experiments.grid_search.run_grid_search` drives both simulations. The SGD
series varies `learning_rate` over `{1e-3, 1e-2, 1e-1, 1}` at `rho=0`; the RSGD
series varies the same learning rates over `rho` in `{0, 0.1, 0.3, 1.0, 3.0}`
(`rho=0` included). Each point runs for seeds `{1, 2, 3, 4, 5}`, and the point
with the smallest **mean validation loss** across seeds is selected; test data
is never used for selection.

The gradient-noise scale is measured from 50 warmup updates at the selected
parameters: for each update take the mean L2 norm of the flattened per-example
gradients and divide by `sqrt(number of trainable parameters)`, then average
over seeds. Each method has its own scale; the sweep uses
`scale * {0, 0.1, 0.3, 1, 2.5}` for both SGD and RSGD. These runs save the full
per-run artifacts; tuning points only contribute metrics to the summary tables.

Outputs are written under `results/<experiment>/grid_search/`:

- `sgd_lr_grid.csv`, `rsgd_lr_rho_grid.csv`: every tuning point for every seed.
- `noise_sweep_summary.csv`: every method/seed/noise run and its metrics.
- `tuning_aggregated.csv`, `noise_sweep_aggregated.csv`: per-group sample mean
  and standard deviation of every metric across successful evaluation seeds
  (`n_runs` counts those seeds; std uses the n−1 denominator). The error bars
  therefore describe seed-to-seed variability, not measurement error.
- `failed_runs.csv`: context and error for every failed run. A tuning point with
  any failed seed is dropped from selection so all remaining points use the
  same number of seeds; failed noise runs are skipped but recorded.
- `selection.json`: chosen parameters, mean validation losses, per-seed scales,
  noise grids, and a `variability` block referencing the aggregated files.
- `noise_sweep/`: one folder of saved run artifacts per method and selected
  parameter pair. Tuning runs use `save=False`, so no `tuning/` folder is written.

Run the search from the repository root:

```bash
python -m experiments.run_grid_search --experiment logreg
python -m experiments.run_grid_search --experiment semicircle
```

`--experiment` is required. Comma-separated `--seeds`, `--lr-grid`, `--rho-grid`,
and `--noise-multipliers` override the defaults above. Training settings can be
overridden with `--updates`, `--warmup-updates`, `--batch-size`, `--train-size`,
`--validation-size`, `--test-size`, `--decay-scale`, and `--eval-every`;
`--output-dir` changes the results root. By default each seed uses its own
dataset (`data_seed=seed`), while `--data-seed` pins one dataset for all seeds
so methods stay paired. See `--help` for the full list.

## Batch-size uncertainty check

Run `python -m experiments.run_batch_size_exp` from the repository root.
The experiment keeps one logistic-regression model fixed on 1,000 Simulation 1
examples and samples 200 batches with replacement for each size in
`[8, 16, 32, 64, 128]`. There is no training or artificial gradient noise.
It computes Equation (4), including the ridge, and saves results under
`experiments/batch_size_outputs/`: full mean-S heatmaps in PNG/PDF, settings and
summary matrices in JSON, and every sampled diagonal and ridge in CSV.
The second heatmap row shows batch size times mean S to check the expected
approximately inverse-batch-size scaling. Each row uses a shared colour scale.
The off-diagonal entries are zero because the homework specifies a diagonal S.
