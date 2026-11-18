# Plots for the homework

The source is `HW_robust_sgd.pdf`, Sections 2(c), 3 and 4. The plotting command
reads the existing grid-search outputs; it does not train or tune anything.
Grid searches save per-run data and weights without individual figures by
default; use `--plot-runs` on the existing grid-search command to enable those.
Standalone runs keep their figures by default (`plot=False` disables them).

## Required outputs and their figures

For **each simulation**, the command writes six figures under `grid_search/plots/`:

| File | What it shows | Why it is included |
| --- | --- | --- |
| `loss_curves.png` | Training and validation loss versus updates, SGD/R-SGD together, one row per noise level | Explicitly required by Section 3 |
| `validation_accuracy.png` | Validation accuracy versus updates at every noise level, with mean ± sample SD across seeds | Additional prediction-quality view; recorded at the same updates as the losses |
| `decision_boundaries.png` | Both selected models on the test set, at one stated evaluation seed, for every noise level | Section 3 explicitly requires a decision boundary for a stated seed; showing every noise level makes comparison easier |
| `noise_summary.png` | Final training loss, test loss, test accuracy, runtime, gradient-evaluation count and zero-step frequency versus noise | Sections 3 and 4 require reporting these quantities; plotting them is our presentation choice |
| `radius_sensitivity.png` | Training loss, validation loss and zero-step frequency versus radius, at each noise level | Section 4.4 requires a small sensitivity study of loss and zero-step frequency; the exact plot format is our choice |
| `tuning.png` | Mean validation loss over learning rates/radii; selected cells outlined | Supports Section 4.2's tuning justification and checking selections at the grid boundaries; a heatmap is optional |

Section 2(c)'s batch-size uncertainty check is a separate experiment, already
implemented on the batch-size branch/PR. It does not need another training study.
The solver checks in Section 2(b) need numerical evidence, not a prescribed plot.

## Fair comparisons and error bars

The main loss/validation-accuracy curves, noise summary and boundaries use the settings saved in
`selection.json`: SGD at its selected learning rate and rho=0; R-SGD at its
selected learning rate and radius. Those choices stay frozen across noise
levels. The plotter does not choose a better radius using test results or the
noise-sweep scores. If validation selected rho=0 for R-SGD, it is reported as
such, even if the two selected methods coincide.

The sensitivity plot instead includes every radius in the saved R-SGD sweep,
holding the R-SGD learning rate fixed. Validation-based tuning and radius
sensitivity use no test scores.

Curves and error bars show the mean plus/minus **one sample standard deviation**
across successful seeds (denominator n-1), not a confidence interval or standard
error. Loss legends show the successful seed count for each noise level. Counts
for every main comparison are in `plot_manifest.json`; the source table is
`noise_sweep_aggregated.csv` in the grid-search directory.
A single successful seed has no estimated spread; its plotted band/error is zero.
At least five evaluation seeds are required for the actual homework study.

With one fixed `data_seed`, spread reflects training randomness on one dataset.
With the grid-search default `data_seed=None`, it combines dataset and training
randomness. The manifest records which interpretation applies. Failed runs
remain recorded by the existing pipeline in `failed_runs.csv`; missing runs are
not replaced by zero-valued measurements. A missing boundary run is labelled
unavailable.

Decision boundaries reload the saved weights and regenerate the exact test set
using that run's sampler, sample size and data seed. The black line is logit zero;
the background represents class-1 probability. There is no boundary line if the
model predicts one class throughout the displayed region.

The final-metric and radius figures read means and standard deviations directly
from the existing `noise_sweep_aggregated.csv`, filtering for the appropriate
settings. No second metric table or duplicate aggregation is created. Loss
histories are averaged across seeds separately because the grid search does
not aggregate those curves. Compare training loss with test loss/accuracy to discuss
whether optimization and prediction rank the methods the same way. A reduction
in loss alone is not evidence of better accuracy.

These figures do not isolate step-length effects. Discuss that limitation;
a learning-rate-matched SGD control or logged displacement norms would be a
focused further experiment. The plotter does not change the optimizer or trainer.

## Run on existing results

From the repository root, after running the existing grid-search command:

```sh
python -m experiments.plot_results results/logreg/grid_search --seed 1
python -m experiments.plot_results results/semicircle/grid_search --seed 1
```

Choose a seed present in the saved evaluation runs. Without `--seed`, the first
seed in `selection.json` is used. No config file or additional dependency is needed.

Validation accuracy is recorded in each loss-history row, starting at update
zero and then every `eval_every` updates. It reuses the validation evaluation
already needed for loss, so there is no additional validation pass. Old histories
without this column still produce the original five figures; the accuracy
curve is skipped with a message. Intermediate accuracies cannot be recovered
from final weights alone. Replaying the selected runs with the same settings
and seeds records the missing history without retuning.

## Local verification examples

The example figures under `results/plot_smoke/` use reduced budgets: 30 updates,
five seeds, a fixed dataset seed of 42, two noise levels (multipliers 0 and 1),
learning rates 0.1 and 0.3, and radii 0, 0.5 and 1. Each simulation has 100
training points, 40 validation points, 40 test points, batch size 16, five warmup
updates and loss recording every ten updates. Boundary evaluation uses seed 1.
These are plotting checks, not final research results or the full homework study.

```sh
python -m experiments.run_grid_search --experiment logreg --output-dir results/plot_smoke --seeds 1,2,3,4,5 --lr-grid 0.1,0.3 --rho-grid 0,0.5,1 --noise-multipliers 0,1 --updates 30 --warmup-updates 5 --train-size 100 --validation-size 40 --test-size 40 --batch-size 16 --eval-every 10 --data-seed 42
python -m experiments.plot_results results/plot_smoke/logreg/grid_search --seed 1
```

Use `--experiment semicircle` for the other simulation and its corresponding
results path in the plotting command.

## Descent trajectories

The separate [logistic descent experiment](logreg_descent.md) trains paired
SGD/R-SGD runs and records every parameter update. It provides looping GIFs
of two trainable weights on exact loss contours and a 3D loss surface,
with bias fixed at zero for this visualization experiment. These are
illustrative additions to the required report figures.
