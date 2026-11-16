# Optimization_HW1
SGD versus Robust SGD under Gradient Uncertainty
## Requirements

* python==3.13
* torch==2.15

## Project layout

- `src/data/log_regression.py`: Simulation 1 data generation.
- `src/data/semicircles.py`: Simulation 2 data generation.
- `src/data/common.py`: splitting and replacement-sampling loaders.
- `src/models.py`: the two-layer network; Simulation 1 uses `torch.nn.Linear`.
- `src/grad_noise.py`: per-example gradient and noise helpers.
- `src/train.py`: shared training and evaluation, including the existing `train_with_noise` API.
- `experiments/run_logreg_exp.py`: the existing Simulation 1 setup.
- `experiments/run_semicircle_exp.py`: Simulation 2 orchestration via `run_experiment`.
- `experiments/common.py`: seed setup, saving results, and plotting.
- `tests/`: data, gradient, model, and training checks.

## Simulation 2

Settings are ordinary function arguments with defaults; there is no config file.
`src/data/semicircles.py` includes the full generator: `sample_angles` samples t
uniformly from [0, pi], and `points_from_angles` implements the two
semicircle formulas with independent Gaussian data noise of standard deviation
0.1. Each split is balanced and independently generated; no preprocessing is used.
Training batches are sampled independently with replacement, as the statement
requires. Reusing the same seeds reproduces datasets and batch sequences.

Run the data checks with `python -m unittest discover -s tests`.

### Training and optimizers

Install the dependencies with `python -m pip install torch==2.15.0 matplotlib`.
The training code runs on CPU and uses stable binary cross-entropy from logits,
without weight decay or preprocessing. It uses per-example automatic gradients,
adds independent Gaussian gradient noise, and evaluates only uncorrupted data.
Both methods use the schedule `lr / sqrt(1 + k / decay_scale)` with zero-based k.

`train(model, loaders, optimizer)` accepts a `torch.optim.Optimizer` instance.
It follows the incoming `RobustSGD` interface from optimizer PR #2:

```python
optimizer.zero_grad()
# noisy_grads maps each model parameter to a tensor shaped (batch_size, *parameter.shape).
optimizer.step(noisy_grads=noisy_grads)
```

The optimizer computes the mean gradient, uncertainty, and parameter updates.
Learning-rate decay is applied through each parameter group's `lr`, preserving
any differences between groups. Use `RobustSGD` for both methods: `rho=0` is the
ordinary SGD baseline, and `rho>0` enables robustness. The optimizer PR is kept
separate and must be merged before using its class in this branch.

After that PR is merged:

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

For R-SGD, use the same class with, for example, `rho=0.5` and
`name="rsgd_rho_0.5"`. The radius is passed to its constructor and saved with
the run settings.

Reuse the same settings and seeds for paired SGD/R-SGD comparisons; each call
recreates its model, data, batches, and gradient-noise stream. `data_seed` fixes
the dataset independently of `seed`, which controls initialization and training
randomness. Each experiment creates a fresh optimizer instance for its model.
Repeat over at least five seeds and noise levels when conducting the full study;
tuning and cross-run aggregation are left to the experiment caller. Freeze tuned
settings before comparing noise levels and keep test data out of selection.

Each run saves settings/environment metadata, final training/test losses, test
accuracy, runtime, per-example gradient-evaluation count, zero-step frequency,
loss history, weights, loss curves, and a test decision boundary under `results/`.
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

Run all checks with `python -m unittest discover -s tests -v`. These include
finite differences for all 33 parameters, per-example gradients, paired training,
the incoming optimizer interface, zero-step frequency, and the separation of
gradient noise from data. Interface tests use a test double; the optimizer
implementation itself is maintained in the separate PR.
The local checks and output-generation smoke run used Python 3.12.14 and PyTorch
2.15.0; Python 3.13 remains the repository's declared target.
