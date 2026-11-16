"""Shared training and evaluation for both simulations."""

import math
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .grad_noise import compute_noisy_per_example_grads, per_example_gradients


def eval_bce_with_logits(X, y, model):
    logits = model(X).squeeze(-1)
    bce = F.binary_cross_entropy_with_logits(logits, y, reduction="mean")
    return bce


def train_with_noise(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    n_steps: int,
    sigma: float,
):
    it = iter(loader)

    for step in range(n_steps):
        try:
            X_batch, y_batch = next(it)
        except StopIteration:
            it = iter(loader)
            X_batch, y_batch = next(it)
        logits = model(X_batch).squeeze(-1)
        losses = F.binary_cross_entropy_with_logits(
            logits, y_batch.squeeze(-1), reduction="none"
        )
        V = compute_noisy_per_example_grads(model, losses, sigma)
        optimizer.step(V)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    """Evaluate the original data and stable BCE from logits, without corruption."""
    model.eval()
    loss, correct, count = 0.0, 0, 0
    for x, y in loader:
        logits = model(x)
        loss += F.binary_cross_entropy_with_logits(logits, y, reduction="sum").item()
        correct += ((logits >= 0) == y.bool()).sum().item()
        count += y.numel()
    return {"loss": loss / count, "accuracy": correct / count}


def train(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    optimizer: torch.optim.Optimizer,
    decay_scale: float = 50.0,
    gradient_noise: float = 0.0,
    seed: int = 0,
    eval_every: int = 50,
) -> tuple[list[dict], dict]:
    """Train with the RobustSGD optimizer interface from the optimizer PR.

    Pass per-example observations to step(noisy_grads={parameter: gradients}).
    The optimizer computes gradient means, uncertainty, and parameter updates.
    Recreate the model, optimizer, and loaders with the same seed for paired runs.
    """
    updates = len(loaders["train"])
    noise_generator = torch.Generator().manual_seed(seed + 1)
    initial_lrs = [group["lr"] for group in optimizer.param_groups]
    history = []
    zero_steps = 0
    gradient_evaluations = 0

    def record(update):
        history.append({
            "update": update,
            "train_loss": evaluate(model, loaders["train_eval"])["loss"],
            "validation_loss": evaluate(model, loaders["validation"])["loss"],
        })

    record(0)
    started = time.perf_counter()
    update_seconds = 0.0
    for update, (x, y) in enumerate(loaders["train"], start=1):
        step_started = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        noisy_grads = per_example_gradients(model, x, y)
        for parameter, gradients in noisy_grads.items():
            noisy_grads[parameter] = gradients + gradient_noise * torch.randn(
                gradients.shape, generator=noise_generator, dtype=gradients.dtype
            )
        parameters = list(model.parameters())
        for group, initial_lr in zip(optimizer.param_groups, initial_lrs):
            group["lr"] = initial_lr / math.sqrt(1 + (update - 1) / decay_scale)
        previous = [parameter.detach().clone() for parameter in parameters]
        optimizer.step(noisy_grads=noisy_grads)
        zero_steps += int(all(torch.equal(parameter, before)
                              for parameter, before in zip(parameters, previous)))
        gradient_evaluations += len(x)
        update_seconds += time.perf_counter() - step_started
        if update % eval_every == 0 or update == updates:
            record(update)

    runtime_seconds = time.perf_counter() - started
    train_metrics = evaluate(model, loaders["train_eval"])
    test_metrics = evaluate(model, loaders["test"])
    return history, {
        "train_loss": train_metrics["loss"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "runtime_seconds": runtime_seconds,
        "update_seconds": update_seconds,
        "gradient_evaluations": gradient_evaluations,
        "zero_steps": zero_steps,
        "zero_step_frequency": zero_steps / updates,
    }
