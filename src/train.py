"""Shared training and evaluation for both simulations."""

import math
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .grad_noise import per_example_gradients


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict[str, float]:
    """Evaluate the original data and stable BCE from logits, without corruption."""
    model.eval()
    loss, correct, count = 0.0, 0, 0
    for x, y in loader:
        logits = model(x).reshape(-1)
        y = y.reshape(-1)
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
    """Train with RobustSGD.

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
        train_metrics = evaluate(model, loaders["train_eval"])
        validation_metrics = evaluate(model, loaders["validation"])
        history.append({
            "update": update,
            "train_loss": train_metrics["loss"],
            "validation_loss": validation_metrics["loss"],
            "validation_accuracy": validation_metrics["accuracy"],
        })

    record(0)
    started = time.perf_counter()
    update_seconds = 0.0
    for update, (x, y) in enumerate(loaders["train"], start=1):
        step_started = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        noisy_grads = per_example_gradients(
            model, x, y, sigma=gradient_noise, generator=noise_generator
        )
        for group, initial_lr in zip(optimizer.param_groups, initial_lrs):
            group["lr"] = initial_lr / math.sqrt(1 + (update - 1) / decay_scale)
        optimizer.step(noisy_grads=noisy_grads)
        zero_steps += int(optimizer.zero_step)
        gradient_evaluations += len(x)
        update_seconds += time.perf_counter() - step_started
        if update % eval_every == 0 or update == updates:
            record(update)

    runtime_seconds = time.perf_counter() - started
    train_metrics = evaluate(model, loaders["train_eval"])
    validation_metrics = evaluate(model, loaders["validation"])
    test_metrics = evaluate(model, loaders["test"])
    return history, {
        "train_loss": train_metrics["loss"],
        "validation_loss": validation_metrics["loss"],
        "validation_accuracy": validation_metrics["accuracy"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "runtime_seconds": runtime_seconds,
        "update_seconds": update_seconds,
        "gradient_evaluations": gradient_evaluations,
        "zero_steps": zero_steps,
        "zero_step_frequency": zero_steps / updates,
    }
