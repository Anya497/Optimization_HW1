from typing import Iterator

import torch
from src.grad_noise import compute_noisy_per_example_grads
from torch.utils.data import DataLoader
import torch.nn.functional as F


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
