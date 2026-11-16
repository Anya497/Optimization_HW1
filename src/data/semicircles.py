import math

import torch
from torch.utils.data import DataLoader, TensorDataset

from .common import create_minibatch_dataloader


def sample_angles(size: int, generator: torch.Generator) -> torch.Tensor:
    """Draw independent t ~ Uniform[0, pi)."""
    return math.pi * torch.rand(size, generator=generator)


def points_from_angles(
    angles: torch.Tensor,
    labels: torch.Tensor,
    generator: torch.Generator,
    noise_std: float = 0.1,
) -> torch.Tensor:
    """Apply the two class formulas to already sampled angles, then add e."""
    class_one = labels.bool()
    x = torch.where(class_one, 1 - angles.cos(), angles.cos())
    y = torch.where(class_one, 0.5 - angles.sin(), angles.sin())
    points = torch.stack((x, y), dim=1)
    # e ~ N(0, 0.1^2 I): this is data noise, separate from gradient corruption.
    return points + noise_std * torch.randn(
        points.shape, generator=generator, dtype=points.dtype
    )


def make_dataset(size: int, seed: int) -> TensorDataset:
    """Generate a balanced, shuffled split of an even number of examples."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.cat((torch.zeros(size // 2), torch.ones(size // 2)))
    angles = sample_angles(size, generator)
    points = points_from_angles(angles, labels, generator)
    order = torch.randperm(size, generator=generator)
    return TensorDataset(points[order], labels[order])


def make_loaders(
    train_size: int = 1000,
    validation_size: int = 400,
    test_size: int = 400,
    batch_size: int = 32,
    updates: int = 1000,
    data_seed: int = 42,
    seed: int = 0,
) -> dict[str, DataLoader]:
    train = make_dataset(train_size, data_seed)
    validation = make_dataset(validation_size, data_seed + 1)
    test = make_dataset(test_size, data_seed + 2)
    # Recreating the loaders with the same seed pairs batches across optimizers.
    return {
        "train": create_minibatch_dataloader(train, batch_size, seed, updates),
        "train_eval": DataLoader(train, batch_size=256),
        "validation": DataLoader(validation, batch_size=256),
        "test": DataLoader(test, batch_size=256),
    }
