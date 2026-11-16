import math

import torch


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


def sample_points(labels: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Sample Simulation 2 points for the supplied class labels."""
    angles = sample_angles(len(labels), generator)
    return points_from_angles(angles, labels, generator)
