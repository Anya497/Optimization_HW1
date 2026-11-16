import torch


def sample_points(size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample balanced Gaussian classes using Bernoulli labels and their complements."""
    labels = torch.bernoulli(torch.full((size // 2,), 0.5), generator=generator)
    labels = torch.cat((labels, 1 - labels))
    means = (2 * labels - 1).unsqueeze(1)
    points = means + torch.randn(size, 2, generator=generator)
    return points, labels
