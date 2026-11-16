import torch


def sample_points(labels: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Sample Simulation 1: class means (-1, -1)/(1, 1), covariance I2."""
    means = (2 * labels - 1).unsqueeze(1)
    return means + torch.randn(len(labels), 2, generator=generator)


def get_cls_data(
    n_points: int, features_num: int = 2, y_prob: float = 0.5
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    dataset = list()
    for _ in range(n_points):
        y = torch.bernoulli(torch.tensor([y_prob]))
        x = (2 * y - 1) + torch.randn(features_num)
        dataset.append((x, y))
    return dataset
