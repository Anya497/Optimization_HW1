import torch


def get_cls_data(
    n_points: int, features_num: int = 2, y_prob: float = 0.5
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    dataset = list()
    for _ in range(n_points):
        y = torch.bernoulli(torch.tensor([y_prob]))
        x = torch.normal(
            torch.fill(torch.empty(features_num), 2 * y - 1),
            std=torch.eye(features_num),
        )
        dataset.append((x, y))
    return dataset