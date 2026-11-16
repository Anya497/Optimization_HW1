import torch


def compute_noisy_per_example_grads(
    model: torch.nn.Module, losses: torch.Tensor, sigma: float
):
    params = [p for p in model.parameters() if p.requires_grad]

    rows = []
    for i in range(losses.shape[0]):
        gs = torch.autograd.grad(
            losses[i],
            params,
            retain_graph=(i < losses.shape[0] - 1),
        )
        g_i = torch.cat([g.flatten() for g in gs])
        if sigma > 0:
            g_i = g_i + sigma * torch.randn_like(g_i)
        rows.append(g_i)
    return torch.stack(rows)
