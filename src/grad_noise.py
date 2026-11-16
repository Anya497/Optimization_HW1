import torch
from torch import nn
from torch.func import functional_call, grad, vmap
from torch.nn import functional as F


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


def per_example_gradients(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor
) -> dict[nn.Parameter, torch.Tensor]:
    """Map each parameter to its gradients shaped (batch_size, *parameter.shape)."""
    parameters = dict(model.named_parameters())

    def example_loss(params, point, label):
        logit = functional_call(model, params, (point,))
        return F.binary_cross_entropy_with_logits(logit, label)

    gradients = vmap(grad(example_loss), in_dims=(None, 0, 0))(parameters, x, y)
    return {parameter: gradients[name].detach() for name, parameter in parameters.items()}
