import torch
from torch import nn
from torch.func import functional_call, grad, vmap
from torch.nn import functional as F


def per_example_gradients(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor,
    sigma: float = 0.0, generator: torch.Generator | None = None,
) -> dict[nn.Parameter, torch.Tensor]:
    """Return per-example gradients with optional independent Gaussian noise."""
    parameters = dict(model.named_parameters())

    def example_loss(params, point, label):
        logit = functional_call(model, params, (point,)).reshape(())
        return F.binary_cross_entropy_with_logits(logit, label.reshape(()))

    gradients = vmap(grad(example_loss), in_dims=(None, 0, 0))(parameters, x, y)
    observations = {}
    for name, parameter in parameters.items():
        values = gradients[name].detach()
        if sigma > 0:
            values = values + sigma * torch.randn(
                values.shape, generator=generator, dtype=values.dtype, device=values.device
            )
        observations[parameter] = values
    return observations
