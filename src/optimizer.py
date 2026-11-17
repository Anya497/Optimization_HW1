import torch
from torch.optim import Optimizer


class RobustSGD(Optimizer):

    def __init__(
        self,
        params,
        lr=0.1,
        rho=1.0,
        eps=1e-12,
        max_iter=80      
    ):
        defaults = dict(
            lr=lr,
            rho=rho,
            eps=eps,
            max_iter=max_iter
        )

        super().__init__(params, defaults)
        self.zero_step = False

    @torch.no_grad()
    def step(self, closure=None, noisy_grads=None):
        self.zero_step = False
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if noisy_grads is None:
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.add_(p.grad, alpha=-group["lr"])

            return loss

        grads = []
        uncertainties = []
        learning_rates = []
        params = []

        rho_vals = set()
        eps_vals = set()
        max_iter_vals = set()

        for group in self.param_groups:
            lr = group["lr"]
            rho = group["rho"]
            eps = group["eps"]
            max_iter = group["max_iter"]

            rho_vals.add(rho)
            eps_vals.add(eps)
            max_iter_vals.add(max_iter)

            for p in group["params"]:

                if noisy_grads is None:
                    if p.grad is None:
                        continue

                    g = p.grad.detach()
                    S = torch.zeros_like(g)
                else:
                    if p not in noisy_grads:
                        if p.grad is None:
                            continue

                    per_sample_grads = noisy_grads[p].detach() # (B, *p.shape)
                    if per_sample_grads.shape[0] < 2:
                        raise ValueError(
                            "RobustSGD needs at least two gradient observations "
                            "to estimate uncertainty"
                        )

                    g = per_sample_grads.mean(dim=0)
                    centered = per_sample_grads - g.unsqueeze(0)
                    B = per_sample_grads.shape[0]
                    S = centered.square().sum(dim=0) / (B * (B - 1))

                params.append(p)
                grads.append(g.reshape(-1))
                uncertainties.append(S.reshape(-1))

                learning_rates.append(torch.full_like(g.reshape(-1), lr))

        if len(params) == 0:
            return loss

        rho = next(iter(rho_vals))
        eps = next(iter(eps_vals))
        max_iter = next(iter(max_iter_vals))

        g = torch.cat(grads)
        S = torch.cat(uncertainties)
        lr = torch.cat(learning_rates)
        epsilon = 1e-3 * S.mean().clamp_min(1e-8)
        S = S + epsilon
        # Report the robust zero-step condition, independent of rounding or the LR.
        self.zero_step = bool(torch.sum(g.square() / S) <= rho ** 2)

        if rho == 0:
            direction = -lr * g
        else:
            if self.zero_step:
                direction = torch.zeros_like(g)
            else:

                def equation(lam):
                    val = S * (lr * g).square()
                    denom = 1.0 + lam * lr * S
                    return torch.sum(val * (lam / denom).square()) - rho ** 2

                low = torch.zeros_like(rho_tensor := g.new_tensor(0.0))
                high = g.new_tensor(1.0)

                for _ in range(100):
                    if equation(high) >= 0:
                        break
                    high = high * 2.0

                for _ in range(max_iter):
                    mid = (low + high) / 2.0
                    if equation(mid) < 0:
                        low = mid
                    else:
                        high = mid

                lam = (low + high) / 2.0
                direction = (-lr * g / (1.0 + lam * lr * S))

        offset = 0
        for p in params:
            numel = p.numel()
            d = direction[offset:offset + numel]
            d = d.reshape_as(p)
            p.add_(d)
            offset += numel

        return loss
