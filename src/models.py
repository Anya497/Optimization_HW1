import torch
from torch import nn


class SemicircleNet(nn.Module):
    """t(a) = w2.T tanh(W1 a + b1) + b2: 33 parameters, raw logits."""

    def __init__(self, seed: int = 0):
        super().__init__()
        # Reproducible initialization without changing the caller's RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.hidden = nn.Linear(2, 8)
            self.output = nn.Linear(8, 1)
            nn.init.xavier_uniform_(self.hidden.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.xavier_uniform_(self.output.weight)
            nn.init.zeros_(self.hidden.bias)
            nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(torch.tanh(self.hidden(x))).squeeze(-1)


class LogReg(nn.Module):
    def __init__(self, seed: int = 0):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.lin = nn.Linear(2, 1, bias=True)
            nn.init.xavier_uniform_(self.lin.weight, gain=nn.init.calculate_gain("tanh"))
            nn.init.zeros_(self.lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).squeeze(-1)
