import torch
from torch import nn


def _sample_gumbel_noise(
    shape: torch.Size, device: torch.device, dtype: torch.dtype, eps: float = 1e-20
) -> torch.Tensor:
    uniform = torch.rand(shape, device=device, dtype=dtype).clamp_(eps, 1.0 - eps)
    return -torch.log(-torch.log(uniform))


def _sinkhorn_normalize(log_alpha: torch.Tensor, n_iters: int) -> torch.Tensor:
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
    return log_alpha.exp()


class GumbelSinkhorn(nn.Module):
    """Differentiable sorting (Mena et al., https://arxiv.org/abs/1802.08665): maps
    per-entry attribution scores to a soft, approximately doubly-stochastic
    permutation matrix, via Gumbel noise for stochasticity and temperature-annealed
    Sinkhorn normalization for differentiability."""

    def __init__(self, sinkhorn_iters: int, tau: float, gumbel_on: bool) -> None:
        super().__init__()
        self.sinkhorn_iters = sinkhorn_iters
        self.tau = tau
        self.gumbel_on = gumbel_on

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        scores = scores.float()
        _, d = scores.shape
        device, dtype = scores.device, scores.dtype

        anchors = torch.linspace(1.0, 0.0, steps=d, device=device).view(1, d, 1)
        log_alpha = -(scores.unsqueeze(1) - anchors).pow(2)

        if self.gumbel_on:
            log_alpha = log_alpha + _sample_gumbel_noise(log_alpha.shape, device, dtype)

        log_alpha = log_alpha / max(self.tau, 1e-6)
        return _sinkhorn_normalize(log_alpha, self.sinkhorn_iters)
