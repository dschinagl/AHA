import torch
import torch.nn.functional as F


def peak_suppression_penalty(heatmap: torch.Tensor, kernel_size: int, relative: bool, eps: float) -> torch.Tensor:
    """Penalizes sharp, localized peaks by comparing each pixel to its local
    average (discourages the explainer from collapsing onto a single pixel)"""
    heatmap = heatmap.float()
    pad = kernel_size // 2
    padded = F.pad(heatmap, (pad, pad, pad, pad), mode="reflect")
    local_mean = F.avg_pool2d(padded, kernel_size, stride=1)
    residual = torch.relu(heatmap - local_mean)

    if relative:
        residual = residual / local_mean.abs().clamp(min=eps)

    return residual.pow(2).mean()
