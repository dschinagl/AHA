import torch
from torch import nn

from aha.pipeline.interventions import build_intervention_images


def build_percentiles(count: int, include_endpoints: bool, device: torch.device) -> torch.Tensor:
    if include_endpoints:
        return torch.linspace(0.0, 1.0, steps=count, device=device)
    return torch.linspace(0.0, 1.0, steps=count + 2, device=device)[1:-1]


class AUCLoss:
    """Trapezoidal area under the (percentile, target-class confidence) curve"""

    def __init__(
        self,
        base_model: nn.Module,
        percentiles: torch.Tensor,
        smoothing_sigma: float | None = None,
        smoothing_kernel_size: int | None = None,
        smoothing_padding_mode: str = "reflect",
    ) -> None:
        self.base_model = base_model
        self.percentiles = percentiles
        self.smoothing_sigma = smoothing_sigma
        self.smoothing_kernel_size = smoothing_kernel_size
        self.smoothing_padding_mode = smoothing_padding_mode

    def compute(
        self,
        permutation: torch.Tensor,
        grid_size: int,
        images: torch.Tensor,
        reference_images: torch.Tensor,
        target_idx: torch.Tensor,
        shift: tuple[int, int] = (0, 0),
    ) -> torch.Tensor:
        intervention_images = build_intervention_images(
            permutation,
            grid_size,
            images,
            reference_images,
            self.percentiles,
            self.smoothing_sigma,
            self.smoothing_kernel_size,
            self.smoothing_padding_mode,
            shift,
        )

        batch = images.shape[0]
        num_steps = self.percentiles.numel()
        scores = self.base_model(intervention_images).view(batch, num_steps, -1)
        confidence = scores.gather(2, target_idx.view(batch, 1, 1).expand(batch, num_steps, 1)).squeeze(2)

        percentiles = self.percentiles.to(confidence.dtype).unsqueeze(0).expand(batch, -1)
        return torch.trapezoid(confidence, percentiles, dim=1).mean()
