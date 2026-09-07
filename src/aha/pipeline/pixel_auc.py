import torch
from torch import nn

from aha.pipeline.hard_rank import hard_rank_masks
from aha.pipeline.interventions import blend_by_mask


class PixelAUCMetric:
    """Trapezoidal area under the (percentile, target-class confidence) curve.
    pass `heatmap` for deletion, `-heatmap` for insertion.
    """

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
        heatmap: torch.Tensor,
        images: torch.Tensor,
        reference_images: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, height, width = images.shape
        scores = heatmap.reshape(batch, -1)
        masks = hard_rank_masks(scores, self.percentiles)
        num_steps = masks.shape[1]

        intervention_images = blend_by_mask(
            base=images.repeat_interleave(num_steps, dim=0),
            source=reference_images.repeat_interleave(num_steps, dim=0),
            mask_flat=masks.reshape(batch * num_steps, -1),
            grid_size=height,
            smoothing_sigma=self.smoothing_sigma,
            smoothing_kernel_size=self.smoothing_kernel_size,
            smoothing_padding_mode=self.smoothing_padding_mode,
        )

        scores_out = self.base_model(intervention_images).view(batch, num_steps, -1)
        confidence = scores_out.gather(2, target_idx.view(batch, 1, 1).expand(batch, num_steps, 1)).squeeze(2)

        percentiles = self.percentiles.to(confidence.dtype).unsqueeze(0).expand(batch, -1)
        return torch.trapezoid(confidence, percentiles, dim=1).mean()
