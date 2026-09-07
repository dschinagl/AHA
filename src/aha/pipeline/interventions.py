import math

import torch
import torch.nn.functional as F

from aha.pipeline.gaussian import gaussian_kernel_2d
from aha.pipeline.grid_shift import shift_grid


def _smooth_mask(mask: torch.Tensor, sigma: float, kernel_size: int | None, padding_mode: str) -> torch.Tensor:
    if kernel_size is None:
        kernel_size = 2 * math.ceil(3.0 * sigma) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = gaussian_kernel_2d(kernel_size, sigma).to(mask.device, mask.dtype).view(1, 1, kernel_size, kernel_size)
    pad = kernel_size // 2
    mask = F.pad(mask, (pad, pad, pad, pad), mode=padding_mode)
    return F.conv2d(mask, kernel).clamp_(0.0, 1.0)


def blend_by_mask(
    base: torch.Tensor,
    source: torch.Tensor,
    mask_flat: torch.Tensor,
    grid_size: int,
    smoothing_sigma: float | None = None,
    smoothing_kernel_size: int | None = None,
    smoothing_padding_mode: str = "reflect",
    shift: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """Blend base towards source, per pixel, using a per-grid-cell mask
    (0 = base, 1 = source) upsampled to image resolution.
    """
    batch, _, height, width = base.shape
    mask = mask_flat.clamp(0, 1).view(batch, 1, grid_size, grid_size).to(base.dtype)
    mask = F.interpolate(mask, size=(height, width), mode="nearest")
    mask = shift_grid(mask, -shift[0], -shift[1])

    if smoothing_sigma is not None:
        mask = _smooth_mask(mask, smoothing_sigma, smoothing_kernel_size, smoothing_padding_mode)

    return torch.lerp(base, source, mask)


def build_intervention_images(
    permutation: torch.Tensor,
    grid_size: int,
    images: torch.Tensor,
    reference_images: torch.Tensor,
    percentiles: torch.Tensor,
    smoothing_sigma: float | None = None,
    smoothing_kernel_size: int | None = None,
    smoothing_padding_mode: str = "reflect",
    shift: tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """A deletion sweep: for each requested percentile p (sorted ascending, [0, 1]),
    the top-p fraction of grid cells in permutations ranking are replaced by the
    reference_images.
    """
    batch, num_cells, _ = permutation.shape

    cumulative_delete = permutation.cumsum(dim=1)  # [B, num_cells, num_cells]
    zeros_row = cumulative_delete.new_zeros(batch, 1, num_cells)
    cumulative_delete = torch.cat([zeros_row, cumulative_delete], dim=1)  # [B, num_cells + 1, num_cells]

    step = percentiles * num_cells
    step_floor = step.floor().long()
    step_ceil = (step_floor + 1).clamp(max=num_cells)
    delete_at_floor = cumulative_delete.index_select(1, step_floor)
    delete_at_ceil = cumulative_delete.index_select(1, step_ceil)
    alpha = (step - step_floor).to(cumulative_delete.dtype).view(1, -1, 1)
    delete_mask = torch.lerp(delete_at_floor, delete_at_ceil, alpha)  # [B, num_steps, num_cells]

    num_steps = delete_mask.shape[1]
    return blend_by_mask(
        base=images.repeat_interleave(num_steps, dim=0),
        source=reference_images.repeat_interleave(num_steps, dim=0),
        mask_flat=delete_mask.reshape(batch * num_steps, num_cells),
        grid_size=grid_size,
        smoothing_sigma=smoothing_sigma,
        smoothing_kernel_size=smoothing_kernel_size,
        smoothing_padding_mode=smoothing_padding_mode,
        shift=shift,
    )
