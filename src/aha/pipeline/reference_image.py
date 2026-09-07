import random

import torch
import torch.nn.functional as F

from aha.pipeline.gaussian import gaussian_kernel_2d

_CONCRETE_MODES = ("black", "mean", "blur")
_CONSTANT_COLORS = {
    "black": (0.0, 0.0, 0.0),
    "mean": (0.485, 0.456, 0.406),
}


def build_reference_image(
    images: torch.Tensor, mode: str, blur_kernel_size: int = 37, blur_sigma: float = 37.0
) -> torch.Tensor:
    """The reference image interventions blend towards (solid black, solid mean of
    the dataset, a heavily blurred version of the original image, or picks one of
    the three at random per call (mode = "all")."""
    if mode == "all":
        mode = random.choice(_CONCRETE_MODES)

    if mode in _CONSTANT_COLORS:
        color = images.new_tensor(_CONSTANT_COLORS[mode]).view(1, 3, 1, 1)
        return color.expand_as(images).contiguous()

    if mode == "blur":
        kernel_size = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        kernel = gaussian_kernel_2d(kernel_size, blur_sigma).to(images.device, images.dtype)
        kernel = kernel.view(1, 1, kernel_size, kernel_size).expand(images.shape[1], 1, kernel_size, kernel_size)
        pad = kernel_size // 2
        padded = F.pad(images, (pad, pad, pad, pad), mode="reflect")
        return F.conv2d(padded, kernel, groups=images.shape[1])

    raise ValueError(f"Unsupported reference image mode: {mode!r}")
