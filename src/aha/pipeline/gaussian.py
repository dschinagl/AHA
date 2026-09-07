import torch

_cache: dict[tuple[int, float], torch.Tensor] = {}


def gaussian_kernel_2d(kernel_size: int, sigma: float) -> torch.Tensor:
    """A normalized 2D Gaussian kernel, cached on CPU as float32."""
    key = (kernel_size, sigma)
    if key not in _cache:
        coords = torch.arange(kernel_size, dtype=torch.float64) - (kernel_size - 1) / 2.0
        kernel_1d = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel_2d = kernel_2d / kernel_2d.sum()
        _cache[key] = kernel_2d.float()
    return _cache[key]
