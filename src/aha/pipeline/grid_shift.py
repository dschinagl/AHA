import random

import torch
import torch.nn.functional as F


def sample_grid_shift(height: int, grid_size: int, max_frac: float) -> tuple[int, int]:
    """Random pixel offset (dy, dx), up to max_frac of one grid cell's size."""
    max_shift = round((height // grid_size) * max_frac)
    if max_shift <= 0:
        return 0, 0
    return random.randint(-max_shift, max_shift), random.randint(-max_shift, max_shift)


def shift_grid(x: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """Shifts spatial content by (dy, dx) pixels, reflect-filling the edge."""
    if dy == 0 and dx == 0:
        return x
    return F.pad(x, (dx, -dx, dy, -dy), mode="reflect")
