import random

import torch
from torch import nn


def sample_grid_size(min_size: int, max_size: int, step: int) -> int:
    return random.choice(range(min_size, max_size + 1, step))


def pool_heatmap_to_grid(heatmap: torch.Tensor, grid_size: int) -> torch.Tensor:
    pooled = nn.functional.adaptive_avg_pool2d(heatmap, output_size=grid_size)
    return pooled.view(pooled.shape[0], -1)
