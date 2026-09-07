import random

import numpy as np
import torch


def set_seed(seed: int, rank: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
