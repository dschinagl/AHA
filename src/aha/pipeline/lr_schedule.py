import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


def _linear(start: float, end: float, pct: float) -> float:
    return start + (end - start) * pct


def _cosine(start: float, end: float, pct: float) -> float:
    return end + (start - end) * 0.5 * (math.cos(math.pi * pct) + 1.0)


class LinearWarmupCosineDecayLR(LRScheduler):
    """Linear warmup to max_lr over the first pct_start of training, then
    cosine decay down to max_lr / div_factor / final_div_factor."""

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        max_lr: float,
        pct_start: float,
        div_factor: float,
        final_div_factor: float,
        last_epoch: int = -1,
    ) -> None:
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.max_lr = max_lr
        self.initial_lr = max_lr / div_factor
        self.min_lr = self.initial_lr / final_div_factor
        for group in optimizer.param_groups:
            group["lr"] = self.initial_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        step = max(0, self._step_count - 1)
        progress = min(step / max(self.total_steps - 1, 1), 1.0)

        if progress <= self.pct_start:
            lr = _linear(self.initial_lr, self.max_lr, progress / self.pct_start)
        else:
            pct = (progress - self.pct_start) / (1.0 - self.pct_start)
            lr = _cosine(self.max_lr, self.min_lr, pct)

        return [lr for _ in self.optimizer.param_groups]
