from collections import defaultdict


class MetricTracker:
    """Accumulates scalar metrics, returns their average over flush_interval updates."""

    def __init__(self, flush_interval: int) -> None:
        self.flush_interval = flush_interval
        self._count = 0
        self._totals: dict[str, float] = defaultdict(float)

    def update(self, **metrics: float) -> None:
        self._count += 1
        for name, value in metrics.items():
            self._totals[name] += value

    def should_flush(self, is_last: bool = False) -> bool:
        return self._count >= self.flush_interval or is_last

    def flush(self) -> dict[str, float]:
        averages = {name: total / self._count for name, total in self._totals.items()}
        self._count = 0
        self._totals = defaultdict(float)
        return averages
