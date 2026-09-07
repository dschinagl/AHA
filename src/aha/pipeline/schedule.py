def linear_anneal(progress: float, start: float, end: float) -> float:
    progress = min(max(progress, 0.0), 1.0)
    return (1 - progress) * start + progress * end
