import torch


def hard_rank_masks(scores: torch.Tensor, percentiles: torch.Tensor) -> torch.Tensor:
    """For each percentile p, a mask over the top-p fraction of entries by score
    (rank 0 = highest score).
    """
    batch, num_cells = scores.shape
    order = torch.argsort(scores, dim=1, descending=True)
    rank = torch.empty_like(order).scatter_(
        1, order, torch.arange(num_cells, device=scores.device).expand(batch, num_cells)
    )
    thresholds = (percentiles * num_cells).round().long()
    return (rank.unsqueeze(1) < thresholds.view(1, -1, 1)).to(scores.dtype)
