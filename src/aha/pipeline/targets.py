import torch
from torch import nn


def resolve_target_classes(
    base_model: nn.Module, images: torch.Tensor, labels: torch.Tensor, mode: str
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pick which class(es) the explainer should explain for this batch.

    `mode="prediction_and_ground_truth"` explains two classes per image: the
    model's top-1 prediction, and either top-2 or the ground truth, depending
    on whether the top-1 prediction is correct.
    """
    if mode == "ground_truth":
        return labels.view(-1), None

    with torch.no_grad():
        logits = base_model(images)
    top1 = logits.argmax(dim=1)

    if mode == "prediction":
        return top1, None

    if mode == "prediction_and_ground_truth":
        gt = labels.view(-1).long()
        runner_up = logits.topk(k=2, dim=1).indices[:, 1]
        secondary = torch.where(top1 == gt, runner_up, gt)
        return top1, secondary

    raise ValueError(f"Unknown cls_to_optimize mode: {mode!r}")
