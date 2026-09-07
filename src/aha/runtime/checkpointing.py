import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

_BACKBONE_PREFIX = "encoder.backbone."


def _trainable_state_dict(model: nn.Module) -> dict:
    return {k: v for k, v in model.state_dict().items() if not k.startswith(_BACKBONE_PREFIX)}


def _load_trainable_state_dict(model: nn.Module, state_dict: dict) -> None:
    result = model.load_state_dict(state_dict, strict=False)
    unexpected_missing = [k for k in result.missing_keys if not k.startswith(_BACKBONE_PREFIX)]
    if unexpected_missing or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch — missing: {unexpected_missing}, unexpected: {result.unexpected_keys}"
        )


def save_trainable_state(model: nn.Module, path: str) -> None:
    """Just the trainable weights — e.g. for exporting a final model."""
    torch.save(_trainable_state_dict(model), path)


def load_trainable_state(model: nn.Module, path: str) -> None:
    _load_trainable_state_dict(model, torch.load(path, map_location="cpu"))


def save_checkpoint(path: str, model: nn.Module, optimizer: Optimizer, scheduler: LRScheduler, global_step: int) -> None:
    """Full training state — for resuming an interrupted run."""
    torch.save(
        {
            "model": _trainable_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "global_step": global_step,
        },
        path,
    )


def load_checkpoint(path: str, model: nn.Module, optimizer: Optimizer, scheduler: LRScheduler) -> int:
    """Restores model/optimizer/scheduler state in place, returns the saved `global_step`."""
    checkpoint = torch.load(path, map_location="cpu")
    _load_trainable_state_dict(model, checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    return checkpoint["global_step"]
