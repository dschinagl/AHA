import torch
from torch import nn

def register_contiguous_grad_hooks(module: nn.Module) -> None:
    """ DDP optimization: ensure gradients are contiguous in memory."""
    for param in module.parameters():
        if not param.requires_grad:
            continue
        ref_stride = tuple(param.stride())

        def _hook(grad: torch.Tensor | None, ref_stride: tuple[int, ...] = ref_stride) -> torch.Tensor | None:
            if grad is None or tuple(grad.stride()) == ref_stride:
                return grad
            if grad.dim() == 4:
                return grad.clone(memory_format=torch.contiguous_format)
            return grad.contiguous()

        param.register_hook(_hook)
