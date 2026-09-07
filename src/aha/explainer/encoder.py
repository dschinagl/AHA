import math

import torch
from torch import nn


class CenterPadding(nn.Module):
    """Pads H/W up to the nearest multiple of `multiple`, centered."""

    def __init__(self, multiple: int) -> None:
        super().__init__()
        self.multiple = multiple

    def _pad(self, size: int) -> tuple[int, int]:
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad = new_size - size
        return pad // 2, pad - pad // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_w = self._pad(x.shape[-1])
        pad_h = self._pad(x.shape[-2])
        return nn.functional.pad(x, (*pad_w, *pad_h))


class Dinov3FeatureEncoder(nn.Module):
    """Frozen DINOv3 backbone exposing 4 intermediate-layer feature maps."""

    def __init__(self, backbone: nn.Module, out_layers: tuple[int, int, int, int]) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.out_layers = out_layers
        self.embed_dim = backbone.embed_dim
        self.pad = CenterPadding(backbone.patch_size)

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        x = self.pad(x)
        return self.backbone.get_intermediate_layers(
            x, n=self.out_layers, reshape=True, return_class_token=True, norm=True,
        )
