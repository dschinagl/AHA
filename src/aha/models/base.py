from abc import ABC, abstractmethod
from contextlib import nullcontext

import torch
from torch import nn
from torchvision import transforms as T


class ClassifierBase(ABC, nn.Module):
    """Shared bfloat16-autocast/normalize/softmax forward path for target classifiers.

    Subclasses must build their `nn.Module` and pass it to `__init__`; images
    reach `forward` in raw [0, 1] range (normalization happens here, not in the
    data pipeline) so intervention code can blend against reference images.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self._use_bfloat16 = False

    @abstractmethod
    def _normalize_transform(self) -> T.Compose: ...

    @abstractmethod
    def spatial_transform(self) -> T.Compose: ...

    @abstractmethod
    def input_hw(self) -> tuple[int, int]: ...

    @abstractmethod
    def num_classes(self) -> int: ...

    def set_bfloat16(self, enabled: bool) -> None:
        self._use_bfloat16 = enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize_transform()(x)

        use_bf16 = (
            self._use_bfloat16
            and x.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()

        with ctx:
            logits = self.model(x)
            prob = logits.softmax(dim=1)

        return prob.to(torch.float32) if use_bf16 else prob
