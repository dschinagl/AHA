import timm
from timm.data import ImageNetInfo, infer_imagenet_subset
from torchvision import transforms as T

from aha.models.base import ClassifierBase


class VitBasePatch16_224(ClassifierBase):
    def __init__(self, pretrained_tag: str) -> None:
        model = timm.create_model(pretrained_tag, pretrained=True)
        if not hasattr(model, "set_grad_checkpointing"):
            raise AttributeError("The model does not have the method 'set_grad_checkpointing'")
        model.set_grad_checkpointing(True)

        mean = model.pretrained_cfg["mean"]
        std = model.pretrained_cfg["std"]

        super().__init__(model=model)
        self._normalize = T.Compose([T.Normalize(mean=mean, std=std)])

    def _normalize_transform(self) -> T.Compose:
        return self._normalize

    def spatial_transform(self) -> T.Compose:
        return T.Compose([T.Resize((224, 224))])

    def input_hw(self) -> tuple[int, int]:
        return (224, 224)

    def num_classes(self) -> int:
        return self.model.head.out_features

    def class_names(self) -> list[str]:
        info = ImageNetInfo(infer_imagenet_subset(self.model))
        return [info.index_to_description(i).split(",")[0] for i in range(info.num_classes())]
