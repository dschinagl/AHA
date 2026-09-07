from typing import Literal

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms as T
from torchvision.datasets import ImageNet


def build_imagenet(
    root: str,
    train_split: str,
    val_split: str,
    num_samples: int | None,
    subset_seed: int,
    split: Literal["train", "val"],
    transform: T.Compose,
) -> Dataset:
    actual_split = train_split if split == "train" else val_split
    dataset: Dataset = ImageNet(root=root, split=actual_split, transform=transform)

    if num_samples is not None:
        generator = torch.Generator().manual_seed(subset_seed)
        indices = torch.randperm(len(dataset), generator=generator)[:num_samples]
        dataset = Subset(dataset, indices.tolist())

    return dataset
