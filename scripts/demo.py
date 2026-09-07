import argparse
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms as T

from aha.config import Config
from aha.pipeline.visualize import save_input_heatmap_overlay
from aha.runtime.checkpointing import load_trainable_state

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts_dir", type=Path)
    parser.add_argument("--checkpoint", default="weights_final.pt")
    parser.add_argument("--images", type=Path, default=Path("assets/demo"))
    parser.add_argument("--output", type=Path, default=Path("assets/demo/output"))
    args = parser.parse_args()

    cfg: Config = OmegaConf.merge(OmegaConf.structured(Config), OmegaConf.load(args.artifacts_dir / "config.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = args.artifacts_dir / args.checkpoint
    print(f"explainer weights : {checkpoint_path} ({cfg.explainer._target_}, backbone={cfg.explainer.backbone_name})")
    print(f"classifier under test : {cfg.base_model._target_} ({cfg.base_model.pretrained_tag})")
    print(f"images : {args.images}")

    classifier = hydra.utils.instantiate(cfg.base_model, _execution_whitelist_=("aha.*",)).to(device)
    classifier.eval()
    classifier.requires_grad_(False)
    spatial_transform = classifier.spatial_transform()

    explainer = hydra.utils.instantiate(cfg.explainer, _execution_whitelist_=("aha.*",)).to(device)
    load_trainable_state(explainer, str(checkpoint_path))
    explainer.eval()
    explainer.requires_grad_(False)

    classes = classifier.class_names()

    transform = T.Compose([spatial_transform, T.ToTensor()])
    image_paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in _IMAGE_SUFFIXES)
    args.output.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for path in image_paths:
            image = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            target_idx = classifier(image).argmax(dim=1)
            heatmap = explainer(image, target_idx)

            title = f"predicted: {classes[target_idx.item()]}"
            save_input_heatmap_overlay(image[0], heatmap[0, 0], args.output / f"{path.stem}.jpg", title)
            print(f"{path.name} -> {title}")


if __name__ == "__main__":
    main()
