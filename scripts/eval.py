import argparse
from datetime import datetime
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from tqdm import tqdm

from aha.config import Config
from aha.pipeline.auc_loss import build_percentiles
from aha.pipeline.pixel_auc import PixelAUCMetric
from aha.pipeline.reference_image import build_reference_image
from aha.pipeline.targets import resolve_target_classes
from aha.pipeline.visualize import save_input_heatmap_overlay
from aha.runtime.checkpointing import load_trainable_state
from aha.runtime.distributed import DistributedContext
from aha.runtime.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts_dir", type=Path)
    parser.add_argument("--checkpoint", default="weights_final.pt")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    saved_cfg = OmegaConf.load(args.artifacts_dir / "config.yaml")
    cfg: Config = OmegaConf.merge(OmegaConf.structured(Config), saved_cfg, OmegaConf.from_dotlist(args.overrides))

    ctx = DistributedContext.init()
    set_seed(cfg.seed)

    checkpoint_path = args.artifacts_dir / args.checkpoint
    if ctx.is_main:
        print(f"explainer weights : {checkpoint_path} ({cfg.explainer._target_}, backbone={cfg.explainer.backbone_name})")
        print(f"classifier under test : {cfg.base_model._target_} ({cfg.base_model.pretrained_tag})")
        print(f"dataset : {cfg.dataset._target_} root={cfg.dataset.root}")
        print(f"eval config:\n{OmegaConf.to_yaml(cfg.eval)}")

    classifier = hydra.utils.instantiate(cfg.base_model, _execution_whitelist_=("aha.*",)).to(ctx.device)
    classifier.eval()
    classifier.requires_grad_(False)
    spatial_transform = classifier.spatial_transform()
    classes = classifier.class_names()

    explainer = hydra.utils.instantiate(cfg.explainer, _execution_whitelist_=("aha.*",)).to(ctx.device)
    load_trainable_state(explainer, str(checkpoint_path))
    explainer.eval()
    explainer.requires_grad_(False)

    percentile_count = round(1.0 / cfg.eval.percentile_step_pct) + 1
    percentiles = build_percentiles(percentile_count, include_endpoints=True, device=ctx.device)
    pixel_auc = PixelAUCMetric(classifier, percentiles)

    transform = T.Compose([spatial_transform, T.ToTensor()])
    dataset = hydra.utils.instantiate(
        cfg.dataset, split=cfg.eval.split, transform=transform, _execution_whitelist_=("aha.*",)
    )

    full_dataset = dataset
    if ctx.is_distributed:
        dataset = Subset(dataset, range(ctx.rank, len(dataset), ctx.world_size))
    loader = DataLoader(
        dataset, batch_size=cfg.eval.batch_size, shuffle=False,
        num_workers=cfg.eval.num_workers, pin_memory=True, drop_last=False,
    )

    total_desc = total_asc = total_count = 0.0
    progress_bar = tqdm(loader, disable=not ctx.is_main, desc="eval", unit="batch")

    with torch.no_grad():
        for images, labels in progress_bar:
            images = images.to(ctx.device, non_blocking=True)
            labels = labels.to(ctx.device, non_blocking=True)
            batch = images.shape[0]

            target_idx, _ = resolve_target_classes(classifier, images, labels, cfg.eval.cls_to_eval)
            heatmap = explainer(images, target_idx)
            reference_images = build_reference_image(
                images, cfg.eval.ref_im, cfg.method.ref_im_blur_kernel_size, cfg.method.ref_im_blur_sigma
            )

            auc_desc = pixel_auc.compute(heatmap, images, reference_images, target_idx)
            auc_asc = pixel_auc.compute(-heatmap, images, reference_images, target_idx)

            total_desc += auc_desc.item() * batch
            total_asc += auc_asc.item() * batch
            total_count += batch
            progress_bar.set_postfix(auc_desc=f"{total_desc / total_count:.4f}", auc_asc=f"{total_asc / total_count:.4f}")

    totals = ctx.reduce_sum(torch.tensor([total_desc, total_asc, total_count], device=ctx.device))
    total_desc, total_asc, total_count = totals.tolist()

    if ctx.is_main:
        metrics = {"auc_desc": total_desc / total_count, "auc_asc": total_asc / total_count}
        print(metrics)

        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        record = {
            "timestamp": run_timestamp,
            "checkpoint": args.checkpoint,
            "explainer": cfg.explainer._target_,
            "backbone": cfg.explainer.backbone_name,
            "classifier": cfg.base_model._target_,
            "classifier_pretrained_tag": cfg.base_model.pretrained_tag,
            "dataset": cfg.dataset._target_,
            "split": cfg.eval.split,
            "cls_to_eval": cfg.eval.cls_to_eval,
            "ref_im": cfg.eval.ref_im,
            "percentile_step_pct": cfg.eval.percentile_step_pct,
            "batch_size": cfg.eval.batch_size,
            "metrics": metrics,
        }

        metrics_path = args.artifacts_dir / "eval_metrics.yaml"
        history = OmegaConf.load(metrics_path) if metrics_path.exists() else OmegaConf.create([])
        history.append(record)
        OmegaConf.save(history, metrics_path)

        num_viz_samples = min(50, len(full_dataset))
        viz_indices = torch.randperm(len(full_dataset), generator=torch.Generator().manual_seed(cfg.seed))
        viz_indices = viz_indices[:num_viz_samples].tolist()
        viz_dir = args.artifacts_dir / f"eval_{run_timestamp}_visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            for i, idx in enumerate(viz_indices):
                image, label = full_dataset[idx]
                image = image.unsqueeze(0).to(ctx.device)
                target_idx, _ = resolve_target_classes(
                    classifier, image, torch.tensor([label], device=ctx.device), cfg.eval.cls_to_eval
                )
                heatmap = explainer(image, target_idx)

                if target_idx.item() == label:
                    title, color = f"class: {classes[label]}", "green"
                else:
                    title, color = (
                        f"predicted: {classes[target_idx.item()]} (ground truth: {classes[label]})",
                        "red",
                    )

                save_input_heatmap_overlay(image[0], heatmap[0, 0], viz_dir / f"{i:02d}.jpg", title, color)

    ctx.destroy()


if __name__ == "__main__":
    main()
