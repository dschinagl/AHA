from datetime import datetime
from itertools import islice
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm

from aha.config import Config, register_configs
from aha.pipeline.auc_loss import AUCLoss, build_percentiles
from aha.pipeline.grid import pool_heatmap_to_grid, sample_grid_size
from aha.pipeline.grid_shift import sample_grid_shift, shift_grid
from aha.pipeline.lr_schedule import LinearWarmupCosineDecayLR
from aha.pipeline.metric_tracker import MetricTracker
from aha.pipeline.peak_penalty import peak_suppression_penalty
from aha.pipeline.reference_image import build_reference_image
from aha.pipeline.schedule import linear_anneal
from aha.pipeline.targets import resolve_target_classes
from aha.runtime.checkpointing import load_checkpoint, save_checkpoint, save_trainable_state
from aha.runtime.distributed import DistributedContext
from aha.runtime.grad_hooks import register_contiguous_grad_hooks
from aha.runtime.seed import set_seed

register_configs()


@hydra.main(config_path="../configs", config_name="train", execution_whitelist=("aha.*",))
def main(cfg: Config) -> None:
    ctx = DistributedContext.init()
    set_seed(cfg.seed, ctx.rank)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    output_dir = Path(cfg.training.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = None
    if ctx.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(OmegaConf.to_yaml(cfg))
        OmegaConf.save(cfg, output_dir / "config.yaml")
        writer = SummaryWriter(log_dir=str(output_dir))

    # No DDP wrap: every parameter is frozen in the classifier
    classifier = hydra.utils.instantiate(cfg.base_model).to(ctx.device)
    classifier.eval()
    classifier.requires_grad_(False)
    classifier.set_bfloat16(True)
    spatial_transform = classifier.spatial_transform()

    explainer = hydra.utils.instantiate(cfg.explainer).to(ctx.device)
    explainer.train()
    register_contiguous_grad_hooks(explainer)
    if ctx.is_distributed:
        explainer = DistributedDataParallel(explainer, device_ids=[ctx.local_rank])

    sorter = hydra.utils.instantiate(cfg.sorter).to(ctx.device)
    sorter.train()

    percentiles = build_percentiles(cfg.loss.mask_sampling_count, cfg.loss.mask_sampling_include_endpoints, ctx.device)
    auc_loss = AUCLoss(
        classifier,
        percentiles,
        cfg.method.mask_smoothing_sigma,
        cfg.method.mask_smoothing_kernel_size,
        cfg.method.mask_smoothing_padding_mode,
    )

    transform = T.Compose([spatial_transform, T.ToTensor()])
    dataset = hydra.utils.instantiate(cfg.dataset, split="train", transform=transform)

    sampler = DistributedSampler(dataset, shuffle=True, drop_last=True) if ctx.is_distributed else None
    loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * cfg.training.epochs

    trainable_explainer_params = explainer.module if ctx.is_distributed else explainer
    optimizer = torch.optim.AdamW(
        trainable_explainer_params.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.optimizer.weight_decay
    )
    scheduler = LinearWarmupCosineDecayLR(
        optimizer,
        total_steps,
        cfg.training.learning_rate,
        cfg.scheduler.pct_start,
        cfg.scheduler.div_factor,
        cfg.scheduler.final_div_factor,
    )
    tracker = MetricTracker(cfg.training.log_interval_steps)

    global_step = 0
    if cfg.training.resume_from is not None:
        global_step = load_checkpoint(cfg.training.resume_from, trainable_explainer_params, optimizer, scheduler)
    start_epoch = global_step // steps_per_epoch
    skipped_batches = global_step % steps_per_epoch
    samples_seen = global_step * cfg.training.batch_size * ctx.world_size

    for epoch in range(start_epoch, cfg.training.epochs):
        if ctx.is_distributed:
            sampler.set_epoch(epoch)

        epoch_batches = islice(loader, skipped_batches, None) if skipped_batches else loader
        progress_bar = tqdm(
            epoch_batches,
            total=steps_per_epoch - skipped_batches,
            disable=not ctx.is_main,
            desc=f"epoch {epoch}",
            unit="step",
        )
        skipped_batches = 0
        for images, labels in progress_bar:
            images = images.to(ctx.device, non_blocking=True)
            labels = labels.to(ctx.device, non_blocking=True)

            target_idx, secondary_target_idx = resolve_target_classes(
                classifier, images, labels, cfg.method.cls_to_optimize
            )
            grid_size = sample_grid_size(cfg.method.grid_size_min, cfg.method.grid_size_max, cfg.method.grid_size_step)

            heatmap = explainer(images, target_idx)
            grid_shift = (0, 0)
            if cfg.method.grid_shift_max_frac:
                grid_shift = sample_grid_shift(heatmap.shape[-2], grid_size, cfg.method.grid_shift_max_frac)
            heatmap_grid = pool_heatmap_to_grid(shift_grid(heatmap, *grid_shift), grid_size)
            if not torch.isfinite(heatmap_grid).all():
                raise RuntimeError(f"non-finite heatmap_grid at step {global_step}")

            reference_images = build_reference_image(
                images, cfg.method.ref_im, cfg.method.ref_im_blur_kernel_size, cfg.method.ref_im_blur_sigma
            )

            if cfg.method.tau_anneal:
                sorter.tau = linear_anneal(global_step / max(total_steps - 1, 1), cfg.sorter.tau, cfg.method.tau_final)

            loss_terms = {}
            raw_metrics = {}

            if cfg.loss.desc_auc:
                permutation_desc = sorter(heatmap_grid)
                if not torch.isfinite(permutation_desc).all():
                    raise RuntimeError(f"non-finite permutation_desc at step {global_step}")
                desc_aucs = [
                    auc_loss.compute(permutation_desc, grid_size, images, reference_images, target_idx, grid_shift)
                ]
                if secondary_target_idx is not None:
                    desc_aucs.append(
                        auc_loss.compute(
                            permutation_desc, grid_size, images, reference_images, secondary_target_idx, grid_shift
                        )
                    )
                auc_desc = torch.stack(desc_aucs).mean()
                raw_metrics["auc_desc"] = auc_desc
                loss_terms["desc_auc"] = cfg.loss.desc_auc_weight * auc_desc

            if cfg.loss.asc_auc:
                # Deleting least-important cells first, then reading the curve backwards, is
                # equivalent to an insertion metric
                permutation_asc = sorter(-heatmap_grid)
                if not torch.isfinite(permutation_asc).all():
                    raise RuntimeError(f"non-finite permutation_asc at step {global_step}")
                asc_aucs = [
                    auc_loss.compute(permutation_asc, grid_size, images, reference_images, target_idx, grid_shift)
                ]
                if secondary_target_idx is not None:
                    asc_aucs.append(
                        auc_loss.compute(
                            permutation_asc, grid_size, images, reference_images, secondary_target_idx, grid_shift
                        )
                    )
                auc_asc = torch.stack(asc_aucs).mean()
                raw_metrics["auc_asc"] = auc_asc
                loss_terms["asc_auc"] = cfg.loss.asc_auc_weight * (1.0 - auc_asc)

            if cfg.loss.peak_penalty_weight != 0.0:
                peak_penalty = peak_suppression_penalty(
                    heatmap, cfg.loss.peak_penalty_kernel_size, cfg.loss.peak_penalty_relative, cfg.loss.peak_penalty_eps
                )
                raw_metrics["peak"] = peak_penalty
                loss_terms["peak_penalty"] = cfg.loss.peak_penalty_weight * peak_penalty

            loss = sum(loss_terms.values())

            optimizer.zero_grad()
            loss.backward()
            if cfg.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable_explainer_params.parameters(), cfg.training.max_grad_norm)
            optimizer.step()
            scheduler.step()

            global_step += 1
            samples_seen += cfg.training.batch_size * ctx.world_size

            names = [*raw_metrics.keys(), "loss_total"]
            reduced = ctx.reduce_mean(torch.stack([*raw_metrics.values(), loss.detach()]))
            reduced = dict(zip(names, reduced.tolist()))

            if ctx.is_main:
                metrics = {"loss/total": reduced["loss_total"], "lr": scheduler.get_last_lr()[0]}
                if "auc_desc" in reduced:
                    metrics["metrics/auc_desc"] = reduced["auc_desc"]
                    metrics["loss/desc"] = cfg.loss.desc_auc_weight * reduced["auc_desc"]
                if "auc_asc" in reduced:
                    metrics["metrics/auc_asc"] = reduced["auc_asc"]
                    metrics["loss/asc"] = cfg.loss.asc_auc_weight * (1.0 - reduced["auc_asc"])
                if "peak" in reduced:
                    metrics["loss/peak"] = cfg.loss.peak_penalty_weight * reduced["peak"]

                progress_bar.set_postfix({k: f"{v:.4f}" for k, v in metrics.items()})
                tracker.update(**metrics)

                if tracker.should_flush(is_last=(global_step == total_steps)):
                    for name, value in tracker.flush().items():
                        writer.add_scalar(name, value, global_step)
                    writer.add_scalar("train/samples_seen", samples_seen, global_step)

                if global_step % cfg.training.checkpoint_interval_steps == 0:
                    save_checkpoint(
                        str(output_dir / "checkpoint_latest.pt"),
                        trainable_explainer_params,
                        optimizer,
                        scheduler,
                        global_step,
                    )

    if ctx.is_main:
        save_checkpoint(
            str(output_dir / "checkpoint_final.pt"), trainable_explainer_params, optimizer, scheduler, global_step
        )
        save_trainable_state(trainable_explainer_params, str(output_dir / "weights_final.pt"))
        writer.close()

    ctx.destroy()


if __name__ == "__main__":
    main()
