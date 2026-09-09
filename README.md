# Learn to Rank: Visual Attribution by Learning Importance Ranking

[David Schinagl](https://dschinagl.github.io/), [Christian Fruhwirth-Reisinger](https://github.com/chreisinger), [Alexander Prutsch](https://a-pru.github.io/), [Samuel Schulter](https://samschulter.github.io/), and [Horst Possegger](https://snototter.github.io/)

Official code for the ECCV 2026 paper **"Learn to Rank: Visual Attribution by Learning Importance Ranking"** (AHA — Amortized Hybrid Attribution).

[![arXiv paper](https://img.shields.io/badge/arXiv-2604.05819-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2604.05819)
[![YouTube video](https://img.shields.io/badge/YouTube-Video-FF0000?logo=youtube&logoColor=white)](https://www.youtube.com/watch?v=Lq7KKjG-kus)

An explainer network is trained to predict, for a frozen target classifier, a dense attribution map whose ranking directly optimizes the (otherwise non-differentiable) Deletion and Insertion metrics via a Gumbel-Sinkhorn relaxation of sorting. At inference, the explainer produces a pixel-level attribution map in a single forward pass.

<p align="left">
  <a href="https://www.youtube.com/watch?v=Lq7KKjG-kus">
    <img src="https://img.youtube.com/vi/Lq7KKjG-kus/hqdefault.jpg" alt="Learn to Rank — watch the video on YouTube" width="480">
  </a>
</p>
<p align="left">
  <a href="https://www.youtube.com/watch?v=Lq7KKjG-kus">▶ Watch the video on YouTube</a>
</p>

## Repository structure

```
configs/train.yaml          Hydra config (overrides on top of the defaults in src/aha/config.py)
scripts/train.py            Training entry point
scripts/eval.py             Evaluation entry point (deletion/insertion AUC on a trained checkpoint)
scripts/demo.py             Runs the explainer on a handful of standalone images
pretrained_models/          Released explainer weights, one folder per target classifier
assets/demo/                Example images for the demo script
src/aha/
  config.py                 Structured config schema (Hydra ConfigStore)
  data/                     ImageNet dataset construction
  models/                   Frozen target classifier wrapper
  explainer/                DINOv3 encoder + DPT decoder producing the attribution map
  sorter/                   Gumbel-Sinkhorn differentiable sorting
  pipeline/                 Shared building blocks: interventions, AUC loss, LR schedule, metrics, ...
  runtime/                  Distributed setup, checkpointing, seeding
```

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14 (pinned via `.python-version`).

```bash
uv sync
```

**DINOv3 backbone.** The explainer uses a frozen DINOv3 ViT-L/16 backbone, provided via `torch.hub` from a pinned commit:

```bash
scripts/setup_dinov3.sh   # clones facebookresearch/dinov3 into ./dinov3
```

The pretrained DINOv3 weights themselves are gated by Meta's license — request access and download `dinov3_vitl16_pretrain_lvd1689m-*.pth` following the instructions in the [DINOv3 repo](https://github.com/facebookresearch/dinov3), then point `explainer.backbone_weights` at the downloaded file (see `configs/train.yaml`).

**ImageNet.** Training and evaluation use the standard ImageNet-1k `train`/`val` splits (`torchvision.datasets.ImageNet`, requires the devkit alongside the image folders). Set `dataset.root` in `configs/train.yaml` to your local copy.

The target classifier (`timm`'s `vit_base_patch16_224.orig_in21k_ft_in1k`) is downloaded automatically from the Hugging Face Hub on first use.

**Mandatory config values.** Fields marked `???` (`dataset.root` and `explainer.backbone_weights`) must be filled in, either by editing the config file, or by passing them on the command line as shown below.

## Training

```bash
uv run torchrun --nproc_per_node=<num_gpus> scripts/train.py \
  dataset.root=/path/to/imagenet \
  explainer.backbone_weights=/path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

Both paths can equally well be written into `configs/train.yaml`. Every other config value (see `src/aha/config.py`) is overridable the same way, e.g.:

```bash
uv run torchrun --nproc_per_node=4 scripts/train.py training.batch_size=32 method.ref_im=blur
```

Each run writes to a timestamped folder under `training.output_dir` (default `./artifacts`): the resolved `config.yaml`, TensorBoard logs, `checkpoint_latest.pt` / `checkpoint_final.pt` (full training state, for `training.resume_from=<path>`), and `weights_final.pt` (explainer weights only).

## Evaluation

```bash
uv run torchrun --nproc_per_node=<num_gpus> scripts/eval.py <artifacts_dir> [--checkpoint weights_final.pt] [key=value ...]
```

`<artifacts_dir>` is a folder produced by `train.py`. The evaluation config is loaded from its `config.yaml` and can be overridden the same way as training (e.g. `eval.split=val eval.ref_im=black eval.cls_to_eval=ground_truth`). This computes the pixel-level deletion/insertion AUC, appends the result to `<artifacts_dir>/eval_metrics.yaml`, and saves 50 random qualitative visualizations (input | heatmap | overlay).

## Pretrained model

`pretrained_models/vit_base_patch16_224/` holds an explainer trained for the ViT-B/16 target classifier: the resolved `config.yaml` of that run plus `weights_final.pt` (explainer weights only. The frozen DINOv3 backbone is not redistributed here, obtain it as described under [Setup](#setup)).

The folder has the same layout as a training run, so it can be passed to `eval.py` and `demo.py` directly:

```bash
uv run torchrun --nproc_per_node=1 scripts/eval.py pretrained_models/vit_base_patch16_224 \
  dataset.root=/path/to/imagenet \
  explainer.backbone_weights=/path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

## Demo

Runs the explainer on a handful of standalone images, using each image's own top-1 prediction as the target class. No dataset needed — only the explainer weights and the DINOv3 backbone:

```bash
uv run python scripts/demo.py pretrained_models/vit_base_patch16_224 \
  explainer.backbone_weights=/path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

Reads from `assets/demo` and writes to `assets/demo/output` by default; both are configurable via `--images` and `--output`, and any config value can be overridden as `key=value`.

## Acknowledgements

We thank the [DINOv3](https://github.com/facebookresearch/dinov3) authors and Meta AI Research, for releasing their implementation and pretrained models. Our explainer uses DINOv3 as its frozen visual backbone. Please also cite [DINOv3](https://arxiv.org/abs/2508.10104) when using these models.

## Citation

```bibtex
@inproceedings{schinagl2026learn,
  title     = {Learn to Rank: Visual Attribution by Learning Importance Ranking},
  author    = {Schinagl, David and Fruhwirth-Reisinger, Christian and Prutsch, Alexander and Schulter, Samuel and Possegger, Horst},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
