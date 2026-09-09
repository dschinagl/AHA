from dataclasses import dataclass, field
from typing import Literal

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class DatasetConfig:
    _target_: str = "aha.data.imagenet.build_imagenet"
    root: str = MISSING
    train_split: str = "train"
    val_split: str = "val"
    num_samples: int | None = None
    subset_seed: int = 42


@dataclass
class BaseModelConfig:
    _target_: str = "aha.models.vit_timm.VitBasePatch16_224"
    pretrained_tag: str = "vit_base_patch16_224.orig_in21k_ft_in1k"


@dataclass
class ExplainerConfig:
    _target_: str = "aha.explainer.dinov3_dpt.Dinov3DptExplainer"
    dinov3_path: str = MISSING
    backbone_name: str = "dinov3_vitl16"
    backbone_weights: str = MISSING


@dataclass
class SorterConfig:
    _target_: str = "aha.sorter.gumbel_sinkhorn.GumbelSinkhorn"
    sinkhorn_iters: int = 30
    tau: float = 1.0
    gumbel_on: bool = True


@dataclass
class MethodConfig:
    grid_size_min: int = 7
    grid_size_max: int = 28
    grid_size_step: int = 1
    cls_to_optimize: str = "prediction_and_ground_truth"
    ref_im: str = "all"
    ref_im_blur_kernel_size: int = 37
    ref_im_blur_sigma: float = 37.0
    mask_smoothing_sigma: float | None = 1.0
    mask_smoothing_kernel_size: int | None = None
    mask_smoothing_padding_mode: str = "reflect"
    grid_shift_max_frac: float | None = None
    tau_anneal: bool = True
    tau_final: float = 0.3


@dataclass
class LossConfig:
    asc_auc: bool = True
    desc_auc: bool = True
    asc_auc_weight: float = 1.0
    desc_auc_weight: float = 1.0
    peak_penalty_weight: float = 0.0025
    peak_penalty_kernel_size: int = 15
    peak_penalty_relative: bool = False
    peak_penalty_eps: float = 1e-3
    mask_sampling_count: int = 16
    mask_sampling_include_endpoints: bool = True


@dataclass
class OptimizerConfig:
    weight_decay: float = 1e-3


@dataclass
class SchedulerConfig:
    pct_start: float = 0.06
    div_factor: float = 25.0
    final_div_factor: float = 10000.0


@dataclass
class TrainingConfig:
    epochs: int = 1
    learning_rate: float = 3e-4
    batch_size: int = 32
    num_workers: int = 4
    max_grad_norm: float = 1.0
    log_interval_steps: int = 100
    checkpoint_interval_steps: int = 100
    output_dir: str = "./artifacts"
    resume_from: str | None = None


@dataclass
class EvalConfig:
    split: str = "val"
    percentile_step_pct: float = 0.05
    ref_im: str = "mean"
    cls_to_eval: Literal["prediction", "ground_truth"] = "prediction"
    batch_size: int = 32
    num_workers: int = 4


@dataclass
class Config:
    seed: int = 42
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    base_model: BaseModelConfig = field(default_factory=BaseModelConfig)
    explainer: ExplainerConfig = field(default_factory=ExplainerConfig)
    sorter: SorterConfig = field(default_factory=SorterConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=Config)
