from dataclasses import dataclass
from pathlib import Path

from primary_ml_cka.domain.constants import EPSILON, LAMBDAS, STEP_SIZE


@dataclass(frozen=True, slots=True)
class PathsConfig:
    project_root: Path
    imagenet_root: Path
    hf_home: Path
    output_dir: Path


@dataclass(frozen=True, slots=True)
class AttackConfig:
    epsilon: float = EPSILON
    norm: str = "linf"
    pixel_min: float = 0.0
    pixel_max: float = 1.0
    step_size: float = STEP_SIZE
    batch_size: int = 8
    canvas_size: int = 224
    steps: int = 100
    momentum: float = 1.0
    random_start: bool = True
    restarts: int = 1
    lambdas: tuple[float, ...] = LAMBDAS
    main_seed: int = 42
    confirmation_seed: int = 43
    class_margin: float = 2.0
    margin_weight: float = 1.0
    margin_temperature: float = 0.5
    proxy_probability_threshold: float = 0.9
    require_proxy_free_generation: bool = True
    cka_target_weight: float = 1.0


@dataclass(frozen=True, slots=True)
class DataConfig:
    source_human_label: int
    target_human_label: int
    candidate_count: int
    target_reference_count: int
    main_max_count: int
    confirmation_max_count: int
    candidate_split: str = "val"
    calibration_per_class: int = 5
    allow_partial_main_batch: bool = False


@dataclass(frozen=True, slots=True)
class SmokeConfig:
    batch_size: int
    lambdas: tuple[float, ...]
    steps: int
    seed: int
    cka_target_weight: float = 1.0


@dataclass(frozen=True, slots=True)
class AlphaScanConfig:
    lambda_cka: float
    alphas: tuple[float, ...]
    steps: int
    seed: int
