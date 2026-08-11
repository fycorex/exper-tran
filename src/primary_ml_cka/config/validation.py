import math

from primary_ml_cka.config.schema import (
    AlphaScanConfig,
    AttackConfig,
    DataConfig,
    SmokeConfig,
)
from primary_ml_cka.domain.constants import LAMBDAS
from primary_ml_cka.domain.labels import human_label_to_index


def validate_attack_config(
    config: AttackConfig, *, require_canonical_lambda_grid: bool = True
) -> None:
    if config.batch_size < 2:
        raise ValueError("CKA requires batch_size >= 2")
    if config.canvas_size < 16 or config.canvas_size % 16:
        raise ValueError("Attack canvas must be at least 16 and divisible by 16")
    if require_canonical_lambda_grid and config.lambdas != LAMBDAS:
        raise ValueError(f"Lambda grid must be exactly {LAMBDAS}")
    if not config.lambdas or any(
        not math.isfinite(value) or value < 0 for value in config.lambdas
    ):
        raise ValueError("Lambda values must be finite and non-negative")
    if config.epsilon <= 0 or config.step_size <= 0:
        raise ValueError("epsilon and step_size must be positive")
    if config.norm != "linf":
        raise ValueError("Only linf attacks are implemented")
    if config.pixel_min != 0.0 or config.pixel_max != 1.0:
        raise ValueError("Attack projection currently requires pixel_min=0 and pixel_max=1")
    if config.restarts != 1:
        raise ValueError("Only one attack restart is currently implemented")
    if config.class_margin <= 0 or config.margin_temperature <= 0:
        raise ValueError("class_margin and margin_temperature must be positive")
    if not 0 < config.proxy_probability_threshold < 1:
        raise ValueError("proxy_probability_threshold must be in (0,1)")
    if not math.isfinite(config.cka_target_weight) or config.cka_target_weight <= 0:
        raise ValueError("cka_target_weight must be finite and positive")


def validate_data_config(config: DataConfig) -> None:
    human_label_to_index(config.source_human_label)
    human_label_to_index(config.target_human_label)
    if config.source_human_label == config.target_human_label:
        raise ValueError("Source and target labels must differ")
    if config.candidate_split not in {"train", "val"}:
        raise ValueError("candidate_split must be 'train' or 'val'")
    counts = (
        config.candidate_count,
        config.target_reference_count,
        config.main_max_count,
        config.confirmation_max_count,
        config.calibration_per_class,
    )
    if any(count < 1 for count in counts):
        raise ValueError("All configured image counts must be positive")
    if config.main_max_count + config.confirmation_max_count > config.candidate_count:
        raise ValueError("main_max_count + confirmation_max_count cannot exceed candidate_count")
    required_references = config.main_max_count + config.confirmation_max_count
    if config.target_reference_count < required_references:
        raise ValueError(
            "target_reference_count must cover disjoint main and confirmation "
            f"references ({required_references})"
        )


def validate_smoke_config(config: SmokeConfig, attack_config: AttackConfig) -> None:
    if config.batch_size != attack_config.batch_size:
        raise ValueError("Smoke and attack batch sizes must match")
    if config.steps < 1:
        raise ValueError("Smoke steps must be positive")
    if not config.lambdas:
        raise ValueError("Smoke lambda scan cannot be empty")
    if any(not math.isfinite(value) or value < 0 for value in config.lambdas):
        raise ValueError("Smoke lambdas must be finite and non-negative")
    if not math.isfinite(config.cka_target_weight) or config.cka_target_weight <= 0:
        raise ValueError("Smoke CKA target weight must be finite and positive")


def validate_alpha_scan_config(
    config: AlphaScanConfig,
    attack_config: AttackConfig,
) -> None:
    if config.lambda_cka <= 0:
        raise ValueError("Alpha scan requires a positive lambda")
    if config.steps != attack_config.steps:
        raise ValueError("Alpha scan must use the configured full attack steps")
    if not config.alphas or any(alpha < 1 for alpha in config.alphas):
        raise ValueError("Alpha scan values must all be at least one")
