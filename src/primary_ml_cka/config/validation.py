from primary_ml_cka.config.schema import (
    AlphaScanConfig,
    AttackConfig,
    DataConfig,
    PrototypeScanConfig,
    SharedCKAScanConfig,
    SmokeConfig,
)
from primary_ml_cka.domain.constants import LAMBDAS
from primary_ml_cka.domain.labels import human_label_to_index


def validate_attack_config(config: AttackConfig) -> None:
    if config.batch_size < 2:
        raise ValueError("CKA requires batch_size >= 2")
    if config.canvas_size < 16 or config.canvas_size % 16:
        raise ValueError("Attack canvas must be at least 16 and divisible by 16")
    if config.lambdas != LAMBDAS:
        raise ValueError(f"Lambda grid must be exactly {LAMBDAS}")
    if config.epsilon <= 0 or config.step_size <= 0:
        raise ValueError("epsilon and step_size must be positive")


def validate_data_config(config: DataConfig) -> None:
    human_label_to_index(config.source_human_label)
    human_label_to_index(config.target_human_label)
    if config.source_human_label == config.target_human_label:
        raise ValueError("Source and target labels must differ")
    counts = (
        config.candidate_count,
        config.target_reference_count,
        config.main_max_count,
        config.confirmation_max_count,
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
    if tuple(config.lambdas) != attack_config.lambdas:
        raise ValueError("Smoke must scan the complete configured lambda grid")


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


def validate_prototype_scan_config(
    config: PrototypeScanConfig,
    attack_config: AttackConfig,
) -> None:
    if config.batch_size != attack_config.batch_size:
        raise ValueError("Prototype scan and attack batch sizes must match")
    if config.steps != attack_config.steps:
        raise ValueError("Prototype scan must use the configured full attack steps")
    if not config.lambda_values or any(value < 0 for value in config.lambda_values):
        raise ValueError("Prototype lambda values must be non-negative")
    if config.margin < 0 or config.separation_weight < 0:
        raise ValueError("Prototype margin and separation weight must be non-negative")


def validate_shared_cka_scan_config(
    config: SharedCKAScanConfig,
    attack_config: AttackConfig,
) -> None:
    if config.batch_size != attack_config.batch_size:
        raise ValueError("Shared-CKA scan and attack batch sizes must match")
    if config.steps != attack_config.steps:
        raise ValueError("Shared-CKA scan must use the configured full attack steps")
    if config.prototype_lambda <= 0:
        raise ValueError("Shared-CKA scan requires a positive prototype weight")
    if config.clean_separation_weight < 0:
        raise ValueError("Own-clean separation weight must be non-negative")
    if not config.shared_clean_weights or any(
        value < 0 for value in config.shared_clean_weights
    ):
        raise ValueError("Shared-clean weights must be non-empty and non-negative")
    if config.view_consistency_weight < 0:
        raise ValueError("View-consistency weight must be non-negative")
    if not config.view_scales or any(not 0 < value <= 1 for value in config.view_scales):
        raise ValueError("View scales must be in (0,1]")
