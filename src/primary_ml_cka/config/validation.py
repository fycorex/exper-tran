from primary_ml_cka.config.schema import AttackConfig, DataConfig
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
    if config.candidate_count < 1 or config.target_reference_count < 2:
        raise ValueError("Candidate and reference counts are too small")
