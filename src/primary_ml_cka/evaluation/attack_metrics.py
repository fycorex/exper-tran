from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AttackRates:
    clean_valid_count: int
    targeted_hit_count: int
    tasr_percent: float
    untargeted_hit_count: int
    asr_percent: float


def attack_rates(
    clean_labels: tuple[int | None, ...],
    adversarial_labels: tuple[int | None, ...],
    *,
    source_human_label: int,
    target_human_label: int,
) -> AttackRates:
    if len(clean_labels) != len(adversarial_labels):
        raise ValueError("Clean and adversarial outputs must align")
    eligible = [index for index, label in enumerate(clean_labels) if label == source_human_label]
    denominator = len(eligible)
    targeted = sum(adversarial_labels[index] == target_human_label for index in eligible)
    untargeted = sum(adversarial_labels[index] != source_human_label for index in eligible)
    scale = 100.0 / denominator if denominator else 0.0
    return AttackRates(denominator, targeted, targeted * scale, untargeted, untargeted * scale)
