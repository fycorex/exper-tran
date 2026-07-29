from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from primary_ml_cka.artifacts.writers import write_json


@dataclass(frozen=True, slots=True)
class LambdaCandidate:
    pair_id: str
    lambda_cka: float
    proxy_representation_shift: float
    reference_cka_gain: float
    source_cka_drop: float
    proxy_target_nll: float


def select_positive_lambda(candidates: Iterable[LambdaCandidate]) -> LambdaCandidate:
    positive = [candidate for candidate in candidates if candidate.lambda_cka > 0]
    if not positive:
        raise ValueError("Lambda selection requires at least one positive candidate")
    return max(
        positive,
        key=lambda item: (
            item.proxy_representation_shift,
            item.reference_cka_gain,
            item.source_cka_drop,
            -item.proxy_target_nll,
        ),
    )


def write_selected(path: Path, selected: Iterable[LambdaCandidate]) -> None:
    write_json(path, [asdict(item) for item in selected])
