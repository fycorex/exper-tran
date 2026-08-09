from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

TensorT = TypeVar("TensorT")


@dataclass(frozen=True, slots=True)
class AttackBatch:
    batch_id: str
    image_ids: tuple[str, ...]
    image_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class ReferenceBatch:
    source_batch_id: str
    image_ids: tuple[str, ...]
    image_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class TapContract:
    model_id: str
    revision: str
    module_path: str
    extraction: str
    pooling: str
    normalization: str
    dtype: str
    status: str
    shape: tuple[int, ...] | None = None
    token_mask: str | None = None
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ProxyLossOutput(Generic[TensorT]):
    loss: TensorT
    target_nll: TensorT
    target_probability: TensorT
    class_logits: TensorT | None = None
    classification_ce: TensorT | None = None
    margin_loss: TensorT | None = None
    max_other_probability: TensorT | None = None
    answer_token_ids: tuple[int, ...] = ()
    label_positions: tuple[int, ...] = ()
    rendered_prompt: str = ""


@dataclass(frozen=True, slots=True)
class ImageEmbeddingOutput(Generic[TensorT]):
    embeddings: TensorT
    tokens: TensorT
    mask: TensorT
    tap: TapContract


@dataclass(frozen=True, slots=True)
class AttackStepMetrics:
    step: int
    loss_ml: float
    loss_cka: float
    loss_total: float
    cka_source: float
    cka_reference: float
    grad_ml_l1: float | None = None
    grad_cka_weighted_l1: float | None = None
    grad_component_cosine: float | None = None


@dataclass(frozen=True, slots=True)
class AttackArtifact:
    pair_id: str
    phase: str
    batch_id: str
    lambda_cka: float
    seed: int
    image_paths: tuple[Path, ...]
    source_image_ids: tuple[str, ...]
    target_reference_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    image_id: str
    clean_raw: str
    clean_label: int | None
    adversarial_raw: str
    adversarial_label: int | None


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    pair_id: str
    phase: str
    denominator: int
    targeted_hits: int
    untargeted_hits: int
    status: str
