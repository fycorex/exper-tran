"""Shared configuration helpers for the self-contained V4 experiment."""

from dataclasses import dataclass
from pathlib import Path

from primary_ml_cka.config.loader import load_config
from primary_ml_cka.domain.identifiers import get_pair
from primary_ml_cka.domain.labels import CLASS_NAMES

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_ROOT / "config" / "primary.yaml"
DEFAULT_OUTPUT = Path("outputs/pull_push_multiclass_v4")


@dataclass(frozen=True, slots=True)
class Transition:
    transition_id: str
    source: int
    target: int

    @property
    def source_name(self) -> str:
        return CLASS_NAMES[self.source - 1]

    @property
    def target_name(self) -> str:
        return CLASS_NAMES[self.target - 1]


def load_experiment(path: Path = DEFAULT_CONFIG) -> dict:
    raw = load_config(path)
    validate_experiment(raw)
    return raw


def transitions(raw: dict) -> tuple[Transition, ...]:
    return tuple(
        Transition(str(item["id"]), int(item["source"]), int(item["target"]))
        for item in raw["transitions"]
    )


def pair_specs(raw: dict) -> dict[str, dict]:
    return {str(item["pair_id"]): item for item in raw["pairs"]}


def validate_experiment(raw: dict) -> None:
    items = tuple(raw["transitions"])
    if len(items) != 10:
        raise ValueError("V4 requires exactly ten transitions")
    ids = [str(item["id"]) for item in items]
    sources = [int(item["source"]) for item in items]
    targets = [int(item["target"]) for item in items]
    if len(set(ids)) != 10:
        raise ValueError("Transition IDs must be unique")
    if sorted(sources) != list(range(1, 11)):
        raise ValueError("Every class must appear exactly once as a source")
    if sorted(targets) != list(range(1, 11)):
        raise ValueError("Every class must appear exactly once as a target")
    if any(source == target for source, target in zip(sources, targets, strict=True)):
        raise ValueError("Source and target must differ")
    edges = {frozenset((source, target)) for source, target in zip(sources, targets, strict=True)}
    if len(edges) != 10:
        raise ValueError("Undirected transition edges must not repeat")
    for spec in raw["pairs"]:
        pair = get_pair(str(spec["pair_id"]))
        if pair.pair_id not in {"P14", "P16", "P19"}:
            raise ValueError("Primary V4 optimization is restricted to P14/P16/P19")
    arms = {str(arm["name"]): arm for arm in raw["arms"]}
    required = {
        "pull_push_standard",
        "multiclass_standard",
        "pull_push_small_steps",
        "multiclass_small_steps",
    }
    if set(arms) != required:
        raise ValueError("The four controlled loss/schedule arms are required")
    for arm in arms.values():
        if int(arm["steps"]) < 1 or float(arm["step_size"]) <= 0:
            raise ValueError("Attack steps and step size must be positive")


def transition_dir(output_dir: Path, transition_id: str) -> Path:
    return output_dir / "evaluation" / "manifests" / "transitions" / transition_id
