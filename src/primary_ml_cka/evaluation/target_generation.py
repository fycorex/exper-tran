import gc
from dataclasses import dataclass
from pathlib import Path

import torch

from primary_ml_cka.evaluation.attack_metrics import AttackRates, attack_rates
from primary_ml_cka.models.backends.target_transformers_generation import (
    load_target_for_generation,
)
from primary_ml_cka.models.backends.transformers_backend import load_processor
from primary_ml_cka.models.common.loading import local_snapshot
from primary_ml_cka.models.common.outputs import GenerationOutput
from primary_ml_cka.models.common.protocols import TargetGenerator
from primary_ml_cka.models.targets.generation import TransformersTargetGenerator


def evaluate_paths(
    generator: TargetGenerator, paths: tuple[Path, ...], prompt: str
) -> tuple[GenerationOutput, ...]:
    return tuple(generator.generate_label(path, prompt) for path in paths)


@dataclass(frozen=True, slots=True)
class FrozenBatchEvaluation:
    clean_outputs: tuple[GenerationOutput, ...]
    adversarial_outputs: tuple[GenerationOutput, ...]
    rates: AttackRates


def evaluate_local_frozen_batch(
    *,
    model_id: str,
    hf_home: Path,
    artifact_dir: Path,
    image_count: int,
    prompt: str,
    source_human_label: int,
    target_human_label: int,
) -> FrozenBatchEvaluation:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU target evaluation is forbidden")
    model = None
    try:
        snapshot = local_snapshot(hf_home, model_id)
        processor = load_processor(snapshot)
        model = load_target_for_generation(snapshot, torch.device("cuda"))
        generator = TransformersTargetGenerator(model, processor)
        clean_paths = tuple(artifact_dir / f"{index:02d}_clean.png" for index in range(image_count))
        adversarial_paths = tuple(
            artifact_dir / f"{index:02d}_adv.png" for index in range(image_count)
        )
        clean_outputs = evaluate_paths(generator, clean_paths, prompt)
        adversarial_outputs = evaluate_paths(generator, adversarial_paths, prompt)
        rates = attack_rates(
            tuple(output.parsed_label for output in clean_outputs),
            tuple(output.parsed_label for output in adversarial_outputs),
            source_human_label=source_human_label,
            target_human_label=target_human_label,
        )
        return FrozenBatchEvaluation(clean_outputs, adversarial_outputs, rates)
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()
