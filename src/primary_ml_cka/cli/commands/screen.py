import gc
import json
import os
from dataclasses import asdict
from pathlib import Path

import torch

from primary_ml_cka.data.manifests import read_manifest, write_manifest
from primary_ml_cka.data.selection import clean_valid_split, require_minimum_sets
from primary_ml_cka.domain.identifiers import MODEL_PAIRS
from primary_ml_cka.experiment.orchestration import (
    CommandContext,
    resolve_attack_config,
    resolve_data_config,
)
from primary_ml_cka.infrastructure.atomic_io import atomic_text_write
from primary_ml_cka.models.backends.target_transformers_generation import (
    load_target_for_generation,
)
from primary_ml_cka.models.backends.transformers_backend import load_processor
from primary_ml_cka.models.common.loading import local_snapshot
from primary_ml_cka.models.targets.generation import TransformersTargetGenerator
from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT


def run(context: CommandContext) -> str:
    if context.dry_run:
        return "dry-run: screen 50 candidates per unique target with exact parser"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU screening is forbidden")
    manifests = context.output_dir / "evaluation" / "manifests"
    candidates_path = manifests / "source_validation_candidates.jsonl"
    if not candidates_path.is_file():
        raise RuntimeError("Candidate manifest missing; run `data prepare` first")
    candidates = read_manifest(candidates_path)
    attack_config = resolve_attack_config(context)
    data_config = resolve_data_config(context)
    imagenet_root = Path(
        os.environ.get("IMAGENET_ROOT", context.project_root / "data/imagenet_vehicle_official")
    )
    target_ids = tuple(
        dict.fromkeys(
            pair.target_model
            for pair in MODEL_PAIRS
            if context.pair_id is None or pair.pair_id == context.pair_id
        )
    )
    summaries = []
    for model_id in target_ids:
        model = None
        safe_name = model_id.replace("/", "__")
        try:
            snapshot = local_snapshot(context.project_root / ".hf-cache", model_id)
            processor = load_processor(snapshot)
            model = load_target_for_generation(snapshot, torch.device("cuda"))
            generator = TransformersTargetGenerator(model, processor)
            outputs = tuple(
                generator.generate_label(
                    imagenet_root / record.relative_path, CLASSIFICATION_PROMPT
                )
                for record in candidates
            )
            labels = tuple(output.parsed_label for output in outputs)
            raw_lines = [
                json.dumps(
                    {"image_id": record.image_id, **asdict(output)},
                    sort_keys=True,
                )
                for record, output in zip(candidates, outputs, strict=True)
            ]
            atomic_text_write(
                context.output_dir / "evaluation" / f"{safe_name}__clean_screen.jsonl",
                "".join(f"{line}\n" for line in raw_lines),
            )
            main, confirmation = clean_valid_split(
                candidates,
                labels,
                data_config.source_human_label,
                attack_config.batch_size,
            )
            main_ok, confirmation_ok = require_minimum_sets(
                main, confirmation, attack_config.batch_size
            )
            write_manifest(manifests / f"{safe_name}__main.jsonl", main)
            write_manifest(manifests / f"{safe_name}__confirmation.jsonl", confirmation)
            if not main_ok:
                raise RuntimeError(
                    f"Only {len(main)} clean-valid main images; "
                    f"minimum is batch_size={attack_config.batch_size}"
                )
            summaries.append(
                f"{model_id}: clean_valid="
                f"{sum(label == data_config.source_human_label for label in labels)} "
                f"main={len(main)} confirmation={len(confirmation)} "
                f"confirmation_available={confirmation_ok}"
            )
        except Exception as exc:
            summaries.append(f"{model_id}: BLOCKED {exc!r}")
        finally:
            if model is not None:
                del model
            gc.collect()
            torch.cuda.empty_cache()
    return "\n".join(summaries)
