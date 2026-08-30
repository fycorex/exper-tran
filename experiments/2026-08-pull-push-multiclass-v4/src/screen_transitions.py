#!/usr/bin/env python3
"""Screen transition candidates on both models of every small-to-large pair."""

import argparse
import csv
import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch
from common import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    classification_prompt,
    load_experiment,
    pair_specs,
    transition_dir,
    transitions,
)

from primary_ml_cka.data.manifests import read_manifest, write_manifest
from primary_ml_cka.domain.identifiers import get_pair
from primary_ml_cka.infrastructure.atomic_io import atomic_text_write
from primary_ml_cka.models.backends.target_transformers_generation import (
    load_target_for_generation,
)
from primary_ml_cka.models.backends.transformers_backend import load_processor
from primary_ml_cka.models.common.loading import local_snapshot
from primary_ml_cka.models.targets.generation import TransformersTargetGenerator


def safe_model_id(model_id: str) -> str:
    return model_id.replace("/", "__")


def prediction_path(output_dir: Path, model_id: str, transition_id: str) -> Path:
    return (
        output_dir
        / "diagnostics"
        / "clean_screen"
        / safe_model_id(model_id)
        / f"{transition_id}.jsonl"
    )


def read_predictions(path: Path) -> dict[str, int | None]:
    values = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            values[str(payload["image_id"])] = payload["parsed_label"]
    return values


def predictions_complete(path: Path, expected_ids: set[str]) -> bool:
    """Return true only when a resumed screen covers the current manifest."""
    return path.is_file() and set(read_predictions(path)) == expected_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for clean screening")
    raw = load_experiment(args.config)
    prompt = classification_prompt(raw)
    specs = pair_specs(raw)
    pair_ids = tuple(specs)
    models = tuple(
        dict.fromkeys(
            model_id
            for pair_id in pair_ids
            for model_id in (
                get_pair(pair_id).proxy_model,
                get_pair(pair_id).target_model,
            )
        )
    )

    for model_index, model_id in enumerate(models, 1):
        pending = []
        for transition in transitions(raw):
            candidates = read_manifest(
                transition_dir(args.output_dir, transition.transition_id)
                / "candidates.jsonl"
            )
            path = prediction_path(args.output_dir, model_id, transition.transition_id)
            if not (
                args.resume
                and predictions_complete(
                    path,
                    {record.image_id for record in candidates},
                )
            ):
                pending.append(transition)
        if not pending:
            print(f"screen model={model_index}/{len(models)} resume {model_id}", flush=True)
            continue
        model = None
        processor = None
        generator = None
        try:
            snapshot = local_snapshot(Path(".hf-cache"), model_id)
            processor = load_processor(snapshot)
            model = load_target_for_generation(snapshot, torch.device("cuda"))
            generator = TransformersTargetGenerator(model, processor)
            for transition in pending:
                candidates = read_manifest(
                    transition_dir(args.output_dir, transition.transition_id) / "candidates.jsonl"
                )
                lines = []
                for index, record in enumerate(candidates, 1):
                    output = generator.generate_label(
                        args.output_dir / "canonical_images" / record.relative_path,
                        prompt,
                    )
                    lines.append(
                        json.dumps(
                            {"image_id": record.image_id, **asdict(output)},
                            sort_keys=True,
                        )
                    )
                    if index % 10 == 0:
                        print(
                            f"screen {model_id} {transition.transition_id} "
                            f"{index}/{len(candidates)}",
                            flush=True,
                        )
                atomic_text_write(
                    prediction_path(args.output_dir, model_id, transition.transition_id),
                    "".join(f"{line}\n" for line in lines),
                )
        finally:
            if generator is not None:
                del generator
            if model is not None:
                del model
            if processor is not None:
                del processor
            gc.collect()
            torch.cuda.empty_cache()

    rows = []
    failures = []
    attack_count = int(raw["attack_count"])
    common_across_pairs = bool(raw.get("common_across_pairs", False))
    for transition in transitions(raw):
        candidates = read_manifest(
            transition_dir(args.output_dir, transition.transition_id) / "candidates.jsonl"
        )
        common_selected = None
        common_valid_count = None
        if common_across_pairs:
            predictions = tuple(
                read_predictions(
                    prediction_path(args.output_dir, model_id, transition.transition_id)
                )
                for model_id in models
            )
            common_valid = tuple(
                record
                for record in candidates
                if all(
                    model_predictions.get(record.image_id) == transition.source
                    for model_predictions in predictions
                )
            )
            common_selected = common_valid[:attack_count]
            common_valid_count = len(common_valid)
        for pair_id in pair_ids:
            pair = get_pair(pair_id)
            if common_selected is None:
                proxy_predictions = read_predictions(
                    prediction_path(args.output_dir, pair.proxy_model, transition.transition_id)
                )
                target_predictions = read_predictions(
                    prediction_path(args.output_dir, pair.target_model, transition.transition_id)
                )
                valid = tuple(
                    record
                    for record in candidates
                    if proxy_predictions.get(record.image_id) == transition.source
                    and target_predictions.get(record.image_id) == transition.source
                )
                selected = valid[:attack_count]
                valid_count = len(valid)
            else:
                selected = common_selected
                valid_count = int(common_valid_count)
            write_manifest(
                transition_dir(args.output_dir, transition.transition_id)
                / f"{pair_id}_attack_images.jsonl",
                selected,
            )
            rows.append(
                {
                    "pair_id": pair_id,
                    "transition_id": transition.transition_id,
                    "source": transition.source,
                    "target": transition.target,
                    "common_clean_count": valid_count,
                    "selected_count": len(selected),
                }
            )
            if len(selected) < attack_count:
                failures.append(f"{pair_id}/{transition.transition_id}={len(selected)}")

    summary = args.output_dir / "diagnostics" / "clean_screen_summary.csv"
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if failures:
        raise RuntimeError("Insufficient common-clean images: " + ", ".join(failures))
    print(f"wrote {len(rows)} pair-transition cohorts to {summary}", flush=True)


if __name__ == "__main__":
    main()
