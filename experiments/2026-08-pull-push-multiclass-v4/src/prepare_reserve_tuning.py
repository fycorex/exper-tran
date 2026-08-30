#!/usr/bin/env python3
"""Materialize an eight-image cohort disjoint from scale50 confirmation."""

import argparse
import os
from pathlib import Path

from common import load_experiment, pair_specs, transition_dir, transitions
from screen_transitions import prediction_path, read_predictions

from primary_ml_cka.data.manifests import read_manifest, write_manifest
from primary_ml_cka.domain.identifiers import get_pair

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_ROOT / "config" / "scale50.yaml"
DEFAULT_SOURCE = Path("outputs/pull_push_multiclass_v4_scale50_diverse10")
DEFAULT_OUTPUT = Path("outputs/pull_push_multiclass_v4_reserve8_diverse10")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-output", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raw = load_experiment(DEFAULT_CONFIG)
    specs = pair_specs(raw)
    models = tuple(
        dict.fromkeys(
            model_id
            for pair_id in specs
            for model_id in (
                get_pair(pair_id).proxy_model,
                get_pair(pair_id).target_model,
            )
        )
    )
    canonical_link = args.output_dir / "canonical_images"
    canonical_link.parent.mkdir(parents=True, exist_ok=True)
    expected_target = (args.source_output / "canonical_images").resolve()
    if canonical_link.is_symlink():
        if canonical_link.resolve() != expected_target:
            raise RuntimeError("Reserve canonical symlink points at the wrong cohort")
    elif canonical_link.exists():
        raise RuntimeError("Reserve canonical_images exists but is not a symlink")
    else:
        os.symlink(expected_target, canonical_link)

    manifests = args.output_dir / "evaluation" / "manifests"
    for label in range(1, 11):
        records = read_manifest(
            args.source_output
            / "evaluation"
            / "manifests"
            / f"class_references_{label:02d}.jsonl"
        )
        write_manifest(manifests / f"class_references_{label:02d}.jsonl", records)

    for transition in transitions(raw):
        source_dir = transition_dir(args.source_output, transition.transition_id)
        candidates = read_manifest(source_dir / "candidates.jsonl")
        predictions = tuple(
            read_predictions(
                prediction_path(args.source_output, model_id, transition.transition_id)
            )
            for model_id in models
        )
        common = tuple(
            record
            for record in candidates
            if all(values.get(record.image_id) == transition.source for values in predictions)
        )
        confirmation_ids = {
            record.image_id
            for record in read_manifest(source_dir / "P14_attack_images.jsonl")
        }
        reserve = tuple(
            record for record in common if record.image_id not in confirmation_ids
        )[:8]
        if len(reserve) != 8:
            raise RuntimeError(
                f"{transition.transition_id} has only {len(reserve)} disjoint reserve images"
            )
        if confirmation_ids & {record.image_id for record in reserve}:
            raise RuntimeError("Reserve and confirmation cohorts overlap")
        for pair_id in specs:
            write_manifest(
                transition_dir(args.output_dir, transition.transition_id)
                / f"{pair_id}_attack_images.jsonl",
                reserve,
            )
    print(f"prepared 10 x 8 disjoint reserve images at {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
