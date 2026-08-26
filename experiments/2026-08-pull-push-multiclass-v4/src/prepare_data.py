#!/usr/bin/env python3
"""Prepare canonical candidates and disjoint ten-class reference banks."""

import argparse
import os
from pathlib import Path

from common import DEFAULT_CONFIG, DEFAULT_OUTPUT, load_experiment, transition_dir, transitions

from primary_ml_cka.data.imagenet import _images
from primary_ml_cka.data.manifests import write_manifest
from primary_ml_cka.data.preprocessing import canonicalize_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    raw = load_experiment(args.config)
    project_root = Path.cwd().resolve()
    output_dir = (project_root / args.output_dir).resolve()
    imagenet_root = Path(
        os.environ.get("IMAGENET_ROOT", project_root / "data/imagenet_vehicle_official")
    )
    canonical_root = output_dir / "canonical_images"
    candidate_count = int(raw["candidate_count"])
    reference_count = int(raw["reference_count"])

    class_banks = {}
    for label in range(1, 11):
        references = _images(imagenet_root, "train", label)[:reference_count]
        if len(references) != reference_count:
            raise RuntimeError(f"Class {label} reference bank is incomplete")
        bank = canonicalize_records(
            references,
            imagenet_root,
            canonical_root,
            f"class_references/class_{label:02d}",
            224,
        )
        class_banks[label] = bank
        write_manifest(
            output_dir / "evaluation" / "manifests" / f"class_references_{label:02d}.jsonl",
            bank,
        )

    for transition in transitions(raw):
        candidates = _images(imagenet_root, "val", transition.source)[:candidate_count]
        if len(candidates) != candidate_count:
            raise RuntimeError(f"{transition.transition_id} candidate bank is incomplete")
        canonical = canonicalize_records(
            candidates,
            imagenet_root,
            canonical_root,
            f"transition_candidates/{transition.transition_id}",
            224,
        )
        manifest_dir = transition_dir(output_dir, transition.transition_id)
        write_manifest(manifest_dir / "candidates.jsonl", canonical)
        write_manifest(manifest_dir / "source_references.jsonl", class_banks[transition.source])
        write_manifest(manifest_dir / "target_references.jsonl", class_banks[transition.target])

    print(
        f"prepared transitions=10 candidates={candidate_count} "
        f"references_per_class={reference_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
