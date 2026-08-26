#!/usr/bin/env python3
"""Materialize the fixed v3 attack cohort and disjoint semantic banks."""

import argparse
import os
import shutil
from pathlib import Path

from primary_ml_cka.data.imagenet import _images
from primary_ml_cka.data.manifests import ImageRecord, read_manifest, write_manifest
from primary_ml_cka.data.preprocessing import canonicalize_records


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/proxy_selector_semantic_contrastive_v3"),
    )
    parser.add_argument("--v2-output", type=Path, default=Path("outputs/proxy_selector_cka_v2"))
    parser.add_argument("--reference-count", type=int, default=48)
    return parser.parse_args()


def copy_canonical_records(
    records: tuple[ImageRecord, ...], source_root: Path, destination_root: Path
) -> tuple[ImageRecord, ...]:
    copied = []
    for record in records:
        source = source_root / record.relative_path
        destination = destination_root / record.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(record)
    return tuple(copied)


def main() -> None:
    args = arguments()
    project_root = Path.cwd().resolve()
    imagenet_root = Path(
        os.environ.get("IMAGENET_ROOT", project_root / "data/imagenet_vehicle_official")
    )
    output_dir = (project_root / args.output_dir).resolve()
    v2_output = (project_root / args.v2_output).resolve()
    canonical_root = output_dir / "canonical_images"
    manifests = output_dir / "evaluation" / "manifests"

    attack_manifest = (
        v2_output / "diagnostics" / "objective_split_all9v2_common48_rho03" / "common_clean.jsonl"
    )
    attack_records = read_manifest(attack_manifest)
    if len(attack_records) != 8:
        raise RuntimeError(f"Expected fixed v2 cohort of 8, got {len(attack_records)}")
    attack_records = copy_canonical_records(
        attack_records, v2_output / "canonical_images", canonical_root
    )

    target_records = read_manifest(
        v2_output / "evaluation" / "manifests" / "target_training_references.jsonl"
    )[: args.reference_count]
    target_records = copy_canonical_records(
        target_records, v2_output / "canonical_images", canonical_root
    )

    # The attacked cohort is validation data; deterministic source references
    # come from training class 8, so IDs and underlying files are disjoint.
    source_originals = _images(imagenet_root, "train", 8)[: args.reference_count]
    source_records = canonicalize_records(
        source_originals,
        imagenet_root,
        canonical_root,
        "source_references",
        224,
    )

    attack_ids = {record.image_id for record in attack_records}
    source_ids = {record.image_id for record in source_records}
    target_ids = {record.image_id for record in target_records}
    if attack_ids & source_ids or attack_ids & target_ids:
        raise RuntimeError("Attack images overlap a semantic reference bank")
    if source_ids & target_ids:
        raise RuntimeError("Source and target semantic banks overlap")
    if len(source_ids) != args.reference_count or len(target_ids) != args.reference_count:
        raise RuntimeError("Semantic reference bank contains duplicate or missing IDs")

    class_reference_banks = []
    for human_label in range(1, 11):
        if human_label == 7:
            records = target_records
        elif human_label == 8:
            records = source_records
        else:
            originals = _images(imagenet_root, "train", human_label)[: args.reference_count]
            records = canonicalize_records(
                originals,
                imagenet_root,
                canonical_root,
                f"class_references/class_{human_label:02d}",
                224,
            )
        record_ids = {record.image_id for record in records}
        if len(record_ids) != args.reference_count:
            raise RuntimeError(f"Class {human_label} reference bank is incomplete")
        if attack_ids & record_ids:
            raise RuntimeError(f"Class {human_label} reference bank overlaps attacked images")
        class_reference_banks.append(records)

    write_manifest(manifests / "attack_images.jsonl", attack_records)
    write_manifest(manifests / "source_references.jsonl", source_records)
    write_manifest(manifests / "target_references.jsonl", target_records)
    for human_label, records in enumerate(class_reference_banks, start=1):
        write_manifest(manifests / f"class_references_{human_label:02d}.jsonl", records)
    print(
        f"prepared attack={len(attack_records)} source_refs={len(source_records)} "
        f"target_refs={len(target_records)} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
