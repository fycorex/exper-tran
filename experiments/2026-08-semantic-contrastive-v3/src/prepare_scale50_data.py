#!/usr/bin/env python3
"""Prepare the existing all-model-consensus 50-image cohort for v3 loss runs."""

import argparse
import os
import shutil
from pathlib import Path

from primary_ml_cka.data.imagenet import _images
from primary_ml_cka.data.manifests import ImageRecord, read_manifest, write_manifest
from primary_ml_cka.data.preprocessing import canonicalize_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, default=Path("outputs/proxy_selector_cka_v2_scale50"))
    parser.add_argument("--reference-count", type=int, default=48)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = (root / args.output_dir).resolve()
    source_output = (root / args.source_output).resolve()
    imagenet_root = Path(os.environ.get("IMAGENET_ROOT", root / "data/imagenet_vehicle_official"))
    canonical_root = output / "canonical_images"
    manifests = output / "evaluation" / "manifests"

    attack = read_manifest(source_output / "evaluation/manifests/common_clean_scale50.jsonl")
    if len(attack) != 50:
        raise RuntimeError(f"Expected 50 common clean images, got {len(attack)}")

    def copy_records(records: tuple[ImageRecord, ...], source_root: Path, subdir: str) -> tuple[ImageRecord, ...]:
        copied = []
        for record in records:
            src = source_root / record.relative_path
            dst = canonical_root / subdir / Path(record.relative_path).name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(
                ImageRecord(
                    image_id=record.image_id,
                    relative_path=dst.relative_to(canonical_root),
                    human_label=record.human_label,
                    class_name=record.class_name,
                    synset=record.synset,
                )
            )
        return tuple(copied)

    attack = copy_records(attack, source_output / "canonical_images", "attack_images")
    target = read_manifest(source_output / "evaluation/manifests/target_training_references.jsonl")[: args.reference_count]
    target = copy_records(target, source_output / "canonical_images", "target_references")
    attack_ids = {r.image_id for r in attack}
    source_originals = tuple(
        record
        for record in _images(imagenet_root, "train", 8)
        if record.image_id not in attack_ids
    )[: args.reference_count]
    source = canonicalize_records(source_originals, imagenet_root, canonical_root, "source_references", 224)
    banks = []
    for label in range(1, 11):
        if label == 7:
            bank = target
        elif label == 8:
            bank = source
        else:
            originals = tuple(
                record
                for record in _images(imagenet_root, "train", label)
                if record.image_id not in attack_ids
            )[: args.reference_count]
            bank = canonicalize_records(
                originals,
                imagenet_root,
                canonical_root,
                f"class_references/class_{label:02d}",
                224,
            )
        if attack_ids & {r.image_id for r in bank}:
            raise RuntimeError(f"Class {label} bank overlaps attack images")
        banks.append(bank)
    write_manifest(manifests / "attack_images_scale50.jsonl", attack)
    write_manifest(manifests / "common_clean_scale50.jsonl", attack)
    write_manifest(manifests / "source_references.jsonl", source)
    write_manifest(manifests / "target_training_references.jsonl", target)
    write_manifest(manifests / "target_references.jsonl", target)
    for label, bank in enumerate(banks, 1):
        write_manifest(manifests / f"class_references_{label:02d}.jsonl", bank)
    print(f"prepared v3 scale50 attack={len(attack)} class_banks=10 refs={args.reference_count}")


if __name__ == "__main__":
    main()
