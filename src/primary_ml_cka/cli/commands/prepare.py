import os
from pathlib import Path

from primary_ml_cka.data.imagenet import discover_vehicle_pools
from primary_ml_cka.data.manifests import write_manifest
from primary_ml_cka.data.preprocessing import canonicalize_records
from primary_ml_cka.experiment.orchestration import (
    CommandContext,
    resolve_attack_config,
    resolve_data_config,
)


def run(context: CommandContext) -> str:
    root = Path(
        os.environ.get("IMAGENET_ROOT", context.project_root / "data/imagenet_vehicle_official")
    )
    data_config = resolve_data_config(context)
    if context.dry_run:
        return (
            f"dry-run: discover {data_config.candidate_count} validation candidates for "
            f"source label {data_config.source_human_label} and "
            f"{data_config.target_reference_count} training references for target label "
            f"{data_config.target_human_label}, plus "
            f"{data_config.calibration_per_class} calibration images per class under {root}"
        )
    candidates, references, calibration = discover_vehicle_pools(root, data_config)
    attack_config = resolve_attack_config(context)
    canonical_root = context.output_dir / "canonical_images"
    candidates = canonicalize_records(
        candidates, root, canonical_root, "source_candidates", attack_config.canvas_size
    )
    references = canonicalize_records(
        references, root, canonical_root, "target_references", attack_config.canvas_size
    )
    calibration = canonicalize_records(
        calibration, root, canonical_root, "calibration", attack_config.canvas_size
    )
    manifest_dir = context.output_dir / "evaluation" / "manifests"
    write_manifest(manifest_dir / "source_validation_candidates.jsonl", candidates)
    write_manifest(manifest_dir / "target_training_references.jsonl", references)
    write_manifest(manifest_dir / "calibration.jsonl", calibration)
    return (
        f"prepared canonical_{attack_config.canvas_size} candidates={len(candidates)} "
        f"references={len(references)} calibration={len(calibration)}"
    )
