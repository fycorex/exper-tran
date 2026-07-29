import os
from pathlib import Path

from primary_ml_cka.data.imagenet import discover_vehicle_pools
from primary_ml_cka.data.manifests import write_manifest
from primary_ml_cka.experiment.orchestration import CommandContext, resolve_data_config


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
            f"{data_config.target_human_label} under {root}"
        )
    candidates, references = discover_vehicle_pools(root, data_config)
    manifest_dir = context.output_dir / "evaluation" / "manifests"
    write_manifest(manifest_dir / "source_validation_candidates.jsonl", candidates)
    write_manifest(manifest_dir / "target_training_references.jsonl", references)
    return f"prepared candidates={len(candidates)} references={len(references)}"
