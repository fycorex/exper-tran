import argparse
import json
from pathlib import Path

from primary_ml_cka.data.manifests import read_manifest, write_manifest
from primary_ml_cka.domain.identifiers import MODEL_PAIRS

CONTRASTIVE_MODELS = (
    "openai/clip-vit-large-patch14",
    "google/siglip2-so400m-patch14-384",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--source-label", type=int, default=8)
    parser.add_argument("--manifest-name", default="common_clean_scale50.jsonl")
    return parser.parse_args()


def _screen_ids(output_dir: Path, model_id: str, source_label: int) -> set[str]:
    path = output_dir / "evaluation" / f"{model_id.replace('/', '__')}__clean_screen.jsonl"
    if not path.is_file():
        raise RuntimeError(f"Missing clean screen: {path}")
    return {
        row["image_id"]
        for row in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        if row.get("parsed_label") == source_label
    }


def main() -> None:
    args = _arguments()
    output_dir = args.output_dir.resolve()
    manifests = output_dir / "evaluation" / "manifests"
    candidates = read_manifest(manifests / "source_validation_candidates.jsonl")
    target_models = tuple(dict.fromkeys(pair.target_model for pair in MODEL_PAIRS))
    required_models = target_models + CONTRASTIVE_MODELS
    valid_sets = [
        _screen_ids(output_dir, model_id, args.source_label) for model_id in required_models
    ]
    common_ids = set.intersection(*valid_sets)
    common = tuple(record for record in candidates if record.image_id in common_ids)[: args.count]
    if len(common) != args.count:
        raise RuntimeError(
            f"Only {len(common)} of {len(candidates)} candidates are clean-correct for all "
            f"{len(required_models)} models; requested {args.count}"
        )
    common_path = manifests / args.manifest_name
    write_manifest(common_path, common)
    for model_id in target_models:
        write_manifest(manifests / f"{model_id.replace('/', '__')}__main.jsonl", common)
    print(
        f"wrote={common_path} candidates={len(candidates)} common_valid={len(common_ids)} "
        f"frozen={len(common)} models={len(required_models)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
