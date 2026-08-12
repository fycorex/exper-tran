import argparse
from collections import Counter
from pathlib import Path

from primary_ml_cka.data.manifests import ImageRecord, read_manifest, write_manifest


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--common-clean-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser.parse_args()


def _under_output(output_dir: Path, path: Path) -> Path:
    return path if path.is_absolute() else output_dir / path


def disjoint_calibration(
    calibration: tuple[ImageRecord, ...],
    source_candidates: tuple[ImageRecord, ...],
    attack_records: tuple[ImageRecord, ...],
) -> tuple[ImageRecord, ...]:
    attack_ids = {record.image_id for record in attack_records}
    desired_counts = Counter(record.human_label for record in calibration)
    selected = [record for record in calibration if record.image_id not in attack_ids]
    selected_ids = {record.image_id for record in selected}
    current_counts = Counter(record.human_label for record in selected)

    for candidate in source_candidates:
        label = candidate.human_label
        if current_counts[label] >= desired_counts[label]:
            continue
        if candidate.image_id in attack_ids or candidate.image_id in selected_ids:
            continue
        selected.append(candidate)
        selected_ids.add(candidate.image_id)
        current_counts[label] += 1

    if current_counts != desired_counts:
        raise RuntimeError(
            f"Unable to preserve calibration class counts: expected={dict(desired_counts)} "
            f"actual={dict(current_counts)}"
        )
    selected.sort(key=lambda record: (record.human_label, record.image_id))
    if attack_ids & selected_ids:
        raise RuntimeError("Disjoint calibration still overlaps attack images")
    return tuple(selected)


def main() -> None:
    args = _arguments()
    output_dir = args.output_dir.resolve()
    manifests = output_dir / "evaluation" / "manifests"
    attack_records = read_manifest(_under_output(output_dir, args.common_clean_manifest))
    result = disjoint_calibration(
        read_manifest(manifests / "calibration.jsonl"),
        read_manifest(manifests / "source_validation_candidates.jsonl"),
        attack_records,
    )
    output_manifest = _under_output(output_dir, args.output_manifest)
    write_manifest(output_manifest, result)
    counts = Counter(record.human_label for record in result)
    print(
        f"wrote={output_manifest} rows={len(result)} "
        f"attack_overlap=0 class_counts={dict(sorted(counts.items()))}",
        flush=True,
    )


if __name__ == "__main__":
    main()
