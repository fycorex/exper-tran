import argparse
import csv
import json
from pathlib import Path

from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.domain.identifiers import MODEL_PAIRS
from primary_ml_cka.infrastructure.atomic_io import atomic_text_write


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    output_dir = args.output_dir.resolve()
    expected = len(read_manifest(output_dir / "evaluation/manifests/common_clean_scale50.jsonl"))
    rows = []
    for pair in MODEL_PAIRS:
        logs = sorted((output_dir / "logs" / pair.pair_id / "scale_50").glob("*.json"))
        if not logs:
            raise RuntimeError(f"{pair.pair_id} has no completed attack batches")
        proxy_hits = 0
        proxy_denominator = 0
        target_hits = 0
        target_valid = 0
        untargeted_hits = 0
        conditional_hits = 0
        eligible_batches = 0
        for log_path in logs:
            attack = json.loads(log_path.read_text(encoding="utf-8"))
            batch_index = int(attack["batch_id"])
            evaluation_path = (
                output_dir / "evaluation/scale_50" / pair.pair_id / f"batch_{batch_index:02d}.json"
            )
            if not evaluation_path.is_file():
                raise RuntimeError(f"Missing target evaluation: {evaluation_path}")
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            rates = evaluation["rates"]
            proxy_mask = attack["proxy_target_hit_mask"]
            target_mask = [
                clean["parsed_label"] == 8 and adversarial["parsed_label"] == 7
                for clean, adversarial in zip(
                    evaluation["clean_outputs"], evaluation["adversarial_outputs"], strict=True
                )
            ]
            proxy_hits += sum(proxy_mask)
            proxy_denominator += len(proxy_mask)
            target_hits += int(rates["targeted_hit_count"])
            target_valid += int(rates["clean_valid_count"])
            untargeted_hits += int(rates["untargeted_hit_count"])
            conditional_hits += sum(p and t for p, t in zip(proxy_mask, target_mask, strict=True))
            eligible_batches += int(all(proxy_mask))
        if proxy_denominator != expected or target_valid != expected:
            raise RuntimeError(
                f"{pair.pair_id} denominator mismatch proxy={proxy_denominator} "
                f"target={target_valid} expected={expected}"
            )
        rows.append(
            {
                "pair_id": pair.pair_id,
                "pair_type": pair.exp_type.value,
                "proxy_model": pair.proxy_model,
                "target_model": pair.target_model,
                "image_count": expected,
                "proxy_hits": proxy_hits,
                "proxy_rate_percent": 100 * proxy_hits / expected,
                "eligible_batches": eligible_batches,
                "batch_count": len(logs),
                "target_hits": target_hits,
                "tasr_percent": 100 * target_hits / expected,
                "target_hits_among_proxy_hits": conditional_hits,
                "conditional_tasr_percent": (
                    100 * conditional_hits / proxy_hits if proxy_hits else float("nan")
                ),
                "untargeted_hits": untargeted_hits,
                "asr_percent": 100 * untargeted_hits / expected,
            }
        )
    fieldnames = tuple(rows[0])
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    result_path = output_dir / "summaries/scale_50_semantic_all9.csv"
    atomic_text_write(result_path, buffer.getvalue())
    print(result_path, flush=True)


if __name__ == "__main__":
    main()
