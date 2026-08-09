import csv
import json
from collections import defaultdict

import pandas as pd
import torch

from primary_ml_cka.artifacts.writers import write_json
from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.domain.identifiers import MODEL_PAIRS
from primary_ml_cka.evaluation.model_similarity import proxy_target_similarity
from primary_ml_cka.experiment.orchestration import CommandContext, require_real_run_ready
from primary_ml_cka.infrastructure.atomic_io import atomic_text_write
from primary_ml_cka.models.analysis.representations import extract_representations

LOCAL_NEIGHBOR_COUNT = 8


def _eligible_rows(path, pair_id: str) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [
            row
            for row in csv.DictReader(handle)
            if row["pair_id"] == pair_id
            and row["phase"] == "main"
            and row["status"] == "ok"
            and row.get("tasr_percent", "") != ""
        ]


def _target_hits(context: CommandContext, pair_id: str, target_label: int):
    path = context.output_dir / "evaluation" / f"{pair_id}__main_outputs.jsonl"
    hits = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            key = (payload["batch_id"], payload["lambda"], payload["image_id"])
            hits[key] = int(payload["adversarial"]["parsed_label"] == target_label)
    return hits


def run(context: CommandContext) -> str:
    """Post-attack analysis. Target representations cannot affect attack generation."""
    require_real_run_ready(context, require_taps=False)
    if context.dry_run:
        return "dry-run: sequential model extraction; global/local CKA after frozen evaluation"
    results_path = context.output_dir / "summaries" / "all_results.csv"
    calibration_records = read_manifest(
        context.output_dir / "evaluation" / "manifests" / "calibration.jsonl"
    )
    canonical_root = context.output_dir / "canonical_images"
    calibration_paths = [canonical_root / record.relative_path for record in calibration_records]
    pair_lines = []
    local_lines = []
    summaries = []
    for pair in MODEL_PAIRS:
        if context.pair_id is not None and pair.pair_id != context.pair_id:
            continue
        rows = _eligible_rows(results_path, pair.pair_id)
        if not rows:
            summaries.append(f"{pair.pair_id}: no eligible evaluated rows")
            continue
        query_metadata = []
        query_paths = []
        for row in rows:
            artifact_dir = (
                context.output_dir
                / "attacks"
                / pair.pair_id
                / "main"
                / f"batch_{int(row['batch_id']):02d}"
                / f"lambda_{float(row['lambda']):g}"
            )
            for index, image_id in enumerate(row["source_image_ids"].split("|")):
                query_paths.append(artifact_dir / f"{index:02d}_clean.png")
                query_metadata.append((row["batch_id"], row["lambda"], image_id))
        proxy_all = extract_representations(
            pair.proxy_model,
            calibration_paths + query_paths,
            context.project_root / ".hf-cache",
            torch.device("cuda"),
        )
        target_calibration = extract_representations(
            pair.target_model,
            calibration_paths,
            context.project_root / ".hf-cache",
            torch.device("cuda"),
        )
        calibration_count = len(calibration_paths)
        similarity = proxy_target_similarity(
            proxy_all[:calibration_count],
            target_calibration,
            proxy_all[calibration_count:],
            neighbor_count=min(LOCAL_NEIGHBOR_COUNT, calibration_count),
        )
        by_lambda: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            by_lambda[row["lambda"]].append(row)
        for lambda_cka, lambda_rows in by_lambda.items():
            hits = sum(int(row["targeted_hit_count"]) for row in lambda_rows)
            denominator = sum(int(row["clean_valid_count"]) for row in lambda_rows)
            tasr = 100.0 * hits / denominator if denominator else float("nan")
            pair_lines.append(
                {
                    "pair_id": pair.pair_id,
                    "exp_type": pair.exp_type.value,
                    "lambda": lambda_cka,
                    "global_cka": similarity.global_cka,
                    "targeted_hits": hits,
                    "denominator": denominator,
                    "tasr_percent": tasr,
                }
            )
        hit_lookup = _target_hits(context, pair.pair_id, int(rows[0]["target_human_label"]))
        for metadata, local_cka, neighbors in zip(
            query_metadata,
            similarity.local_cka,
            similarity.neighbor_indices,
            strict=True,
        ):
            batch_id, lambda_cka, image_id = metadata
            local_lines.append(
                {
                    "pair_id": pair.pair_id,
                    "batch_id": batch_id,
                    "lambda": lambda_cka,
                    "image_id": image_id,
                    "local_cka": local_cka,
                    "targeted_hit": hit_lookup[(batch_id, lambda_cka, image_id)],
                    "neighbor_indices": "|".join(map(str, neighbors)),
                }
            )
        summaries.append(
            f"{pair.pair_id}: global_cka={similarity.global_cka:.6f} "
            f"local_images={len(similarity.local_cka)}"
        )
    _write_csv(context.output_dir / "summaries" / "model_similarity.csv", pair_lines)
    _write_csv(context.output_dir / "summaries" / "local_similarity.csv", local_lines)
    write_json(
        context.output_dir / "summaries" / "similarity_correlations.json",
        _correlation_summary(pair_lines, local_lines),
    )
    return "\n".join(summaries)


def _write_csv(path, rows: list[dict[str, object]]) -> None:
    if not rows:
        atomic_text_write(path, "")
        return
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=tuple(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text_write(path, buffer.getvalue())


def _safe_correlation(frame: pd.DataFrame, left: str, right: str, method: str):
    if len(frame) < 2 or frame[left].nunique() < 2 or frame[right].nunique() < 2:
        return None
    return float(frame[left].corr(frame[right], method=method))


def _correlation_summary(pair_rows, local_rows) -> dict[str, object]:
    pair_frame = pd.DataFrame(pair_rows)
    local_frame = pd.DataFrame(local_rows)
    by_lambda = {}
    if not pair_frame.empty:
        for lambda_cka, group in pair_frame.groupby("lambda"):
            by_lambda[str(lambda_cka)] = {
                "pair_count": len(group),
                "pearson_global_cka_tasr": _safe_correlation(
                    group, "global_cka", "tasr_percent", "pearson"
                ),
                "spearman_global_cka_tasr": _safe_correlation(
                    group, "global_cka", "tasr_percent", "spearman"
                ),
            }
    local_summary = {
        "image_count": len(local_frame),
        "point_biserial_local_cka_hit": (
            _safe_correlation(local_frame, "local_cka", "targeted_hit", "pearson")
            if not local_frame.empty
            else None
        ),
    }
    return {
        "pair_level_by_lambda": by_lambda,
        "image_level": local_summary,
        "note": (
            "Correlations are descriptive. Pair-level inference must use model pair as the "
            "unit; repeated lambdas are reported separately, not pooled as independent pairs."
        ),
    }
