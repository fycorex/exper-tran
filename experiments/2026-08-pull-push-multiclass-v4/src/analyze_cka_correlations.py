#!/usr/bin/env python3
"""Post-hoc class-conditioned CKA and transfer correlation analysis."""

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as functional
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from common import class_names, load_experiment, pair_specs, transitions

from primary_ml_cka.attack.cka.linear import linear_cka
from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.domain.identifiers import get_pair
from primary_ml_cka.evaluation.model_similarity import cka_permutation_baseline
from primary_ml_cka.models.analysis.representations import extract_representations

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_ROOT / "config" / "scale50_tuned.yaml"
DEFAULT_OUTPUT = Path("outputs/pull_push_multiclass_v4_scale50_diverse10")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        rank = (start + stop + 1) / 2
        for index in range(start, stop):
            result[order[index]] = rank
        start = stop
    return result


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Correlation inputs must have matching length >= 2")
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0:
        return float("nan")
    return sum(
        left_value * right_value
        for left_value, right_value in zip(
            left_centered, right_centered, strict=True
        )
    ) / denominator


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(average_ranks(left), average_ranks(right))


def permutation_p_value(
    left: list[float],
    right: list[float],
    observed: float,
    *,
    permutations: int,
    seed: int,
) -> float:
    if not math.isfinite(observed):
        return float("nan")
    generator = random.Random(seed)
    shuffled = right[:]
    exceedances = 0
    for _ in range(permutations):
        generator.shuffle(shuffled)
        value = spearman(left, shuffled)
        exceedances += abs(value) >= abs(observed) - 1e-12
    return (exceedances + 1) / (permutations + 1)


def stratified_rank_correlation(
    rows: list[dict],
    metric: str,
    outcome: str,
    *,
    permutations: int,
    seed: int,
) -> tuple[float, float]:
    by_pair = defaultdict(list)
    for row in rows:
        by_pair[str(row["pair_id"])].append(row)
    left = []
    right_groups = []
    for pair_id in sorted(by_pair):
        group = by_pair[pair_id]
        metric_ranks = average_ranks([float(row[metric]) for row in group])
        outcome_ranks = average_ranks([float(row[outcome]) for row in group])
        metric_mean = sum(metric_ranks) / len(metric_ranks)
        outcome_mean = sum(outcome_ranks) / len(outcome_ranks)
        left.extend(value - metric_mean for value in metric_ranks)
        right_groups.append([value - outcome_mean for value in outcome_ranks])
    right = [value for group in right_groups for value in group]
    observed = pearson(left, right)
    if not math.isfinite(observed):
        return observed, float("nan")
    generator = random.Random(seed)
    exceedances = 0
    for _ in range(permutations):
        shuffled = []
        for group in right_groups:
            values = group[:]
            generator.shuffle(values)
            shuffled.extend(values)
        value = pearson(left, shuffled)
        exceedances += abs(value) >= abs(observed) - 1e-12
    return observed, (exceedances + 1) / (permutations + 1)


def path_fingerprint(paths: list[Path]) -> str:
    return hashlib.sha256(
        "\n".join(str(path.resolve()) for path in paths).encode("utf-8")
    ).hexdigest()


def cached_representations(
    model_id: str,
    paths: list[Path],
    cache_path: Path,
    *,
    hf_home: Path,
    resume: bool,
) -> torch.Tensor:
    fingerprint = path_fingerprint(paths)
    if resume and cache_path.is_file():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if (
            payload.get("model_id") == model_id
            and payload.get("path_count") == len(paths)
            and payload.get("path_fingerprint") == fingerprint
        ):
            print(f"resume representations model={model_id} rows={len(paths)}", flush=True)
            return payload["features"]
    print(f"extract representations model={model_id} rows={len(paths)}", flush=True)
    features = extract_representations(
        model_id,
        paths,
        hf_home,
        torch.device("cuda"),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_id": model_id,
            "path_count": len(paths),
            "path_fingerprint": fingerprint,
            "features": features,
        },
        cache_path,
    )
    return features


def prototype_distances(features: torch.Tensor, class_size: int) -> torch.Tensor:
    centers = []
    for class_index in range(10):
        start = class_index * class_size
        values = functional.normalize(
            features[start : start + class_size].float(), dim=-1
        )
        centers.append(functional.normalize(values.mean(dim=0), dim=0))
    centers = torch.stack(centers)
    return 1.0 - centers @ centers.T


def class_geometry(features: torch.Tensor, class_size: int) -> tuple[torch.Tensor, list[dict]]:
    """Return normalized centers and within-class compactness statistics."""
    centers = []
    rows = []
    for class_index in range(10):
        start = class_index * class_size
        values = functional.normalize(
            features[start : start + class_size].float(), dim=-1
        )
        center = functional.normalize(values.mean(dim=0), dim=0)
        similarities = values @ center
        centered = values - values.mean(dim=0, keepdim=True)
        singular_values = torch.linalg.svdvals(centered)
        variance = singular_values.square()
        probabilities = variance / variance.sum().clamp_min(1e-12)
        effective_rank = torch.exp(
            -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        )
        centers.append(center)
        rows.append(
            {
                "mean_cosine_to_centroid": float(similarities.mean()),
                "std_cosine_to_centroid": float(similarities.std(unbiased=False)),
                "dispersion": float(1.0 - similarities.mean()),
                "effective_rank": float(effective_rank),
            }
        )
    return torch.stack(centers), rows


def rsa_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left = functional.normalize(left.float(), dim=-1)
    right = functional.normalize(right.float(), dim=-1)
    indices = torch.triu_indices(left.shape[0], left.shape[0], offset=1)
    left_values = (left @ left.T)[indices[0], indices[1]].tolist()
    right_values = (right @ right.T)[indices[0], indices[1]].tolist()
    return pearson(left_values, right_values)


def distance_percentile(matrix: torch.Tensor, source_index: int, target_index: int) -> float:
    indices = torch.triu_indices(matrix.shape[0], matrix.shape[0], offset=1)
    values = matrix[indices[0], indices[1]]
    selected = matrix[source_index, target_index]
    return float((values <= selected).float().mean())


def read_transfer_rows(path: Path) -> dict[tuple[str, str], dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["pair_id"], row["transition_id"]): row
            for row in csv.DictReader(handle)
        }


def read_selected_distances(path: Path) -> dict[tuple[str, str], float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["pair_id"], row["transition_id"]): float(
                row["prototype_cosine_distance"]
            )
            for row in csv.DictReader(handle)
        }


def augment_perturbation_metrics(rows: list[dict], output_dir: Path, raw: dict) -> None:
    """Add frozen-PNG and saved proxy/target-output diagnostics in place."""
    by_key = {(row["pair_id"], row["transition_id"]): row for row in rows}
    state_namespace = str(raw["state_namespace"])
    for key, row in by_key.items():
        pair_id, transition_id = key
        states = []
        state_dir = output_dir / state_namespace / pair_id / transition_id
        for path in sorted(state_dir.glob("batch_*.json")):
            states.append(json.loads(path.read_text(encoding="utf-8")))
        if len(states) != 7 or any(state.get("status") != "complete" for state in states):
            raise RuntimeError(f"Incomplete tuned states for {pair_id}/{transition_id}")
        image_metrics = []
        adversarial_labels = []
        proxy_probabilities = []
        proxy_margins = []
        effective_lambdas = []
        for state in states:
            directory = (
                output_dir
                / "attacks"
                / pair_id
                / str(state["attack"]["phase"])
                / f"batch_{int(state['batch_index']):02d}"
                / str(state["objective_tag"])
                / "lambda_1"
            )
            for index in range(int(state["source_count"])):
                with Image.open(directory / f"{index:02d}_clean.png") as image:
                    clean = pil_to_tensor(image.convert("RGB")).float().div(255)
                with Image.open(directory / f"{index:02d}_adv.png") as image:
                    adversarial = pil_to_tensor(image.convert("RGB")).float().div(255)
                delta = adversarial - clean
                absolute = delta.abs()
                horizontal = (delta[:, :, 1:] - delta[:, :, :-1]).abs().mean()
                vertical = (delta[:, 1:, :] - delta[:, :-1, :]).abs().mean()
                image_metrics.append(
                    {
                        "mean_abs": float(absolute.mean()),
                        "rms": float(delta.square().mean().sqrt()),
                        "l2": float(delta.flatten().norm()),
                        "linf": float(absolute.max()),
                        "nonzero_fraction": float((absolute >= 0.5 / 255).float().mean()),
                        "saturation_fraction": float(
                            (absolute >= 15.5 / 255).float().mean()
                        ),
                        "total_variation": float(horizontal + vertical),
                    }
                )
            adversarial_labels.extend(
                item["parsed_label"] for item in state["target"]["adversarial_outputs"]
            )
            proxy_probabilities.append(float(state["attack"]["proxy_target_probability"]))
            proxy_margins.append(float(state["attack"]["proxy_min_target_logit_margin"]))
            effective_lambdas.append(float(state["attack"]["effective_lambda_cka"]))
        if len(image_metrics) != 50:
            raise RuntimeError(f"Expected 50 frozen deltas for {pair_id}/{transition_id}")
        for metric in image_metrics[0]:
            values = [item[metric] for item in image_metrics]
            row[f"perturbation_{metric}"] = sum(values) / len(values)
        counts = defaultdict(int)
        for label in adversarial_labels:
            counts[str(label)] += 1
        probabilities = [count / len(adversarial_labels) for count in counts.values()]
        entropy = -sum(value * math.log(value) for value in probabilities)
        row["target_output_entropy_normalized"] = entropy / math.log(10)
        row["target_output_distinct_labels"] = len(counts)
        row["proxy_mean_target_probability"] = sum(proxy_probabilities) / len(
            proxy_probabilities
        )
        row["proxy_min_target_margin_across_batches"] = min(proxy_margins)
        row["mean_effective_lambda"] = sum(effective_lambdas) / len(
            effective_lambdas
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cka-permutations", type=int, default=1_000)
    parser.add_argument("--correlation-permutations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for post-hoc representation extraction")
    raw = load_experiment(args.config)
    output_dir = args.output_dir.resolve()
    result_dir = output_dir / "diagnostics" / "tuned_transfer_correlations"
    references = []
    for label in range(1, 11):
        records = read_manifest(
            output_dir
            / "evaluation"
            / "manifests"
            / f"class_references_{label:02d}.jsonl"
        )
        if len(records) != int(raw["reference_count"]):
            raise RuntimeError(f"Class {label} reference count changed")
        references.extend(records)
    paths = [
        output_dir / "canonical_images" / record.relative_path for record in references
    ]
    image_ids = [record.image_id for record in references]
    if len(set(image_ids)) != len(image_ids):
        raise RuntimeError("CKA reference rows must identify unique images")
    class_size = int(raw["reference_count"])
    transfer = read_transfer_rows(
        output_dir / "summaries" / str(raw["summary_filename"])
    )
    selected_distances = read_selected_distances(
        Path("outputs/pull_push_multiclass_v4_diverse10")
        / "diagnostics"
        / "prototype_distances.csv"
    )
    metric_rows = []
    pair_rows = []
    names = class_names(raw)
    for pair_id, _spec in pair_specs(raw).items():
        pair = get_pair(pair_id)
        proxy = cached_representations(
            pair.proxy_model,
            paths,
            result_dir / "cache" / f"{pair_id}_proxy_projected.pt",
            hf_home=Path(".hf-cache"),
            resume=args.resume,
        )
        target = cached_representations(
            pair.target_model,
            paths,
            result_dir / "cache" / f"{pair_id}_target_projected.pt",
            hf_home=Path(".hf-cache"),
            resume=args.resume,
        )
        proxy_centers, proxy_class_geometry = class_geometry(proxy, class_size)
        target_centers, target_class_geometry = class_geometry(target, class_size)
        proxy_distance = 1.0 - proxy_centers @ proxy_centers.T
        target_distance = 1.0 - target_centers @ target_centers.T
        global_null = cka_permutation_baseline(
            proxy,
            target,
            permutation_count=args.cka_permutations,
            seed=args.seed,
        )
        class_cka = {}
        for label in range(1, 11):
            start = (label - 1) * class_size
            stop = label * class_size
            class_cka[label] = cka_permutation_baseline(
                proxy[start:stop],
                target[start:stop],
                permutation_count=args.cka_permutations,
                seed=args.seed + label,
            )
        centroid_null = cka_permutation_baseline(
            proxy_centers,
            target_centers,
            permutation_count=args.cka_permutations,
            seed=args.seed,
        )
        pair_rows.append(
            {
                "pair_id": pair_id,
                "proxy_model": pair.proxy_model,
                "target_model": pair.target_model,
                "image_count": len(paths),
                "projected_global_cka": global_null.true_cka,
                "global_null_mean": global_null.null_mean,
                "global_cka_normalized": (
                    (global_null.true_cka - global_null.null_mean)
                    / (1.0 - global_null.null_mean)
                ),
                "global_z_score": global_null.z_score,
                "global_empirical_p": global_null.empirical_p_value,
                "class_centroid_cka": centroid_null.true_cka,
                "class_centroid_cka_normalized": (
                    (centroid_null.true_cka - centroid_null.null_mean)
                    / (1.0 - centroid_null.null_mean)
                ),
                "class_geometry_rsa": rsa_correlation(
                    proxy_centers, target_centers
                ),
                "mean_proxy_class_dispersion": sum(
                    row["dispersion"] for row in proxy_class_geometry
                )
                / 10,
                "mean_target_class_dispersion": sum(
                    row["dispersion"] for row in target_class_geometry
                )
                / 10,
                "aggregate_tasr_percent": sum(
                    float(transfer[pair_id, transition.transition_id]["tasr_hits"])
                    for transition in transitions(raw)
                )
                / 5,
                "aggregate_asr_percent": sum(
                    float(transfer[pair_id, transition.transition_id]["asr_hits"])
                    for transition in transitions(raw)
                )
                / 5,
            }
        )
        for transition in transitions(raw):
            source_start = (transition.source - 1) * class_size
            target_start = (transition.target - 1) * class_size
            indices = list(range(source_start, source_start + class_size)) + list(
                range(target_start, target_start + class_size)
            )
            transition_null = cka_permutation_baseline(
                proxy[indices],
                target[indices],
                permutation_count=args.cka_permutations,
                seed=args.seed + int(transition.transition_id[1:]),
            )
            null_headroom = 1.0 - transition_null.null_mean
            result = transfer[pair_id, transition.transition_id]
            source_cka = class_cka[transition.source]
            target_cka = class_cka[transition.target]
            proxy_source_geometry = proxy_class_geometry[transition.source - 1]
            proxy_target_geometry = proxy_class_geometry[transition.target - 1]
            target_source_geometry = target_class_geometry[transition.source - 1]
            target_target_geometry = target_class_geometry[transition.target - 1]
            proxy_projected = float(
                proxy_distance[transition.source - 1, transition.target - 1]
            )
            target_projected = float(
                target_distance[transition.source - 1, transition.target - 1]
            )
            metric_rows.append(
                {
                    "pair_id": pair_id,
                    "transition_id": transition.transition_id,
                    "source_label": transition.source,
                    "source_name": names[transition.source - 1],
                    "target_label": transition.target,
                    "target_name": names[transition.target - 1],
                    "proxy_selected_distance": selected_distances[
                        pair_id, transition.transition_id
                    ],
                    "proxy_projected_distance": proxy_projected,
                    "target_projected_distance": target_projected,
                    "prototype_distance_abs_gap": abs(
                        proxy_projected - target_projected
                    ),
                    "prototype_distance_ratio_target_proxy": (
                        target_projected / proxy_projected
                        if proxy_projected > 0
                        else float("nan")
                    ),
                    "proxy_distance_percentile": distance_percentile(
                        proxy_distance,
                        transition.source - 1,
                        transition.target - 1,
                    ),
                    "target_distance_percentile": distance_percentile(
                        target_distance,
                        transition.source - 1,
                        transition.target - 1,
                    ),
                    "proxy_source_dispersion": proxy_source_geometry["dispersion"],
                    "proxy_target_dispersion": proxy_target_geometry["dispersion"],
                    "target_source_dispersion": target_source_geometry["dispersion"],
                    "target_target_dispersion": target_target_geometry["dispersion"],
                    "proxy_source_effective_rank": proxy_source_geometry[
                        "effective_rank"
                    ],
                    "proxy_target_effective_rank": proxy_target_geometry[
                        "effective_rank"
                    ],
                    "target_source_effective_rank": target_source_geometry[
                        "effective_rank"
                    ],
                    "target_target_effective_rank": target_target_geometry[
                        "effective_rank"
                    ],
                    "proxy_normalized_separation": proxy_projected
                    / max(
                        1e-12,
                        (
                            proxy_source_geometry["dispersion"]
                            + proxy_target_geometry["dispersion"]
                        )
                        / 2,
                    ),
                    "target_normalized_separation": target_projected
                    / max(
                        1e-12,
                        (
                            target_source_geometry["dispersion"]
                            + target_target_geometry["dispersion"]
                        )
                        / 2,
                    ),
                    "source_class_cka": source_cka.true_cka,
                    "source_class_cka_normalized": (
                        (source_cka.true_cka - source_cka.null_mean)
                        / (1.0 - source_cka.null_mean)
                    ),
                    "target_class_cka": target_cka.true_cka,
                    "target_class_cka_normalized": (
                        (target_cka.true_cka - target_cka.null_mean)
                        / (1.0 - target_cka.null_mean)
                    ),
                    "transition_cka": transition_null.true_cka,
                    "transition_null_mean": transition_null.null_mean,
                    "transition_cka_normalized": (
                        (transition_null.true_cka - transition_null.null_mean)
                        / null_headroom
                        if null_headroom > 0
                        else float("nan")
                    ),
                    "transition_cka_z_score": transition_null.z_score,
                    "transition_cka_empirical_p": (
                        transition_null.empirical_p_value
                    ),
                    "transition_rsa": rsa_correlation(
                        proxy[indices], target[indices]
                    ),
                    "proxy_hits": int(result["proxy_hits"]),
                    "tasr_hits": int(result["tasr_hits"]),
                    "tasr_percent": float(result["tasr_percent"]),
                    "asr_hits": int(result["asr_hits"]),
                    "asr_percent": float(result["asr_percent"]),
                    "conditional_tasr_percent": float(
                        result["conditional_tasr_percent"]
                    ),
                    "semantic_gap_gain": float(result["mean_semantic_gap_gain"]),
                }
            )
        del proxy, target
        torch.cuda.empty_cache()
    augment_perturbation_metrics(metric_rows, output_dir, raw)
    write_csv(result_dir / "transition_metrics.csv", metric_rows)
    write_csv(result_dir / "pair_metrics.csv", pair_rows)

    metrics = (
        "proxy_selected_distance",
        "proxy_projected_distance",
        "target_projected_distance",
        "prototype_distance_abs_gap",
        "prototype_distance_ratio_target_proxy",
        "proxy_distance_percentile",
        "target_distance_percentile",
        "proxy_source_dispersion",
        "proxy_target_dispersion",
        "target_source_dispersion",
        "target_target_dispersion",
        "proxy_source_effective_rank",
        "proxy_target_effective_rank",
        "target_source_effective_rank",
        "target_target_effective_rank",
        "proxy_normalized_separation",
        "target_normalized_separation",
        "source_class_cka",
        "source_class_cka_normalized",
        "target_class_cka",
        "target_class_cka_normalized",
        "transition_cka",
        "transition_cka_normalized",
        "transition_rsa",
        "perturbation_mean_abs",
        "perturbation_rms",
        "perturbation_l2",
        "perturbation_linf",
        "perturbation_nonzero_fraction",
        "perturbation_saturation_fraction",
        "perturbation_total_variation",
        "target_output_entropy_normalized",
        "target_output_distinct_labels",
        "proxy_mean_target_probability",
        "proxy_min_target_margin_across_batches",
        "mean_effective_lambda",
    )
    outcomes = (
        "tasr_percent",
        "asr_percent",
        "conditional_tasr_percent",
        "semantic_gap_gain",
    )
    correlation_rows = []
    for metric in metrics:
        for outcome in outcomes:
            for pair_id in pair_specs(raw):
                group = [row for row in metric_rows if row["pair_id"] == pair_id]
                left = [float(row[metric]) for row in group]
                right = [float(row[outcome]) for row in group]
                value = spearman(left, right)
                correlation_rows.append(
                    {
                        "scope": pair_id,
                        "metric": metric,
                        "outcome": outcome,
                        "cell_count": len(group),
                        "spearman": value,
                        "permutation_p": permutation_p_value(
                            left,
                            right,
                            value,
                            permutations=args.correlation_permutations,
                            seed=args.seed,
                        ),
                    }
                )
            value, p_value = stratified_rank_correlation(
                metric_rows,
                metric,
                outcome,
                permutations=args.correlation_permutations,
                seed=args.seed,
            )
            correlation_rows.append(
                {
                    "scope": "within_pair_stratified",
                    "metric": metric,
                    "outcome": outcome,
                    "cell_count": len(metric_rows),
                    "spearman": value,
                    "permutation_p": p_value,
                }
            )
    write_csv(result_dir / "correlations.csv", correlation_rows)
    summary = {
        "status": "complete",
        "attack_result": str(
            output_dir / "summaries" / str(raw["summary_filename"])
        ),
        "reference_images_per_class": class_size,
        "total_reference_images": len(paths),
        "cka_permutations": args.cka_permutations,
        "correlation_permutations": args.correlation_permutations,
        "target_usage": "post_hoc_only_after_frozen_attack_evaluation",
        "pair_metrics": pair_rows,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(pair_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
