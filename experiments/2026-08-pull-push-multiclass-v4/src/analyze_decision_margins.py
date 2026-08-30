#!/usr/bin/env python3
"""Post-hoc teacher-forced target margin diagnostics for tuned attacks."""

import argparse
import csv
import gc
import json
import math
import statistics
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from analyze_cka_correlations import (
    permutation_p_value,
    spearman,
    stratified_rank_correlation,
    write_csv,
)
from common import classification_prompt, load_experiment, pair_specs, transitions

from primary_ml_cka.domain.identifiers import get_pair
from primary_ml_cka.models.backends.target_transformers_generation import (
    load_target_for_generation,
)
from primary_ml_cka.models.backends.transformers_backend import load_processor
from primary_ml_cka.models.common.loading import local_snapshot
from primary_ml_cka.models.proxies.generative import GenerativeProxy
from primary_ml_cka.models.proxies.visual import (
    gemma_proxy_embeddings,
    gemma_visual_inputs,
    internvl_proxy_embeddings,
    internvl_visual_inputs,
    qwen_proxy_embeddings,
    qwen_visual_inputs,
)

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_ROOT / "config" / "scale50_tuned.yaml"
DEFAULT_OUTPUT = Path("outputs/pull_push_multiclass_v4_scale50_diverse10")


def target_adapter(model_id: str, hf_home: Path) -> GenerativeProxy:
    snapshot = local_snapshot(hf_home, model_id)
    model = load_target_for_generation(snapshot, torch.device("cuda"))
    model.config.use_cache = False
    processor = load_processor(snapshot)
    if model_id.startswith("Qwen/"):
        visual_inputs = qwen_visual_inputs

        def embeddings(images, **kwargs):
            return qwen_proxy_embeddings(model_id, model, images, **kwargs)

        microbatch_size = 3
    elif model_id.startswith("OpenGVLab/InternVL"):
        visual_inputs = internvl_visual_inputs

        def embeddings(images, **kwargs):
            return internvl_proxy_embeddings(
                model_id, model, images, microbatch_size=4, **kwargs
            )

        microbatch_size = 4
    elif model_id.startswith("google/gemma"):

        def visual_inputs(images):
            return gemma_visual_inputs(processor, images)

        def embeddings(images, **kwargs):
            return gemma_proxy_embeddings(model_id, model, processor, images, **kwargs)

        microbatch_size = 1
    else:
        raise ValueError(f"Unsupported generative target: {model_id}")
    return GenerativeProxy(
        model,
        processor,
        visual_inputs,
        embeddings,
        class_margin=2.0,
        margin_weight=1.0,
        margin_temperature=0.5,
        microbatch_size=microbatch_size,
    )


def png_batch(directory: Path, suffix: str, count: int) -> torch.Tensor:
    rows = []
    for index in range(count):
        with Image.open(directory / f"{index:02d}_{suffix}.png") as image:
            rows.append(pil_to_tensor(image.convert("RGB")).float().div(255))
    return torch.stack(rows).cuda()


def class_logits(
    adapter: GenerativeProxy, images: torch.Tensor, target_label: int, prompt: str
) -> torch.Tensor:
    with torch.no_grad():
        output = adapter.target_loss(
            images, target_label, prompt, cls_loss_mode="margin_only"
        )
    if output.class_logits is None:
        raise RuntimeError("Target adapter did not expose closed-set scores")
    return output.class_logits.detach().float().cpu()


def margin_values(
    logits: torch.Tensor, source_label: int, target_label: int
) -> dict[str, torch.Tensor]:
    source_index = source_label - 1
    target_index = target_label - 1
    target = logits[:, target_index]
    source_target = target - logits[:, source_index]
    mask = torch.arange(logits.shape[1]) != target_index
    robust = target - logits[:, mask].max(dim=1).values
    probability = logits.softmax(dim=1)[:, target_index]
    rank = 1 + logits.gt(target.unsqueeze(1)).sum(dim=1)
    entropy = -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)
    return {
        "source_target_margin": source_target,
        "robust_margin": robust,
        "target_probability": probability,
        "target_rank": rank.float(),
        "closedset_entropy": entropy,
    }


def gap_closure(clean_margin: float, margin_change: float) -> float:
    return margin_change / -clean_margin if clean_margin < 0 else float("nan")


def summarize_transition(rows: list[dict]) -> dict:
    result = {}
    for prefix in ("source_target_margin", "robust_margin", "target_probability", "target_rank", "closedset_entropy"):
        clean = [float(row[f"clean_{prefix}"]) for row in rows]
        adversarial = [float(row[f"adversarial_{prefix}"]) for row in rows]
        changes = [right - left for left, right in zip(clean, adversarial, strict=True)]
        result[f"target_clean_{prefix}"] = statistics.fmean(clean)
        result[f"target_adversarial_{prefix}"] = statistics.fmean(adversarial)
        result[f"target_{prefix}_change"] = statistics.fmean(changes)
    robust_closure = [float(row["robust_gap_closure"]) for row in rows]
    source_target_closure = [float(row["source_target_gap_closure"]) for row in rows]
    result["target_mean_robust_gap_closure"] = statistics.fmean(robust_closure)
    result["target_median_robust_gap_closure"] = statistics.median(robust_closure)
    result["target_mean_source_target_gap_closure"] = statistics.fmean(
        source_target_closure
    )
    result["closedset_boundary_crossings"] = sum(
        float(row["adversarial_robust_margin"]) > 0 for row in rows
    )
    return result


def analyze_pair(
    pair_id: str, raw: dict, output_dir: Path, prompt: str, result_dir: Path
) -> None:
    pair = get_pair(pair_id)
    adapter = target_adapter(pair.target_model, Path(".hf-cache"))
    all_rows = []
    try:
        for transition in transitions(raw):
            transition_rows = []
            state_dir = (
                output_dir
                / str(raw["state_namespace"])
                / pair_id
                / transition.transition_id
            )
            for state_path in sorted(state_dir.glob("batch_*.json")):
                state = json.loads(state_path.read_text(encoding="utf-8"))
                count = int(state["source_count"])
                directory = (
                    output_dir
                    / "attacks"
                    / pair_id
                    / str(state["attack"]["phase"])
                    / f"batch_{int(state['batch_index']):02d}"
                    / str(state["objective_tag"])
                    / "lambda_1"
                )
                clean = png_batch(directory, "clean", count)
                adversarial = png_batch(directory, "adv", count)
                clean_values = margin_values(
                    class_logits(adapter, clean, transition.target, prompt),
                    transition.source,
                    transition.target,
                )
                adversarial_values = margin_values(
                    class_logits(adapter, adversarial, transition.target, prompt),
                    transition.source,
                    transition.target,
                )
                for index in range(count):
                    row = {
                        "pair_id": pair_id,
                        "transition_id": transition.transition_id,
                        "batch_index": int(state["batch_index"]),
                        "image_index": index,
                        "generation_target_hit": int(
                            state["target"]["target_hit_mask"][index]
                        ),
                    }
                    for metric in clean_values:
                        row[f"clean_{metric}"] = float(clean_values[metric][index])
                        row[f"adversarial_{metric}"] = float(
                            adversarial_values[metric][index]
                        )
                    for kind in ("source_target", "robust"):
                        clean_margin = row[f"clean_{kind}_margin"]
                        change = (
                            row[f"adversarial_{kind}_margin"] - clean_margin
                        )
                        row[f"{kind}_gap_closure"] = gap_closure(
                            clean_margin, change
                        )
                    transition_rows.append(row)
                del clean, adversarial
            if len(transition_rows) != 50:
                raise RuntimeError(
                    f"Expected 50 decision rows for {pair_id}/{transition.transition_id}"
                )
            all_rows.extend(transition_rows)
            print(
                f"decision complete {pair_id}/{transition.transition_id}", flush=True
            )
    finally:
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
    write_csv(result_dir / f"{pair_id}_per_image.csv", all_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--correlation-permutations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for target decision diagnostics")
    raw = load_experiment(args.config)
    output_dir = args.output_dir.resolve()
    result_dir = output_dir / "diagnostics" / "tuned_transfer_correlations" / "decision"
    result_dir.mkdir(parents=True, exist_ok=True)
    prompt = classification_prompt(raw)
    for pair_id in pair_specs(raw):
        path = result_dir / f"{pair_id}_per_image.csv"
        if args.resume and path.is_file():
            print(f"resume decision pair={pair_id}", flush=True)
            continue
        analyze_pair(pair_id, raw, output_dir, prompt, result_dir)
    transition_rows = []
    transfer_path = result_dir.parent / "transition_metrics.csv"
    with transfer_path.open(newline="", encoding="utf-8") as handle:
        transfer = {
            (row["pair_id"], row["transition_id"]): row
            for row in csv.DictReader(handle)
        }
    for pair_id in pair_specs(raw):
        with (result_dir / f"{pair_id}_per_image.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            per_image = list(csv.DictReader(handle))
        for transition in transitions(raw):
            group = [
                row
                for row in per_image
                if row["transition_id"] == transition.transition_id
            ]
            row = {
                "pair_id": pair_id,
                "transition_id": transition.transition_id,
                **summarize_transition(group),
            }
            base = transfer[pair_id, transition.transition_id]
            for field in (
                "tasr_percent",
                "asr_percent",
                "conditional_tasr_percent",
                "semantic_gap_gain",
            ):
                row[field] = float(base[field])
            transition_rows.append(row)
    write_csv(result_dir / "transition_decision_metrics.csv", transition_rows)
    metrics = tuple(
        key
        for key in transition_rows[0]
        if key.startswith("target_") or key == "closedset_boundary_crossings"
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
                group = [row for row in transition_rows if row["pair_id"] == pair_id]
                left = [float(row[metric]) for row in group]
                right = [float(row[outcome]) for row in group]
                observed = spearman(left, right)
                correlation_rows.append(
                    {
                        "scope": pair_id,
                        "metric": metric,
                        "outcome": outcome,
                        "cell_count": len(group),
                        "spearman": observed,
                        "permutation_p": permutation_p_value(
                            left,
                            right,
                            observed,
                            permutations=args.correlation_permutations,
                            seed=args.seed,
                        ),
                    }
                )
            observed, p_value = stratified_rank_correlation(
                transition_rows,
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
                    "cell_count": len(transition_rows),
                    "spearman": observed,
                    "permutation_p": p_value,
                }
            )
    write_csv(result_dir / "decision_correlations.csv", correlation_rows)


if __name__ == "__main__":
    main()
