#!/usr/bin/env python3
"""Evaluate frozen same-trajectory checkpoints with one black-box target load."""

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch

from primary_ml_cka.artifacts.writers import write_json
from primary_ml_cka.config.loader import load_config
from primary_ml_cka.domain.identifiers import get_pair
from primary_ml_cka.evaluation.attack_metrics import attack_rates
from primary_ml_cka.evaluation.target_generation import evaluate_paths
from primary_ml_cka.models.backends.target_transformers_generation import (
    load_target_for_generation,
)
from primary_ml_cka.models.backends.transformers_backend import load_processor
from primary_ml_cka.models.common.loading import local_snapshot
from primary_ml_cka.models.targets.generation import TransformersTargetGenerator
from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/proxy_selector_semantic_contrastive_v3"),
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU target evaluation is forbidden")
    raw = load_config(args.config)
    pair = get_pair(str(raw["pair_id"]))
    phase = (
        f"semantic_contrastive_v3_steps_{args.steps}_"
        f"{raw['representation_type']}_layer_{raw['representation_layer']}"
    )
    arm = next(item for item in raw["arms"] if str(item["name"]) == args.arm)
    artifact_root = (
        args.output_dir
        / "attacks"
        / pair.pair_id
        / phase
        / "batch_00"
        / args.arm
        / f"lambda_{float(arm['lambda_sem']):g}"
    )
    summary_path = artifact_root / "checkpoint_summary.json"
    checkpoints = json.loads(summary_path.read_text(encoding="utf-8"))
    if not checkpoints:
        raise RuntimeError("No frozen checkpoints were recorded")

    model = None
    try:
        snapshot = local_snapshot(Path(".hf-cache"), pair.target_model)
        processor = load_processor(snapshot)
        model = load_target_for_generation(snapshot, torch.device("cuda"))
        generator = TransformersTargetGenerator(model, processor)
        first_dir = Path(checkpoints[0]["artifact_dir"])
        clean_paths = tuple(first_dir / f"{index:02d}_clean.png" for index in range(8))
        clean_outputs = evaluate_paths(generator, clean_paths, CLASSIFICATION_PROMPT)
        rows = []
        for checkpoint in checkpoints:
            checkpoint_dir = Path(checkpoint["artifact_dir"])
            adversarial_paths = tuple(
                checkpoint_dir / f"{index:02d}_adv.png" for index in range(8)
            )
            adversarial_outputs = evaluate_paths(
                generator, adversarial_paths, CLASSIFICATION_PROMPT
            )
            rates = attack_rates(
                tuple(output.parsed_label for output in clean_outputs),
                tuple(output.parsed_label for output in adversarial_outputs),
                source_human_label=8,
                target_human_label=7,
            )
            rows.append(
                {
                    **checkpoint,
                    "tasr_hits": rates.targeted_hit_count,
                    "tasr_percent": rates.tasr_percent,
                    "asr_hits": rates.untargeted_hit_count,
                    "asr_percent": rates.asr_percent,
                    "clean_valid_count": rates.clean_valid_count,
                    "adversarial_outputs": [asdict(output) for output in adversarial_outputs],
                }
            )
            print(
                f"{pair.pair_id} {args.arm} step={checkpoint['step']} "
                f"proxy={checkpoint['proxy_hit_count']}/8 "
                f"TASR={rates.targeted_hit_count}/8 ASR={rates.untargeted_hit_count}/8",
                flush=True,
            )
    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()

    eligible = [
        row
        for row in rows
        if row["proxy_all_hit"] and row.get("proxy_free_hit_count") == 8
    ]
    write_json(
        artifact_root / "checkpoint_target_evaluation.json",
        {
            "pair_id": pair.pair_id,
            "arm": args.arm,
            "selection_rule": "earliest strict proxy checkpoint; target never selects",
            "selected_step": eligible[0]["step"] if eligible else None,
            "rows": rows,
        },
    )


if __name__ == "__main__":
    main()
