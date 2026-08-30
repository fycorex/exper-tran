#!/usr/bin/env python3
"""Evaluate frozen attack checkpoints and flatten their gradient traces."""

import argparse
import csv
import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch

from common import classification_prompt, load_experiment

from primary_ml_cka.artifacts.writers import write_json
from primary_ml_cka.evaluation.attack_metrics import attack_rates
from primary_ml_cka.models.backends.target_transformers_generation import (
    load_target_for_generation,
)
from primary_ml_cka.models.backends.transformers_backend import load_processor
from primary_ml_cka.models.common.loading import local_snapshot
from primary_ml_cka.models.targets.generation import TransformersTargetGenerator

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path("outputs/pull_push_multiclass_v4_reserve8_diverse10")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def artifact_dir(output_dir: Path, state: dict) -> Path:
    return (
        output_dir
        / "attacks"
        / state["pair_id"]
        / f"v4_{state['transition_id']}_{state['steps']}steps"
        / "batch_00"
        / state["arm"]
        / "lambda_1"
    )


def state_paths(output_dir: Path, pair_id: str, arms: set[str]) -> tuple[Path, ...]:
    paths = []
    for path in sorted((output_dir / "states" / pair_id).glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("arm") in arms:
            paths.append(path)
    return tuple(paths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for checkpoint evaluation")
    raw = load_experiment(EXPERIMENT_ROOT / "config" / "primary.yaml")
    prompt = classification_prompt(raw)
    paths = state_paths(args.output_dir, args.pair, set(args.arms))
    if not paths:
        raise RuntimeError("No matching diagnostic states")
    model = processor = generator = None
    checkpoint_rows = []
    gradient_rows = []
    try:
        snapshot = local_snapshot(Path(".hf-cache"), args.model_id)
        processor = load_processor(snapshot)
        model = load_target_for_generation(snapshot, torch.device("cuda"))
        generator = TransformersTargetGenerator(model, processor)
        for path in paths:
            state = json.loads(path.read_text(encoding="utf-8"))
            if state.get("status") != "complete":
                raise RuntimeError(f"Incomplete diagnostic state: {path}")
            attack = state["attack"]
            artifacts = artifact_dir(args.output_dir, state)
            checkpoint_summary = json.loads(
                (artifacts / "checkpoint_summary.json").read_text(encoding="utf-8")
            )
            count = int(attack["proxy_target_hit_denominator"])
            clean_outputs = tuple(
                generator.generate_label(artifacts / f"{index:02d}_clean.png", prompt)
                for index in range(count)
            )
            evaluated = []
            for checkpoint in checkpoint_summary:
                checkpoint_dir = Path(checkpoint["artifact_dir"])
                adversarial_outputs = tuple(
                    generator.generate_label(
                        checkpoint_dir / f"{index:02d}_adv.png", prompt
                    )
                    for index in range(count)
                )
                rates = attack_rates(
                    tuple(item.parsed_label for item in clean_outputs),
                    tuple(item.parsed_label for item in adversarial_outputs),
                    source_human_label=int(state["source_human_label"]),
                    target_human_label=int(state["target_human_label"]),
                )
                row = {
                    "pair_id": state["pair_id"],
                    "transition_id": state["transition_id"],
                    "arm": state["arm"],
                    "step": checkpoint["step"],
                    "proxy_hits": checkpoint["proxy_hit_count"],
                    "proxy_denominator": count,
                    "targeted_hits": rates.targeted_hit_count,
                    "clean_valid_count": rates.clean_valid_count,
                    "tasr_percent": rates.tasr_percent,
                    "untargeted_hits": rates.untargeted_hit_count,
                    "asr_percent": rates.asr_percent,
                    "linf_png": checkpoint["linf_png"],
                }
                checkpoint_rows.append(row)
                evaluated.append(
                    {
                        **row,
                        "adversarial_outputs": [asdict(item) for item in adversarial_outputs],
                    }
                )
            state["checkpoint_target_evaluations"] = evaluated
            write_json(path, state)
            for trace in attack.get("gradient_trace", ()):
                row = {
                    "pair_id": state["pair_id"],
                    "transition_id": state["transition_id"],
                    "arm": state["arm"],
                    "step": trace["step"],
                    "cls_rep_cosine": trace.get("cls_rep_cosine"),
                }
                for component in ("target_token", "closedset_ce", "margin", "cls_total", "representation"):
                    stats = trace.get(component, {})
                    for metric in ("mean_abs", "rms", "l2_norm", "max_abs", "zero_fraction", "finite"):
                        row[f"{component}_{metric}"] = stats.get(metric)
                gradient_rows.append(row)
    finally:
        if generator is not None:
            del generator
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        torch.cuda.empty_cache()
    diagnostics = args.output_dir / "diagnostics" / "optimization"
    write_csv(diagnostics / f"{args.pair}_checkpoints.csv", checkpoint_rows)
    write_csv(diagnostics / f"{args.pair}_gradients.csv", gradient_rows)
    print(f"evaluated {len(checkpoint_rows)} checkpoints for {args.pair}", flush=True)


if __name__ == "__main__":
    main()
