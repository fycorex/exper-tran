#!/usr/bin/env python3
"""Run the selected pull+push recipe on 50 common-clean images per cell."""

import argparse
import csv
import gc
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from common import classification_prompt, load_experiment, pair_specs, transition_dir, transitions
from scale50_helpers import batch_slices, conditional_hits

from primary_ml_cka.artifacts.writers import write_json
from primary_ml_cka.config.schema import AttackConfig, DataConfig
from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.domain.identifiers import get_pair
from primary_ml_cka.evaluation.attack_metrics import attack_rates
from primary_ml_cka.experiment.attack_generation import attack_one_batch
from primary_ml_cka.models.backends.target_transformers_generation import (
    load_target_for_generation,
)
from primary_ml_cka.models.backends.transformers_backend import load_processor
from primary_ml_cka.models.common.loading import local_snapshot
from primary_ml_cka.models.targets.generation import TransformersTargetGenerator

EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = EXPERIMENT_ROOT / "config" / "scale50.yaml"
DEFAULT_OUTPUT = Path("outputs/pull_push_multiclass_v4_scale50_diverse10")
OBJECTIVE_TAG = "selected_pull_push"


def run_setting(raw: dict, key: str, default: str) -> str:
    """Return a safe single-path-component run setting."""
    value = str(raw.get(key, default))
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{key} must be a single non-empty path component")
    return value


def state_path(
    output_dir: Path,
    pair_id: str,
    transition_id: str,
    batch: int,
    state_namespace: str = "states_scale50",
) -> Path:
    return output_dir / state_namespace / pair_id / transition_id / f"batch_{batch:02d}.json"


def artifact_dir(
    output_dir: Path,
    pair_id: str,
    transition_id: str,
    batch: int,
    objective_tag: str = OBJECTIVE_TAG,
) -> Path:
    return (
        output_dir
        / "attacks"
        / pair_id
        / f"v4_scale50_{transition_id}"
        / f"batch_{batch:02d}"
        / objective_tag
        / "lambda_1"
    )


def read_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_common_cohorts(output_dir: Path, pair_ids: tuple[str, ...], raw: dict) -> None:
    expected = int(raw["attack_count"])
    for transition in transitions(raw):
        cohorts = tuple(
            read_manifest(
                transition_dir(output_dir, transition.transition_id)
                / f"{pair_id}_attack_images.jsonl"
            )
            for pair_id in pair_ids
        )
        if any(len(cohort) != expected for cohort in cohorts):
            raise RuntimeError(f"{transition.transition_id} does not have {expected} images")
        ids = tuple(tuple(record.image_id for record in cohort) for cohort in cohorts)
        if any(item != ids[0] for item in ids[1:]):
            raise RuntimeError(f"{transition.transition_id} is not common across all pairs")


def attack_batches(
    *,
    pair_id: str,
    transition,
    spec: dict,
    source: tuple,
    class_references: tuple,
    raw: dict,
    output_dir: Path,
    prompt: str,
    resume: bool,
) -> None:
    pair = get_pair(pair_id)
    reference_count = int(raw["reference_count"])
    source_references = class_references[transition.source - 1]
    target_references = class_references[transition.target - 1]
    attacked_ids = {record.image_id for record in source}
    reference_ids = {
        record.image_id for record in source_references + target_references
    }
    if attacked_ids & reference_ids:
        raise RuntimeError(f"{pair_id}/{transition.transition_id} references overlap attacks")
    steps = int(spec.get("steps", raw.get("steps", 50)))
    step_size = float(spec.get("step_size", raw.get("step_size", 1 / 255)))
    semantic_temperature = float(
        spec.get("semantic_temperature", raw["semantic_temperature"])
    )
    early_stop_proxy_gate = bool(spec.get("early_stop_proxy_gate", False))
    state_namespace = run_setting(raw, "state_namespace", "states_scale50")
    objective_tag = run_setting(raw, "objective_tag", OBJECTIVE_TAG)
    attack_config = AttackConfig(
        epsilon=16 / 255,
        step_size=step_size,
        batch_size=8,
        steps=steps,
        momentum=1.0,
        random_start=True,
        class_margin=2.0,
        margin_weight=1.0,
        margin_temperature=0.5,
        proxy_probability_threshold=0.9,
        require_proxy_free_generation=True,
        reference_bank_size=reference_count,
        cls_loss_mode="margin_only",
        semantic_mode="prototype",
        semantic_temperature=semantic_temperature,
        representation_type=str(spec["representation_type"]),
        representation_layer=int(spec["representation_layer"]),
        representation_pooling=str(spec["pooling"]),
    )
    data_config = DataConfig(
        transition.source,
        transition.target,
        int(raw["candidate_count"]),
        reference_count,
        int(raw["attack_count"]),
        0,
        candidate_split=str(raw["candidate_split"]),
        allow_partial_main_batch=True,
        source_reference_count=reference_count,
    )
    for batch_index, (start, stop) in enumerate(batch_slices(len(source))):
        path = state_path(
            output_dir,
            pair_id,
            transition.transition_id,
            batch_index,
            state_namespace,
        )
        if resume and path.is_file() and read_state(path).get("status") in {
            "attack_complete",
            "complete",
        }:
            print(
                f"resume attack {pair_id}/{transition.transition_id} "
                f"batch={batch_index:02d}",
                flush=True,
            )
            continue
        batch = source[start:stop]
        state = {
            "status": "running",
            "pair_id": pair_id,
            "transition_id": transition.transition_id,
            "batch_index": batch_index,
            "seed": int(raw["seed"]) + batch_index,
            "steps": steps,
            "step_size": step_size,
            "rho": float(spec["selected_rho"]),
            "target_logit_weight": float(spec["target_logit_weight"]),
            "source_logit_weight": float(spec["source_logit_weight"]),
            "semantic_temperature": semantic_temperature,
            "representation_layer": int(spec["representation_layer"]),
            "early_stop_proxy_gate": early_stop_proxy_gate,
            "objective_tag": objective_tag,
            "reference_batch_index": 0,
            "source_count": len(batch),
        }
        write_json(path, state)
        try:
            result = attack_one_batch(
                pair,
                project_root=Path.cwd(),
                output_dir=output_dir,
                phase=f"v4_scale50_{transition.transition_id}",
                source_records=batch,
                reference_records=target_references,
                source_reference_records=source_references,
                source_batch_index=batch_index,
                reference_batch_index=0,
                lambda_cka=1.0,
                seed=int(raw["seed"]) + batch_index,
                steps=steps,
                attack_config=attack_config,
                data_config=data_config,
                reference_bank_size=reference_count,
                cka_source_weight=0.0,
                cka_target_weight=0.0,
                semantic_target_weight=1.0,
                gradient_ratio=float(spec["selected_rho"]),
                objective_tag=objective_tag,
                early_stop_proxy_gate=early_stop_proxy_gate,
                progress_interval=max(1, min(10, steps)),
                prompt=prompt,
                cls_loss_mode="margin_only",
                lambda_cls=1.0,
                semantic_mode="prototype",
                semantic_temperature=semantic_temperature,
                semantic_target_logit_weight=float(spec["target_logit_weight"]),
                semantic_source_logit_weight=float(spec["source_logit_weight"]),
                representation_type=str(spec["representation_type"]),
                representation_layer=int(spec["representation_layer"]),
                representation_pooling=str(spec["pooling"]),
            )
            if result.proxy_target_hit_denominator != len(batch):
                raise RuntimeError("Proxy denominator does not equal frozen batch size")
            if result.linf_png > attack_config.epsilon + (1 / 255) + 1e-7:
                raise RuntimeError("Frozen PNG exceeds the L-inf budget tolerance")
            state.update({"status": "attack_complete", "attack": asdict(result)})
            write_json(path, state)
        except Exception as exc:
            state.update({"status": "error", "error": repr(exc)})
            write_json(path, state)
            raise
        finally:
            gc.collect()
            torch.cuda.empty_cache()


def evaluate_batches(
    *,
    pair_id: str,
    transition,
    output_dir: Path,
    prompt: str,
    raw: dict,
) -> None:
    pair = get_pair(pair_id)
    state_namespace = run_setting(raw, "state_namespace", "states_scale50")
    objective_tag = run_setting(raw, "objective_tag", OBJECTIVE_TAG)
    paths = tuple(
        sorted((output_dir / state_namespace / pair_id / transition.transition_id).glob("*.json"))
    )
    if len(paths) != 7:
        raise RuntimeError(f"{pair_id}/{transition.transition_id} requires seven batch states")
    pending = [path for path in paths if read_state(path).get("status") != "complete"]
    if not pending:
        return
    model = processor = generator = None
    try:
        snapshot = local_snapshot(Path(".hf-cache"), pair.target_model)
        processor = load_processor(snapshot)
        model = load_target_for_generation(snapshot, torch.device("cuda"))
        generator = TransformersTargetGenerator(model, processor)
        for path in pending:
            state = read_state(path)
            if state.get("status") != "attack_complete":
                raise RuntimeError(f"Cannot evaluate incomplete attack state: {path}")
            batch_index = int(state["batch_index"])
            count = int(state["source_count"])
            artifacts = artifact_dir(
                output_dir,
                pair_id,
                transition.transition_id,
                batch_index,
                objective_tag,
            )
            clean = tuple(
                generator.generate_label(artifacts / f"{index:02d}_clean.png", prompt)
                for index in range(count)
            )
            adversarial = tuple(
                generator.generate_label(artifacts / f"{index:02d}_adv.png", prompt)
                for index in range(count)
            )
            clean_labels = tuple(item.parsed_label for item in clean)
            adversarial_labels = tuple(item.parsed_label for item in adversarial)
            rates = attack_rates(
                clean_labels,
                adversarial_labels,
                source_human_label=transition.source,
                target_human_label=transition.target,
            )
            if rates.clean_valid_count != count:
                raise RuntimeError(
                    f"Frozen clean denominator changed for {pair_id}/"
                    f"{transition.transition_id}/batch{batch_index}"
                )
            target_mask = tuple(
                clean_label == transition.source and adv_label == transition.target
                for clean_label, adv_label in zip(clean_labels, adversarial_labels, strict=True)
            )
            state.update(
                {
                    "status": "complete",
                    "target": {
                        "clean_outputs": [asdict(item) for item in clean],
                        "adversarial_outputs": [asdict(item) for item in adversarial],
                        "rates": asdict(rates),
                        "target_hit_mask": target_mask,
                    },
                }
            )
            write_json(path, state)
            print(
                f"evaluate {pair_id}/{transition.transition_id} batch={batch_index:02d} "
                f"TASR={rates.targeted_hit_count}/{count} "
                f"ASR={rates.untargeted_hit_count}/{count}",
                flush=True,
            )
    finally:
        if generator is not None:
            del generator
        if model is not None:
            del model
        if processor is not None:
            del processor
        gc.collect()
        torch.cuda.empty_cache()


def summarize(output_dir: Path, raw: dict) -> None:
    state_namespace = run_setting(raw, "state_namespace", "states_scale50")
    summary_filename = run_setting(raw, "summary_filename", "scale50_results.csv")
    rows = []
    for pair_id, spec in pair_specs(raw).items():
        for transition in transitions(raw):
            paths = tuple(
                sorted(
                    (output_dir / state_namespace / pair_id / transition.transition_id).glob(
                        "*.json"
                    )
                )
            )
            states = tuple(read_state(path) for path in paths)
            if len(states) != 7 or any(state.get("status") != "complete" for state in states):
                continue
            total = sum(int(state["source_count"]) for state in states)
            clean_valid = sum(int(state["target"]["rates"]["clean_valid_count"]) for state in states)
            if total != 50 or clean_valid != 50:
                raise RuntimeError(f"Invalid denominator for {pair_id}/{transition.transition_id}")
            proxy_masks = tuple(tuple(state["attack"]["proxy_target_hit_mask"]) for state in states)
            target_masks = tuple(tuple(state["target"]["target_hit_mask"]) for state in states)
            conditional_numerator, conditional_denominator = conditional_hits(
                proxy_masks, target_masks
            )
            tasr_hits = sum(int(state["target"]["rates"]["targeted_hit_count"]) for state in states)
            asr_hits = sum(int(state["target"]["rates"]["untargeted_hit_count"]) for state in states)
            proxy_hits = sum(int(state["attack"]["proxy_target_hit_count"]) for state in states)
            rows.append(
                {
                    "pair_id": pair_id,
                    "transition_id": transition.transition_id,
                    "source": transition.source,
                    "target": transition.target,
                    "images": total,
                    "batches": len(states),
                    "rho": spec["selected_rho"],
                    "target_logit_weight": spec["target_logit_weight"],
                    "source_logit_weight": spec["source_logit_weight"],
                    "semantic_temperature": spec.get(
                        "semantic_temperature", raw["semantic_temperature"]
                    ),
                    "representation_layer": spec["representation_layer"],
                    "steps": spec.get("steps", raw.get("steps", 50)),
                    "step_size": spec.get(
                        "step_size", raw.get("step_size", 1 / 255)
                    ),
                    "proxy_hits": proxy_hits,
                    "proxy_percent": 100 * proxy_hits / total,
                    "tasr_hits": tasr_hits,
                    "tasr_percent": 100 * tasr_hits / clean_valid,
                    "asr_hits": asr_hits,
                    "asr_percent": 100 * asr_hits / clean_valid,
                    "target_hits_among_proxy_hits": conditional_numerator,
                    "conditional_denominator": conditional_denominator,
                    "conditional_tasr_percent": (
                        100 * conditional_numerator / conditional_denominator
                        if conditional_denominator
                        else 0.0
                    ),
                    "mean_semantic_gap_gain": sum(
                        float(state["attack"]["semantic_gap_gain"])
                        * int(state["source_count"])
                        for state in states
                    )
                    / total,
                    "elapsed_seconds": sum(
                        float(state["attack"]["elapsed_seconds"]) for state in states
                    ),
                }
            )
    write_csv(output_dir / "summaries" / summary_filename, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pairs", nargs="+")
    parser.add_argument("--transitions", nargs="+")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU attacks are forbidden")
    raw = load_experiment(args.config)
    specs = pair_specs(raw)
    selected_pairs = tuple(args.pairs or specs)
    selected_transitions = tuple(
        item for item in transitions(raw) if args.transitions is None or item.transition_id in args.transitions
    )
    if set(selected_pairs) - set(specs):
        raise ValueError("Unknown pair requested")
    if args.transitions and len(selected_transitions) != len(set(args.transitions)):
        raise ValueError("Unknown transition requested")
    validate_common_cohorts(args.output_dir, tuple(specs), raw)
    prompt = classification_prompt(raw)
    class_references = tuple(
        read_manifest(
            args.output_dir
            / "evaluation"
            / "manifests"
            / f"class_references_{label:02d}.jsonl"
        )
        for label in range(1, 11)
    )
    for pair_id in selected_pairs:
        for transition in selected_transitions:
            source = read_manifest(
                transition_dir(args.output_dir, transition.transition_id)
                / f"{pair_id}_attack_images.jsonl"
            )
            attack_batches(
                pair_id=pair_id,
                transition=transition,
                spec=specs[pair_id],
                source=source,
                class_references=class_references,
                raw=raw,
                output_dir=args.output_dir,
                prompt=prompt,
                resume=args.resume,
            )
            evaluate_batches(
                pair_id=pair_id,
                transition=transition,
                output_dir=args.output_dir,
                prompt=prompt,
                raw=raw,
            )
            summarize(args.output_dir, raw)


if __name__ == "__main__":
    main()
