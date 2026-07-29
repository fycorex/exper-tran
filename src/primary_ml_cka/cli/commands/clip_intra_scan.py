import gc
from dataclasses import asdict

import torch

from primary_ml_cka.artifacts.writers import write_json, write_results_csv
from primary_ml_cka.data.manifests import read_manifest, write_manifest
from primary_ml_cka.data.selection import clean_valid_split
from primary_ml_cka.domain.identifiers import ExperimentType, ModelPair
from primary_ml_cka.evaluation.attack_metrics import attack_rates
from primary_ml_cka.evaluation.target_generation import evaluate_paths
from primary_ml_cka.experiment.attack_generation import (
    attack_one_batch,
    blocked_result_row,
    result_row,
)
from primary_ml_cka.experiment.orchestration import (
    CommandContext,
    require_proxy_tap,
    require_real_run_ready,
    resolve_attack_config,
    resolve_data_config,
)
from primary_ml_cka.models.targets.contrastive import load_clip_target_generator
from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT

PROXY_MODEL_ID = "openai/clip-vit-large-patch14"
TARGET_MODEL_ID = "openai/clip-vit-base-patch32"
PAIR = ModelPair("DCLIP01", ExperimentType.INTRA_FAMILY, PROXY_MODEL_ID, TARGET_MODEL_ID)


def _artifact_dir(context: CommandContext, lambda_cka: float):
    return (
        context.output_dir
        / "attacks"
        / PAIR.pair_id
        / "clip_intra_scan"
        / "batch_00"
        / f"lambda_{lambda_cka:g}"
    )


def run(context: CommandContext) -> str:
    require_real_run_ready(context)
    require_proxy_tap(context, PROXY_MODEL_ID)
    attack_config = resolve_attack_config(context)
    data_config = resolve_data_config(context)
    if context.dry_run:
        return (
            "dry-run: CLIP-L/14 to CLIP-B/32 one-batch scan; "
            f"lambdas={attack_config.lambdas}, steps={attack_config.steps}, "
            f"seed={attack_config.main_seed}"
        )

    manifests = context.output_dir / "evaluation" / "manifests"
    candidates = read_manifest(manifests / "source_validation_candidates.jsonl")
    references = read_manifest(manifests / "target_training_references.jsonl")
    imagenet_root = context.project_root / "data" / "imagenet_vehicle_official"

    target = load_clip_target_generator(
        TARGET_MODEL_ID,
        context.project_root / ".hf-cache",
        attack_config,
    )
    try:
        candidate_outputs = evaluate_paths(
            target,
            tuple(imagenet_root / record.relative_path for record in candidates),
            CLASSIFICATION_PROMPT,
        )
    finally:
        del target
        gc.collect()
        torch.cuda.empty_cache()
    main, _ = clean_valid_split(
        candidates,
        tuple(output.parsed_label for output in candidate_outputs),
        data_config.source_human_label,
        data_config.main_max_count,
        data_config.confirmation_max_count,
        attack_config.batch_size,
    )
    source = main[: attack_config.batch_size]
    if len(source) != attack_config.batch_size:
        raise RuntimeError(f"CLIP-B/32 clean screening produced only {len(source)} usable images")
    diagnostic_dir = context.output_dir / "evaluation" / "clip_intra"
    write_manifest(diagnostic_dir / "scan_source_manifest.jsonl", source)
    write_json(
        diagnostic_dir / "scan_clean_screen.json",
        {
            "clean_valid_count": sum(
                output.parsed_label == data_config.source_human_label
                for output in candidate_outputs
            ),
            "outputs": [
                {"image_id": record.image_id, **asdict(output)}
                for record, output in zip(candidates, candidate_outputs, strict=True)
            ],
        },
    )

    completed = {}
    rows = []
    summaries = []
    for lambda_cka in attack_config.lambdas:
        try:
            result = attack_one_batch(
                PAIR,
                project_root=context.project_root,
                output_dir=context.output_dir,
                phase="clip_intra_scan",
                source_records=source,
                reference_records=references,
                source_batch_index=0,
                lambda_cka=lambda_cka,
                seed=attack_config.main_seed,
                steps=attack_config.steps,
                attack_config=attack_config,
                data_config=data_config,
            )
            completed[lambda_cka] = result
        except Exception as exc:
            rows.append(
                blocked_result_row(
                    PAIR,
                    phase="clip_intra_scan",
                    seed=attack_config.main_seed,
                    steps=attack_config.steps,
                    error=exc,
                    lambda_cka=lambda_cka,
                )
            )
            summaries.append(f"lambda={lambda_cka:g}: BLOCKED {exc!r}")

    target = load_clip_target_generator(
        TARGET_MODEL_ID,
        context.project_root / ".hf-cache",
        attack_config,
    )
    raw_results = []
    try:
        for lambda_cka in attack_config.lambdas:
            result = completed.get(lambda_cka)
            if result is None:
                continue
            if not result.proxy_target_all_hit:
                rows.append(
                    result_row(
                        PAIR,
                        result,
                        attack_config.main_seed,
                        attack_config.steps,
                    )
                )
                summaries.append(
                    f"lambda={lambda_cka:g}: target SKIPPED; "
                    f"proxy={result.proxy_target_hit_count}/"
                    f"{result.proxy_target_hit_denominator}"
                )
                continue
            artifact_dir = _artifact_dir(context, lambda_cka)
            clean_outputs = evaluate_paths(
                target,
                tuple(artifact_dir / f"{index:02d}_clean.png" for index in range(len(source))),
                CLASSIFICATION_PROMPT,
            )
            adversarial_outputs = evaluate_paths(
                target,
                tuple(artifact_dir / f"{index:02d}_adv.png" for index in range(len(source))),
                CLASSIFICATION_PROMPT,
            )
            rates = attack_rates(
                tuple(output.parsed_label for output in clean_outputs),
                tuple(output.parsed_label for output in adversarial_outputs),
                source_human_label=data_config.source_human_label,
                target_human_label=data_config.target_human_label,
            )
            rows.append(
                result_row(
                    PAIR,
                    result,
                    attack_config.main_seed,
                    attack_config.steps,
                    rates,
                )
            )
            raw_results.append(
                {
                    "lambda": lambda_cka,
                    "rates": asdict(rates),
                    "clean_outputs": [asdict(output) for output in clean_outputs],
                    "adversarial_outputs": [asdict(output) for output in adversarial_outputs],
                }
            )
            summaries.append(
                f"lambda={lambda_cka:g}: "
                f"TASR={rates.targeted_hit_count}/{rates.clean_valid_count} "
                f"ASR={rates.untargeted_hit_count}/{rates.clean_valid_count}"
            )
    finally:
        del target
        gc.collect()
        torch.cuda.empty_cache()

    write_results_csv(diagnostic_dir / "scan_100_results.csv", rows)
    write_json(diagnostic_dir / "scan_100_raw_results.json", raw_results)
    return "\n".join(summaries)
