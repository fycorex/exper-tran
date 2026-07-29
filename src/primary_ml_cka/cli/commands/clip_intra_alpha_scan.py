import gc
from dataclasses import asdict

import torch

from primary_ml_cka.artifacts.writers import write_json, write_results_csv
from primary_ml_cka.cli.commands.clip_intra_scan import PAIR, TARGET_MODEL_ID
from primary_ml_cka.data.manifests import read_manifest
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
    resolve_alpha_scan_config,
    resolve_attack_config,
    resolve_data_config,
)
from primary_ml_cka.models.targets.contrastive import load_clip_target_generator
from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT


def _artifact_dir(context: CommandContext, alpha: float):
    return (
        context.output_dir
        / "attacks"
        / PAIR.pair_id
        / "clip_intra_alpha_scan"
        / "batch_00"
        / f"alpha_{alpha:g}"
        / "lambda_1"
    )


def run(context: CommandContext) -> str:
    require_real_run_ready(context)
    require_proxy_tap(context, PAIR.proxy_model)
    attack_config = resolve_attack_config(context)
    data_config = resolve_data_config(context)
    scan_config = resolve_alpha_scan_config(context, attack_config)
    if context.dry_run:
        return (
            "dry-run: CLIP intra-family target-CKA weight scan; "
            f"lambda={scan_config.lambda_cka:g}, alphas={scan_config.alphas}, "
            f"steps={scan_config.steps}, seed={scan_config.seed}"
        )

    diagnostic_dir = context.output_dir / "evaluation" / "clip_intra"
    source = read_manifest(diagnostic_dir / "scan_source_manifest.jsonl")
    if len(source) < attack_config.batch_size:
        raise RuntimeError("Run `diagnostics clip-intra-scan` before alpha scan")
    source = source[: attack_config.batch_size]
    references = read_manifest(
        context.output_dir / "evaluation" / "manifests" / "target_training_references.jsonl"
    )

    completed = {}
    rows = []
    summaries = []
    for alpha in scan_config.alphas:
        try:
            result = attack_one_batch(
                PAIR,
                project_root=context.project_root,
                output_dir=context.output_dir,
                phase="clip_intra_alpha_scan",
                source_records=source,
                reference_records=references,
                source_batch_index=0,
                lambda_cka=scan_config.lambda_cka,
                seed=scan_config.seed,
                steps=scan_config.steps,
                attack_config=attack_config,
                data_config=data_config,
                cka_target_weight=alpha,
                objective_tag=f"alpha_{alpha:g}",
            )
            completed[alpha] = result
        except Exception as exc:
            rows.append(
                blocked_result_row(
                    PAIR,
                    phase="clip_intra_alpha_scan",
                    seed=scan_config.seed,
                    steps=scan_config.steps,
                    error=exc,
                    lambda_cka=scan_config.lambda_cka,
                )
            )
            summaries.append(f"alpha={alpha:g}: BLOCKED {exc!r}")

    target = load_clip_target_generator(
        TARGET_MODEL_ID,
        context.project_root / ".hf-cache",
        attack_config,
    )
    raw_results = []
    try:
        for alpha in scan_config.alphas:
            result = completed.get(alpha)
            if result is None:
                continue
            if not result.proxy_target_all_hit:
                rows.append(
                    result_row(
                        PAIR,
                        result,
                        scan_config.seed,
                        scan_config.steps,
                    )
                )
                summaries.append(
                    f"alpha={alpha:g}: target SKIPPED; "
                    f"proxy={result.proxy_target_hit_count}/"
                    f"{result.proxy_target_hit_denominator}"
                )
                continue
            artifact_dir = _artifact_dir(context, alpha)
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
                    scan_config.seed,
                    scan_config.steps,
                    rates,
                )
            )
            raw_results.append(
                {
                    "alpha": alpha,
                    "rates": asdict(rates),
                    "clean_outputs": [asdict(output) for output in clean_outputs],
                    "adversarial_outputs": [asdict(output) for output in adversarial_outputs],
                }
            )
            summaries.append(
                f"alpha={alpha:g}: "
                f"TASR={rates.targeted_hit_count}/{rates.clean_valid_count} "
                f"ASR={rates.untargeted_hit_count}/{rates.clean_valid_count}"
            )
    finally:
        del target
        gc.collect()
        torch.cuda.empty_cache()

    write_results_csv(diagnostic_dir / "alpha_scan_100_results.csv", rows)
    write_json(diagnostic_dir / "alpha_scan_100_raw_results.json", raw_results)
    return "\n".join(summaries)
