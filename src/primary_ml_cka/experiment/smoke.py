from primary_ml_cka.artifacts.writers import write_json, write_results_csv
from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.domain.identifiers import MODEL_PAIRS
from primary_ml_cka.evaluation.target_generation import evaluate_local_frozen_batch
from primary_ml_cka.experiment.attack_generation import (
    attack_one_batch,
    blocked_result_row,
    load_phase_records,
    result_row,
)
from primary_ml_cka.experiment.orchestration import (
    CommandContext,
    require_proxy_tap,
    require_real_run_ready,
    resolve_attack_config,
    resolve_data_config,
    resolve_smoke_config,
)
from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT


def run_smoke(context: CommandContext) -> str:
    require_real_run_ready(context)
    attack_config = resolve_attack_config(context)
    smoke_config = resolve_smoke_config(context, attack_config)
    if context.dry_run:
        return (
            "dry-run: smoke plan is one batch, "
            f"lambdas={smoke_config.lambdas}, steps={smoke_config.steps}, "
            f"seed={smoke_config.seed}"
        )
    pairs = tuple(
        pair for pair in MODEL_PAIRS if context.pair_id is None or pair.pair_id == context.pair_id
    )
    references = read_manifest(
        context.output_dir / "evaluation" / "manifests" / "target_training_references.jsonl"
    )
    rows = []
    summaries = []
    data_config = resolve_data_config(context)
    for pair in pairs:
        try:
            require_proxy_tap(context, pair.proxy_model)
            source = load_phase_records(context.output_dir, pair.target_model, "main")[
                : smoke_config.batch_size
            ]
        except Exception as exc:
            for lambda_cka in smoke_config.lambdas:
                rows.append(
                    blocked_result_row(
                        pair,
                        phase="smoke",
                        seed=smoke_config.seed,
                        steps=smoke_config.steps,
                        error=exc,
                        lambda_cka=lambda_cka,
                    )
                )
            summaries.append(f"{pair.pair_id}: setup BLOCKED {exc!r}")
            continue
        for lambda_cka in smoke_config.lambdas:
            try:
                result = attack_one_batch(
                    pair,
                    project_root=context.project_root,
                    output_dir=context.output_dir,
                    phase="smoke",
                    source_records=source,
                    reference_records=references,
                    source_batch_index=0,
                    lambda_cka=lambda_cka,
                    seed=smoke_config.seed,
                    steps=smoke_config.steps,
                    attack_config=attack_config,
                    data_config=data_config,
                )
                if not result.proxy_target_all_hit:
                    rows.append(
                        result_row(
                            pair,
                            result,
                            smoke_config.seed,
                            smoke_config.steps,
                        )
                    )
                    summaries.append(
                        f"{pair.pair_id} lambda={lambda_cka:g}: "
                        "target evaluation SKIPPED; proxy target "
                        f"{result.proxy_target_hit_count}/"
                        f"{result.proxy_target_hit_denominator}"
                    )
                    continue
                if result.final_total >= result.initial_total:
                    raise RuntimeError(
                        "Smoke total loss did not decrease: "
                        f"{result.initial_total} -> {result.final_total}"
                    )
                artifact_dir = (
                    context.output_dir
                    / "attacks"
                    / pair.pair_id
                    / "smoke"
                    / "batch_00"
                    / f"lambda_{lambda_cka:g}"
                )
                target_evaluation = evaluate_local_frozen_batch(
                    model_id=pair.target_model,
                    hf_home=context.project_root / ".hf-cache",
                    artifact_dir=artifact_dir,
                    image_count=len(result.source_image_ids),
                    prompt=CLASSIFICATION_PROMPT,
                    source_human_label=data_config.source_human_label,
                    target_human_label=data_config.target_human_label,
                )
                write_json(
                    context.output_dir
                    / "evaluation"
                    / f"{pair.pair_id}__smoke_lambda_{lambda_cka:g}_outputs.json",
                    target_evaluation,
                )
                rows.append(
                    result_row(
                        pair,
                        result,
                        smoke_config.seed,
                        smoke_config.steps,
                        target_evaluation.rates,
                    )
                )
                summaries.append(
                    f"{pair.pair_id} lambda={lambda_cka:g}: ok "
                    f"TASR={target_evaluation.rates.targeted_hit_count}/"
                    f"{target_evaluation.rates.clean_valid_count} "
                    f"ASR={target_evaluation.rates.untargeted_hit_count}/"
                    f"{target_evaluation.rates.clean_valid_count}"
                )
            except Exception as exc:
                rows.append(
                    blocked_result_row(
                        pair,
                        phase="smoke",
                        seed=smoke_config.seed,
                        steps=smoke_config.steps,
                        error=exc,
                        lambda_cka=lambda_cka,
                    )
                )
                summaries.append(f"{pair.pair_id} lambda={lambda_cka:g}: BLOCKED {exc!r}")
    write_results_csv(context.output_dir / "summaries" / "smoke_results.csv", rows)
    return "\n".join(summaries)
