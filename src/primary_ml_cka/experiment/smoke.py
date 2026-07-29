from primary_ml_cka.artifacts.writers import write_results_csv
from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.domain.identifiers import MODEL_PAIRS
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
)


def run_smoke(context: CommandContext) -> str:
    require_real_run_ready(context)
    if context.dry_run:
        return "dry-run: smoke plan is one batch, lambda=1, 10 steps, seed=42"
    pairs = tuple(
        pair for pair in MODEL_PAIRS if context.pair_id is None or pair.pair_id == context.pair_id
    )
    references = read_manifest(
        context.output_dir / "evaluation" / "manifests" / "target_training_references.jsonl"
    )
    rows = []
    summaries = []
    attack_config = resolve_attack_config(context)
    data_config = resolve_data_config(context)
    for pair in pairs:
        try:
            require_proxy_tap(context, pair.proxy_model)
            source = load_phase_records(context.output_dir, pair.target_model, "main")[
                : attack_config.batch_size
            ]
            result = attack_one_batch(
                pair,
                project_root=context.project_root,
                output_dir=context.output_dir,
                phase="smoke",
                source_records=source,
                reference_records=references,
                source_batch_index=0,
                lambda_cka=1.0,
                seed=42,
                steps=10,
                attack_config=attack_config,
                data_config=data_config,
            )
            if result.final_total >= result.initial_total:
                raise RuntimeError(
                    "Smoke total loss did not decrease: "
                    f"{result.initial_total} -> {result.final_total}"
                )
            if result.reference_cka_gain <= 0 or result.source_cka_drop <= 0:
                raise RuntimeError(
                    "Smoke CKA direction failed: "
                    f"reference_gain={result.reference_cka_gain}, "
                    f"source_drop={result.source_cka_drop}"
                )
            rows.append(result_row(pair, result, 42, 10))
            summaries.append(f"{pair.pair_id}: ok")
        except Exception as exc:
            rows.append(blocked_result_row(pair, phase="smoke", seed=42, steps=10, error=exc))
            summaries.append(f"{pair.pair_id}: BLOCKED {exc!r}")
    write_results_csv(context.output_dir / "summaries" / "smoke_results.csv", rows)
    return "\n".join(summaries)
