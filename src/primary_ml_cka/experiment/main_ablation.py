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


def run_main_ablation(context: CommandContext) -> str:
    require_real_run_ready(context)
    if context.dry_run:
        return "dry-run: main ablation plan validated; no model loaded"
    smoke_path = context.output_dir / "summaries" / "smoke_results.csv"
    if not smoke_path.is_file():
        raise RuntimeError("Smoke results are missing; run `attack smoke` first")
    pairs = tuple(
        pair for pair in MODEL_PAIRS if context.pair_id is None or pair.pair_id == context.pair_id
    )
    references = read_manifest(
        context.output_dir / "evaluation" / "manifests" / "target_training_references.jsonl"
    )
    rows = []
    failures = []
    attack_config = resolve_attack_config(context)
    data_config = resolve_data_config(context)
    batch_size = attack_config.batch_size
    for pair in pairs:
        try:
            require_proxy_tap(context, pair.proxy_model)
            source_records = load_phase_records(context.output_dir, pair.target_model, "main")
            for batch_index in range(len(source_records) // batch_size):
                source = source_records[batch_index * batch_size : (batch_index + 1) * batch_size]
                for lambda_cka in attack_config.lambdas:
                    result = attack_one_batch(
                        pair,
                        project_root=context.project_root,
                        output_dir=context.output_dir,
                        phase="main",
                        source_records=source,
                        reference_records=references,
                        source_batch_index=batch_index,
                        lambda_cka=lambda_cka,
                        seed=context.seed if context.seed is not None else 42,
                        steps=attack_config.steps,
                        attack_config=attack_config,
                        data_config=data_config,
                    )
                    seed = context.seed if context.seed is not None else attack_config.main_seed
                    rows.append(result_row(pair, result, seed, attack_config.steps))
        except Exception as exc:
            seed = context.seed if context.seed is not None else attack_config.main_seed
            rows.append(
                blocked_result_row(
                    pair,
                    phase="main",
                    seed=seed,
                    steps=attack_config.steps,
                    error=exc,
                )
            )
            failures.append(f"{pair.pair_id}: BLOCKED {exc!r}")
    write_results_csv(context.output_dir / "summaries" / "all_results.csv", rows)
    return f"completed_rows={len(rows)}\n" + "\n".join(failures)
