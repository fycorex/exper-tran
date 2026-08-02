import csv
import gc
import io
from dataclasses import asdict

import torch

from primary_ml_cka.artifacts.png import load_png_batch_cuda
from primary_ml_cka.artifacts.writers import write_json
from primary_ml_cka.data.manifests import read_manifest, write_manifest
from primary_ml_cka.data.selection import clean_valid_split
from primary_ml_cka.evaluation.attack_metrics import attack_rates
from primary_ml_cka.evaluation.representation_metrics import cross_model_cka
from primary_ml_cka.evaluation.target_generation import evaluate_paths
from primary_ml_cka.experiment.orchestration import (
    CommandContext,
    require_proxy_tap,
    require_real_run_ready,
    resolve_attack_config,
    resolve_data_config,
    resolve_prototype_scan_config,
)
from primary_ml_cka.experiment.prototype_transfer import generate_prototype_scan
from primary_ml_cka.infrastructure.atomic_io import atomic_text_write
from primary_ml_cka.models.targets.contrastive import load_clip_target_generator
from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT


def _artifact_dir(context: CommandContext, lambda_prototype: float):
    return (
        context.output_dir
        / "attacks"
        / "DPROTO01"
        / "prototype_scan"
        / "batch_00"
        / f"lambda_{lambda_prototype:g}"
    )


def run(context: CommandContext) -> str:
    require_real_run_ready(context)
    attack_config = resolve_attack_config(context)
    data_config = resolve_data_config(context)
    scan_config = resolve_prototype_scan_config(context, attack_config)
    require_proxy_tap(context, scan_config.proxy_model)
    if context.dry_run:
        return (
            "dry-run: per-image proxy prototype scan; "
            f"lambdas={scan_config.lambda_values}, steps={scan_config.steps}, "
            f"margin={scan_config.margin}, seed={scan_config.seed}"
        )

    manifests = context.output_dir / "evaluation" / "manifests"
    candidates = read_manifest(manifests / "source_validation_candidates.jsonl")
    references = read_manifest(manifests / "target_training_references.jsonl")
    imagenet_root = context.project_root / "data" / "imagenet_vehicle_official"
    output_dir = context.output_dir / "evaluation" / "prototype_transfer"

    target = load_clip_target_generator(
        scan_config.target_model,
        context.project_root / ".hf-cache",
        attack_config,
    )
    try:
        screen_outputs = evaluate_paths(
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
        tuple(output.parsed_label for output in screen_outputs),
        data_config.source_human_label,
        data_config.main_max_count,
        data_config.confirmation_max_count,
        attack_config.batch_size,
    )
    source = main[: scan_config.batch_size]
    if len(source) != scan_config.batch_size:
        raise RuntimeError(f"Target clean screening produced only {len(source)} usable images")
    write_manifest(output_dir / "source_manifest.jsonl", source)
    write_json(
        output_dir / "clean_screen.json",
        {
            "clean_valid_count": sum(
                output.parsed_label == data_config.source_human_label for output in screen_outputs
            ),
            "outputs": [
                {"image_id": record.image_id, **asdict(output)}
                for record, output in zip(candidates, screen_outputs, strict=True)
            ],
        },
    )

    scan_output = generate_prototype_scan(
        project_root=context.project_root,
        output_dir=context.output_dir,
        source_records=source,
        reference_records=references,
        attack_config=attack_config,
        data_config=data_config,
        scan_config=scan_config,
    )
    attack_results = scan_output.results
    proxy_embeddings = {item.lambda_prototype: item for item in scan_output.embedding_batches}

    target = load_clip_target_generator(
        scan_config.target_model,
        context.project_root / ".hf-cache",
        attack_config,
    )
    rows = []
    raw_results = []
    summaries = []
    try:
        for result in attack_results:
            row = asdict(result)
            row.update(
                {
                    "clean_valid_count": "",
                    "targeted_hit_count": "",
                    "tasr_percent": "",
                    "untargeted_hit_count": "",
                    "asr_percent": "",
                    "cross_model_image_count": "",
                    "proxy_embedding_dimension": "",
                    "target_embedding_dimension": "",
                    "cka_proxy_target_clean": "",
                    "cka_proxy_target_adversarial": "",
                    "cka_proxy_target_delta": "",
                }
            )
            if not result.target_evaluation_eligible:
                rows.append(row)
                summaries.append(
                    f"lambda={result.lambda_prototype:g}: target SKIPPED; "
                    f"{result.failure_reason}"
                )
                continue
            artifact_dir = _artifact_dir(context, result.lambda_prototype)
            clean_paths = tuple(
                artifact_dir / f"{index:02d}_clean.png" for index in range(len(source))
            )
            adversarial_paths = tuple(
                artifact_dir / f"{index:02d}_adv.png" for index in range(len(source))
            )
            clean_outputs = evaluate_paths(
                target,
                clean_paths,
                CLASSIFICATION_PROMPT,
            )
            adversarial_outputs = evaluate_paths(
                target,
                adversarial_paths,
                CLASSIFICATION_PROMPT,
            )
            clean_pixels = load_png_batch_cuda(clean_paths, attack_config.canvas_size)
            adversarial_pixels = load_png_batch_cuda(adversarial_paths, attack_config.canvas_size)
            with torch.no_grad():
                target_clean_embeddings = target.classifier.image_embeddings(
                    clean_pixels
                ).embeddings.float()
                target_adversarial_embeddings = target.classifier.image_embeddings(
                    adversarial_pixels
                ).embeddings.float()
            proxy_batch = proxy_embeddings[result.lambda_prototype]
            clean_alignment = cross_model_cka(proxy_batch.clean, target_clean_embeddings)
            adversarial_alignment = cross_model_cka(
                proxy_batch.adversarial, target_adversarial_embeddings
            )
            rates = attack_rates(
                tuple(output.parsed_label for output in clean_outputs),
                tuple(output.parsed_label for output in adversarial_outputs),
                source_human_label=data_config.source_human_label,
                target_human_label=data_config.target_human_label,
            )
            row.update(asdict(rates))
            row.update(
                {
                    "cross_model_image_count": clean_alignment.image_count,
                    "proxy_embedding_dimension": clean_alignment.proxy_embedding_dimension,
                    "target_embedding_dimension": clean_alignment.target_embedding_dimension,
                    "cka_proxy_target_clean": clean_alignment.value,
                    "cka_proxy_target_adversarial": adversarial_alignment.value,
                    "cka_proxy_target_delta": (adversarial_alignment.value - clean_alignment.value),
                }
            )
            rows.append(row)
            raw_results.append(
                {
                    "lambda_prototype": result.lambda_prototype,
                    "rates": asdict(rates),
                    "clean_outputs": [asdict(output) for output in clean_outputs],
                    "adversarial_outputs": [asdict(output) for output in adversarial_outputs],
                }
            )
            summaries.append(
                f"lambda={result.lambda_prototype:g}: "
                f"TASR={rates.targeted_hit_count}/{rates.clean_valid_count} "
                f"ASR={rates.untargeted_hit_count}/{rates.clean_valid_count}"
            )
    finally:
        del target
        gc.collect()
        torch.cuda.empty_cache()

    fieldnames = tuple(rows[0])
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text_write(output_dir / "results.csv", buffer.getvalue())
    write_json(output_dir / "raw_target_outputs.json", raw_results)
    return "\n".join(summaries)
