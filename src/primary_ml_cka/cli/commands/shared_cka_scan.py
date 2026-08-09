import csv
import gc
import io
from dataclasses import asdict, replace

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
    resolve_shared_cka_scan_config,
)
from primary_ml_cka.experiment.prototype_transfer import (
    generate_prototype_scan,
    prototype_artifact_dir,
)
from primary_ml_cka.infrastructure.atomic_io import atomic_text_write
from primary_ml_cka.models.targets.contrastive import load_clip_target_generator
from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT

OBJECTIVE_TAG = "su_cka_scan"


def run(context: CommandContext) -> str:
    require_real_run_ready(context)
    attack_config = resolve_attack_config(context)
    data_config = resolve_data_config(context)
    prototype_config = resolve_prototype_scan_config(context, attack_config)
    shared_config = resolve_shared_cka_scan_config(context, attack_config)
    require_proxy_tap(context, prototype_config.proxy_model)
    if context.dry_run:
        return (
            "dry-run: proxy shared-geometry CKA scan; "
            f"weights={shared_config.shared_clean_weights}, "
            f"view_weight={shared_config.view_consistency_weight}"
        )

    manifest_root = context.output_dir / "evaluation" / "manifests"
    candidates = read_manifest(manifest_root / "source_validation_candidates.jsonl")
    references = read_manifest(manifest_root / "target_training_references.jsonl")
    imagenet_root = context.project_root / "data" / "imagenet_vehicle_official"
    output_dir = context.output_dir / "evaluation" / "su_cka_transfer"

    target = load_clip_target_generator(
        prototype_config.target_model,
        context.project_root / ".hf-cache",
        attack_config,
    )
    try:
        screen_outputs = evaluate_paths(
            target,
            tuple(imagenet_root / item.relative_path for item in candidates),
            CLASSIFICATION_PROMPT,
        )
    finally:
        del target
        gc.collect()
        torch.cuda.empty_cache()
    main, _ = clean_valid_split(
        candidates,
        tuple(item.parsed_label for item in screen_outputs),
        data_config.source_human_label,
        data_config.main_max_count,
        data_config.confirmation_max_count,
        attack_config.batch_size,
    )
    source = main[: shared_config.batch_size]
    if len(source) != shared_config.batch_size:
        raise RuntimeError(f"Only {len(source)} clean-correct source images are available")
    write_manifest(output_dir / "source_manifest.jsonl", source)

    scan_output = generate_prototype_scan(
        project_root=context.project_root,
        output_dir=context.output_dir,
        source_records=source,
        reference_records=references,
        attack_config=attack_config,
        data_config=data_config,
        scan_config=replace(
            prototype_config,
            lambda_values=(shared_config.prototype_lambda,),
            steps=shared_config.steps,
            seed=shared_config.seed,
            batch_size=shared_config.batch_size,
        ),
        shared_clean_weights=shared_config.shared_clean_weights,
        clean_separation_weight=shared_config.clean_separation_weight,
        view_consistency_weight=shared_config.view_consistency_weight,
        view_scales=shared_config.view_scales,
        objective_tag=OBJECTIVE_TAG,
    )
    embedding_batches = {
        item.shared_clean_weight: item for item in scan_output.embedding_batches
    }

    rows = []
    raw_outputs = []
    summaries = []
    target = load_clip_target_generator(
        prototype_config.target_model,
        context.project_root / ".hf-cache",
        attack_config,
    )
    try:
        for result in scan_output.results:
            row = asdict(result)
            row.update(
                {
                    "clean_valid_count": "",
                    "targeted_hit_count": "",
                    "tasr_percent": "",
                    "untargeted_hit_count": "",
                    "asr_percent": "",
                    "cross_model_image_count": "",
                    "cka_proxy_target_clean": "",
                    "cka_proxy_target_adversarial": "",
                    "cka_proxy_target_delta": "",
                    "cka_proxy_target_state": "",
                }
            )
            if not result.target_evaluation_eligible:
                rows.append(row)
                summaries.append(
                    f"shared={result.shared_clean_weight:g}: target SKIPPED; "
                    f"{result.failure_reason}"
                )
                continue
            artifact_dir = prototype_artifact_dir(
                context.output_dir,
                OBJECTIVE_TAG,
                result.lambda_prototype,
                result.shared_clean_weight,
            )
            clean_paths = tuple(
                artifact_dir / f"{index:02d}_clean.png" for index in range(len(source))
            )
            adversarial_paths = tuple(
                artifact_dir / f"{index:02d}_adv.png" for index in range(len(source))
            )
            clean_outputs = evaluate_paths(target, clean_paths, CLASSIFICATION_PROMPT)
            adversarial_outputs = evaluate_paths(
                target, adversarial_paths, CLASSIFICATION_PROMPT
            )
            clean_pixels = load_png_batch_cuda(clean_paths, attack_config.canvas_size)
            adversarial_pixels = load_png_batch_cuda(
                adversarial_paths, attack_config.canvas_size
            )
            with torch.no_grad():
                target_clean = target.classifier.image_embeddings(
                    clean_pixels
                ).embeddings.float()
                target_adversarial = target.classifier.image_embeddings(
                    adversarial_pixels
                ).embeddings.float()
            proxy_batch = embedding_batches[result.shared_clean_weight]
            clean_cka = cross_model_cka(proxy_batch.clean, target_clean)
            adversarial_cka = cross_model_cka(
                proxy_batch.adversarial, target_adversarial
            )
            state_cka = cross_model_cka(
                torch.cat((proxy_batch.clean, proxy_batch.adversarial)),
                torch.cat((target_clean, target_adversarial)),
            )
            rates = attack_rates(
                tuple(item.parsed_label for item in clean_outputs),
                tuple(item.parsed_label for item in adversarial_outputs),
                source_human_label=data_config.source_human_label,
                target_human_label=data_config.target_human_label,
            )
            row.update(asdict(rates))
            row.update(
                {
                    "cross_model_image_count": clean_cka.image_count,
                    "cka_proxy_target_clean": clean_cka.value,
                    "cka_proxy_target_adversarial": adversarial_cka.value,
                    "cka_proxy_target_delta": adversarial_cka.value - clean_cka.value,
                    "cka_proxy_target_state": state_cka.value,
                }
            )
            rows.append(row)
            raw_outputs.append(
                {
                    "shared_clean_weight": result.shared_clean_weight,
                    "rates": asdict(rates),
                    "clean_outputs": [asdict(item) for item in clean_outputs],
                    "adversarial_outputs": [asdict(item) for item in adversarial_outputs],
                }
            )
            summaries.append(
                f"shared={result.shared_clean_weight:g}: "
                f"CKA={adversarial_cka.value:.6f}, "
                f"TASR={rates.targeted_hit_count}/{rates.clean_valid_count}, "
                f"ASR={rates.untargeted_hit_count}/{rates.clean_valid_count}"
            )
    finally:
        del target
        gc.collect()
        torch.cuda.empty_cache()

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=tuple(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_text_write(output_dir / "results.csv", buffer.getvalue())
    write_json(output_dir / "raw_target_outputs.json", raw_outputs)
    return "\n".join(summaries)
