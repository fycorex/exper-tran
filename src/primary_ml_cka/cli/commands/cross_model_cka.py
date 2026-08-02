import gc
from dataclasses import asdict, dataclass

import torch

from primary_ml_cka.artifacts.png import load_png_batch_cuda
from primary_ml_cka.artifacts.writers import write_json
from primary_ml_cka.data.manifests import read_manifest
from primary_ml_cka.evaluation.representation_metrics import cross_model_cka
from primary_ml_cka.experiment.orchestration import (
    CommandContext,
    require_real_run_ready,
    resolve_attack_config,
    resolve_prototype_scan_config,
)
from primary_ml_cka.models.proxies.registry import load_proxy
from primary_ml_cka.models.targets.contrastive import load_clip_target_generator


@dataclass(frozen=True, slots=True)
class CrossModelCKARecord:
    lambda_prototype: float
    image_count: int
    image_ids: tuple[str, ...]
    proxy_model: str
    target_model: str
    proxy_embedding_dimension: int
    target_embedding_dimension: int
    clean_cka: float
    adversarial_cka: float
    adversarial_minus_clean: float
    status: str


def _paths(context: CommandContext, value: float, suffix: str, count: int):
    root = (
        context.output_dir
        / "attacks"
        / "DPROTO01"
        / "prototype_scan"
        / "batch_00"
        / f"lambda_{value:g}"
    )
    return tuple(root / f"{index:02d}_{suffix}.png" for index in range(count))


def run(context: CommandContext) -> str:
    require_real_run_ready(context)
    attack_config = resolve_attack_config(context)
    scan_config = resolve_prototype_scan_config(context, attack_config)
    if context.dry_run:
        return "dry-run: cross-model CKA on matched frozen PNG image rows"

    source = read_manifest(
        context.output_dir / "evaluation" / "prototype_transfer" / "source_manifest.jsonl"
    )
    count = len(source)
    if count < 2:
        raise RuntimeError("Cross-model CKA requires at least two source images")

    proxy_batches = []
    proxy = load_proxy(
        scan_config.proxy_model,
        context.project_root / ".hf-cache",
        torch.device("cuda"),
        attack_config,
    )
    try:
        for value in scan_config.lambda_values:
            clean_paths = _paths(context, value, "clean", count)
            adversarial_paths = _paths(context, value, "adv", count)
            if not all(path.is_file() for path in clean_paths + adversarial_paths):
                continue
            clean = load_png_batch_cuda(clean_paths, attack_config.canvas_size)
            adversarial = load_png_batch_cuda(adversarial_paths, attack_config.canvas_size)
            with torch.no_grad():
                proxy_clean = proxy.image_embeddings(clean).embeddings.float()
                proxy_adversarial = proxy.image_embeddings(adversarial).embeddings.float()
            proxy_batches.append((value, proxy_clean, proxy_adversarial))
    finally:
        del proxy
        gc.collect()
        torch.cuda.empty_cache()

    target = load_clip_target_generator(
        scan_config.target_model,
        context.project_root / ".hf-cache",
        attack_config,
    )
    records = []
    try:
        for value, proxy_clean, proxy_adversarial in proxy_batches:
            clean = load_png_batch_cuda(
                _paths(context, value, "clean", count), attack_config.canvas_size
            )
            adversarial = load_png_batch_cuda(
                _paths(context, value, "adv", count), attack_config.canvas_size
            )
            with torch.no_grad():
                target_clean = target.classifier.image_embeddings(clean).embeddings.float()
                target_adversarial = target.classifier.image_embeddings(
                    adversarial
                ).embeddings.float()
            clean_result = cross_model_cka(proxy_clean, target_clean)
            adversarial_result = cross_model_cka(proxy_adversarial, target_adversarial)
            records.append(
                CrossModelCKARecord(
                    lambda_prototype=value,
                    image_count=clean_result.image_count,
                    image_ids=tuple(item.image_id for item in source),
                    proxy_model=scan_config.proxy_model,
                    target_model=scan_config.target_model,
                    proxy_embedding_dimension=(clean_result.proxy_embedding_dimension),
                    target_embedding_dimension=(clean_result.target_embedding_dimension),
                    clean_cka=clean_result.value,
                    adversarial_cka=adversarial_result.value,
                    adversarial_minus_clean=(adversarial_result.value - clean_result.value),
                    status="evaluation_only_target_features_available",
                )
            )
    finally:
        del target
        gc.collect()
        torch.cuda.empty_cache()

    output = context.output_dir / "evaluation" / "prototype_transfer" / "cross_model_cka.json"
    write_json(output, [asdict(record) for record in records])
    return "\n".join(
        f"lambda={record.lambda_prototype:g}: N={record.image_count}, "
        f"clean={record.clean_cka:.6f}, adv={record.adversarial_cka:.6f}, "
        f"delta={record.adversarial_minus_clean:+.6f}"
        for record in records
    )
