import gc
from dataclasses import dataclass
from pathlib import Path

import torch

from primary_ml_cka.attack.losses.component_gradients import (
    component_gradient_diagnostics,
)
from primary_ml_cka.attack.optimization.momentum_pgd import (
    MomentumPGDState,
    descent_step,
)
from primary_ml_cka.attack.optimization.random_start import shared_random_start
from primary_ml_cka.attack.representation.prototype_contrastive import (
    normalized_prototype,
    prototype_contrastive_loss,
)
from primary_ml_cka.config.schema import AttackConfig, DataConfig, PrototypeScanConfig
from primary_ml_cka.data.manifests import ImageRecord
from primary_ml_cka.domain.labels import human_label_to_index
from primary_ml_cka.evaluation.representation_metrics import representation_metrics
from primary_ml_cka.experiment.attack_generation import _cuda_images, _save_png_batch
from primary_ml_cka.infrastructure.memory import peak_memory, reset_peak_memory
from primary_ml_cka.infrastructure.timing import Timer
from primary_ml_cka.models.common.gradients import assert_parameter_gradients_none
from primary_ml_cka.models.proxies.registry import load_proxy
from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT


@dataclass(frozen=True, slots=True)
class PrototypeAttackResult:
    lambda_prototype: float
    source_image_ids: tuple[str, ...]
    target_reference_ids: tuple[str, ...]
    target_reference_valid_count: int
    proxy_target_hit_count: int
    proxy_target_hit_denominator: int
    proxy_target_probability: float
    proxy_min_text_logit_margin: float
    mean_target_similarity: float
    minimum_target_similarity_gain: float
    mean_source_similarity: float
    minimum_target_source_margin: float
    prototype_attraction: float
    prototype_separation: float
    loss_text: float
    loss_prototype: float
    loss_total: float
    cka_adv_source: float
    cka_adv_reference: float
    target_cka_gain: float
    source_cka_drop: float
    grad_text_l1: float
    grad_prototype_weighted_l1: float
    grad_component_cosine: float
    linf_float: float
    linf_png: float
    elapsed_seconds: float
    peak_allocated_vram_gb: float
    peak_reserved_vram_gb: float
    target_evaluation_eligible: bool
    status: str
    failure_reason: str


@dataclass(frozen=True, slots=True)
class PrototypeEmbeddingBatch:
    lambda_prototype: float
    clean: torch.Tensor
    adversarial: torch.Tensor


@dataclass(frozen=True, slots=True)
class PrototypeScanOutput:
    results: tuple[PrototypeAttackResult, ...]
    embedding_batches: tuple[PrototypeEmbeddingBatch, ...]


def _target_reference_embeddings(
    proxy: object,
    imagenet_root: Path,
    records: tuple[ImageRecord, ...],
    attack_config: AttackConfig,
    data_config: DataConfig,
) -> tuple[torch.Tensor, tuple[ImageRecord, ...]]:
    target_index = human_label_to_index(data_config.target_human_label)
    selected_embeddings = []
    selected_records = []
    for start in range(0, len(records), attack_config.batch_size):
        batch_records = records[start : start + attack_config.batch_size]
        images = _cuda_images(imagenet_root, batch_records, attack_config.canvas_size)
        with torch.no_grad():
            output = proxy.target_loss(
                images,
                data_config.target_human_label,
                CLASSIFICATION_PROMPT,
            )
            embeddings = proxy.image_embeddings(images).embeddings.float()
        if output.class_logits is None:
            raise RuntimeError("Proxy did not return reference class logits")
        valid = output.class_logits.argmax(dim=1) == target_index
        selected_embeddings.append(embeddings[valid])
        selected_records.extend(
            record for record, keep in zip(batch_records, valid.tolist(), strict=True) if keep
        )
        del images
    embeddings = torch.cat(selected_embeddings, dim=0)
    if embeddings.shape[0] < attack_config.batch_size:
        raise RuntimeError(
            "Fewer than one batch of target references is recognized by the proxy: "
            f"{embeddings.shape[0]}"
        )
    return embeddings, tuple(selected_records)


def generate_prototype_scan(
    *,
    project_root: Path,
    output_dir: Path,
    source_records: tuple[ImageRecord, ...],
    reference_records: tuple[ImageRecord, ...],
    attack_config: AttackConfig,
    data_config: DataConfig,
    scan_config: PrototypeScanConfig,
) -> PrototypeScanOutput:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU attack execution is forbidden")
    if len(source_records) != scan_config.batch_size:
        raise ValueError("Prototype scan requires exactly one configured source batch")
    imagenet_root = project_root / "data" / "imagenet_vehicle_official"
    clean = _cuda_images(imagenet_root, source_records, attack_config.canvas_size)
    proxy = load_proxy(
        scan_config.proxy_model,
        project_root / ".hf-cache",
        torch.device("cuda"),
        attack_config,
    )
    reference_embeddings, valid_reference_records = _target_reference_embeddings(
        proxy,
        imagenet_root,
        reference_records,
        attack_config,
        data_config,
    )
    with torch.no_grad():
        clean_embeddings = proxy.image_embeddings(clean).embeddings.detach().float()
    source_prototype = normalized_prototype(clean_embeddings).detach()
    target_prototype = normalized_prototype(reference_embeddings).detach()
    clean_target_similarity = clean_embeddings @ target_prototype
    cka_references = reference_embeddings[: scan_config.batch_size].detach()
    results = []
    embedding_batches = []
    try:
        for lambda_prototype in scan_config.lambda_values:
            reset_peak_memory()
            timer = Timer()
            initial = shared_random_start(
                clean,
                attack_config.epsilon,
                scan_config.seed,
            )
            state = MomentumPGDState(initial, torch.zeros_like(initial))
            diagnostics = None
            for step in range(scan_config.steps):
                text_output = proxy.target_loss(
                    state.adversarial,
                    data_config.target_human_label,
                    CLASSIFICATION_PROMPT,
                )
                adversarial_embeddings = proxy.image_embeddings(state.adversarial).embeddings
                prototype_output = prototype_contrastive_loss(
                    adversarial_embeddings,
                    target_prototype,
                    source_prototype,
                    margin=scan_config.margin,
                    separation_weight=scan_config.separation_weight,
                )
                total = text_output.loss + lambda_prototype * prototype_output.total
                if not torch.isfinite(total):
                    raise RuntimeError(f"Non-finite prototype attack loss at step {step}")
                if step == 0 and lambda_prototype > 0:
                    diagnostics = component_gradient_diagnostics(
                        text_output.loss,
                        prototype_output.total,
                        state.adversarial,
                        lambda_prototype,
                    )
                state = descent_step(
                    total,
                    state,
                    clean,
                    epsilon=attack_config.epsilon,
                    step_size=attack_config.step_size,
                    momentum=attack_config.momentum,
                )

            adversarial = state.adversarial.detach()
            artifact_dir = (
                output_dir
                / "attacks"
                / "DPROTO01"
                / "prototype_scan"
                / "batch_00"
                / f"lambda_{lambda_prototype:g}"
            )
            linf_float, linf_png, adversarial_png = _save_png_batch(
                artifact_dir,
                clean,
                adversarial,
                attack_config.epsilon,
            )
            with torch.no_grad():
                png_text = proxy.target_loss(
                    adversarial_png,
                    data_config.target_human_label,
                    CLASSIFICATION_PROMPT,
                )
                png_embeddings = proxy.image_embeddings(adversarial_png).embeddings.float()
                png_prototype = prototype_contrastive_loss(
                    png_embeddings,
                    target_prototype,
                    source_prototype,
                    margin=scan_config.margin,
                    separation_weight=scan_config.separation_weight,
                )
            if png_text.class_logits is None:
                raise RuntimeError("Frozen PNG proxy logits are missing")
            target_index = human_label_to_index(data_config.target_human_label)
            target_logits = png_text.class_logits[:, target_index]
            other_mask = torch.arange(10, device="cuda") != target_index
            maximum_other = png_text.class_logits[:, other_mask].max(dim=1).values
            text_margins = target_logits - maximum_other
            text_hits = text_margins > 0
            target_gains = png_prototype.target_similarity - clean_target_similarity
            prototype_hits = (target_gains > 0) & (png_prototype.target_source_margin > 0)
            eligible = bool((text_hits & prototype_hits).all().item())
            representation = representation_metrics(
                clean_embeddings,
                png_embeddings,
                cka_references,
            )
            memory = peak_memory()
            zero = 0.0
            failure_reason = ""
            if not eligible:
                failure_reason = (
                    "Frozen PNG gate failed: "
                    f"text={int(text_hits.sum())}/{len(source_records)}, "
                    f"prototype={int(prototype_hits.sum())}/{len(source_records)}"
                )
            results.append(
                PrototypeAttackResult(
                    lambda_prototype=lambda_prototype,
                    source_image_ids=tuple(record.image_id for record in source_records),
                    target_reference_ids=tuple(
                        record.image_id for record in valid_reference_records
                    ),
                    target_reference_valid_count=len(valid_reference_records),
                    proxy_target_hit_count=int(text_hits.sum().item()),
                    proxy_target_hit_denominator=len(source_records),
                    proxy_target_probability=float(png_text.target_probability),
                    proxy_min_text_logit_margin=float(text_margins.min()),
                    mean_target_similarity=float(png_prototype.target_similarity.mean()),
                    minimum_target_similarity_gain=float(target_gains.min()),
                    mean_source_similarity=float(png_prototype.source_similarity.mean()),
                    minimum_target_source_margin=float(png_prototype.target_source_margin.min()),
                    prototype_attraction=float(png_prototype.attraction),
                    prototype_separation=float(png_prototype.separation),
                    loss_text=float(png_text.loss),
                    loss_prototype=float(png_prototype.total),
                    loss_total=float(png_text.loss + lambda_prototype * png_prototype.total),
                    cka_adv_source=representation.cka_adv_source,
                    cka_adv_reference=representation.cka_adv_reference,
                    target_cka_gain=representation.reference_cka_gain,
                    source_cka_drop=representation.source_cka_drop,
                    grad_text_l1=diagnostics.grad_ml_l1 if diagnostics else zero,
                    grad_prototype_weighted_l1=(
                        diagnostics.grad_cka_weighted_l1 if diagnostics else zero
                    ),
                    grad_component_cosine=(diagnostics.cosine if diagnostics else zero),
                    linf_float=linf_float,
                    linf_png=linf_png,
                    elapsed_seconds=timer.elapsed(),
                    peak_allocated_vram_gb=memory.allocated_gb,
                    peak_reserved_vram_gb=memory.reserved_gb,
                    target_evaluation_eligible=eligible,
                    status="ok" if eligible else "prototype_gate_not_reached",
                    failure_reason=failure_reason,
                )
            )
            embedding_batches.append(
                PrototypeEmbeddingBatch(
                    lambda_prototype=lambda_prototype,
                    clean=clean_embeddings.detach(),
                    adversarial=png_embeddings.detach(),
                )
            )
    finally:
        assert_parameter_gradients_none(proxy.model)
        del proxy
        gc.collect()
        torch.cuda.empty_cache()
    return PrototypeScanOutput(tuple(results), tuple(embedding_batches))
