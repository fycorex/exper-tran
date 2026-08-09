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
from primary_ml_cka.attack.representation.shared_geometry import (
    center_crop_view,
    random_resized_crop_view,
    shared_geometry_loss,
)
from primary_ml_cka.config.schema import AttackConfig, DataConfig, PrototypeScanConfig
from primary_ml_cka.data.manifests import ImageRecord
from primary_ml_cka.domain.labels import human_label_to_index
from primary_ml_cka.evaluation.representation_metrics import representation_metrics
from primary_ml_cka.experiment.attack_generation import _cuda_images, _save_png_batch
from primary_ml_cka.infrastructure.memory import peak_memory, reset_peak_memory
from primary_ml_cka.infrastructure.timing import Timer
from primary_ml_cka.models.common.gradients import assert_parameter_gradients_none
from primary_ml_cka.models.proxies.contrastive import ContrastiveProxy
from primary_ml_cka.models.proxies.registry import load_proxy
from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT


@dataclass(frozen=True, slots=True)
class PrototypeAttackResult:
    lambda_prototype: float
    shared_clean_weight: float
    view_consistency_weight: float
    source_image_ids: tuple[str, ...]
    target_reference_ids: tuple[str, ...]
    target_reference_valid_count: int
    proxy_target_hit_count: int
    proxy_target_hit_denominator: int
    robust_view_hit_count: int
    robust_view_hit_denominator: int
    proxy_target_probability: float
    proxy_min_text_logit_margin: float
    mean_target_similarity: float
    minimum_target_similarity_gain: float
    mean_source_similarity: float
    minimum_target_source_margin: float
    mean_clean_similarity: float
    minimum_target_clean_margin: float
    prototype_attraction: float
    prototype_separation: float
    prototype_clean_separation: float
    loss_text: float
    loss_prototype: float
    loss_shared_geometry: float
    loss_total: float
    cka_adv_clean_optimized: float
    cosine_adv_view: float
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
    shared_clean_weight: float
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
    shared_clean_weights: tuple[float, ...] = (0.0,),
    clean_separation_weight: float = 0.0,
    view_consistency_weight: float = 0.0,
    view_scales: tuple[float, ...] = (0.875, 0.75),
    objective_tag: str = "prototype_scan",
) -> PrototypeScanOutput:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU attack execution is forbidden")
    if len(source_records) != scan_config.batch_size:
        raise ValueError("Prototype scan requires exactly one configured source batch")
    if not shared_clean_weights or any(weight < 0 for weight in shared_clean_weights):
        raise ValueError("Shared-clean weights must be non-empty and non-negative")
    if clean_separation_weight < 0:
        raise ValueError("Own-clean separation weight must be non-negative")
    if (
        view_consistency_weight < 0
        or not view_scales
        or any(not 0 < scale <= 1 for scale in view_scales)
    ):
        raise ValueError("View configuration is invalid")
    imagenet_root = project_root / "data" / "imagenet_vehicle_official"
    clean = _cuda_images(imagenet_root, source_records, attack_config.canvas_size)
    proxy = load_proxy(
        scan_config.proxy_model,
        project_root / ".hf-cache",
        torch.device("cuda"),
        attack_config,
    )
    if not isinstance(proxy, ContrastiveProxy):
        raise TypeError("Shared-geometry scan currently requires a contrastive proxy")
    reference_embeddings, valid_reference_records = _target_reference_embeddings(
        proxy,
        imagenet_root,
        reference_records,
        attack_config,
        data_config,
    )
    with torch.no_grad():
        clean_output, clean_patch_embeddings = proxy.image_embeddings_with_patches(clean)
        clean_embeddings = clean_output.embeddings.detach().float()
        clean_patch_embeddings = clean_patch_embeddings.detach().float()
    source_prototype = normalized_prototype(clean_embeddings).detach()
    target_prototype = normalized_prototype(reference_embeddings).detach()
    clean_target_similarity = clean_embeddings @ target_prototype
    cka_references = reference_embeddings[: scan_config.batch_size].detach()
    results = []
    embedding_batches = []
    try:
        variants = (
            (lambda_prototype, shared_clean_weight)
            for lambda_prototype in scan_config.lambda_values
            for shared_clean_weight in shared_clean_weights
        )
        for lambda_prototype, shared_clean_weight in variants:
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
                adversarial_output, adversarial_patch_embeddings = (
                    proxy.image_embeddings_with_patches(state.adversarial)
                )
                adversarial_embeddings = adversarial_output.embeddings
                text_output = proxy.target_loss_from_embeddings(
                    adversarial_embeddings,
                    data_config.target_human_label,
                )
                prototype_output = prototype_contrastive_loss(
                    adversarial_embeddings,
                    target_prototype,
                    source_prototype,
                    margin=scan_config.margin,
                    separation_weight=scan_config.separation_weight,
                    clean_embeddings=clean_embeddings,
                    clean_separation_weight=clean_separation_weight,
                )
                view_embeddings = adversarial_embeddings
                view_text_loss = text_output.loss
                view_prototype_loss = prototype_output.total
                if view_consistency_weight > 0:
                    view = random_resized_crop_view(
                        state.adversarial,
                        view_scales[step % len(view_scales)],
                        seed=scan_config.seed + step,
                    )
                    view_embeddings = proxy.image_embeddings(view).embeddings
                    view_text_loss = proxy.target_loss_from_embeddings(
                        view_embeddings, data_config.target_human_label
                    ).loss
                    view_prototype_loss = prototype_contrastive_loss(
                        view_embeddings,
                        target_prototype,
                        source_prototype,
                        margin=scan_config.margin,
                        separation_weight=scan_config.separation_weight,
                        clean_embeddings=clean_embeddings,
                        clean_separation_weight=clean_separation_weight,
                    ).total
                text_loss = 0.5 * (text_output.loss + view_text_loss)
                prototype_loss = 0.5 * (
                    prototype_output.total + view_prototype_loss
                )
                shared_output = shared_geometry_loss(
                    adversarial_embeddings,
                    clean_embeddings,
                    view_embeddings,
                    clean_weight=shared_clean_weight,
                    view_weight=view_consistency_weight,
                    adversarial_patch_embeddings=adversarial_patch_embeddings,
                    clean_patch_embeddings=clean_patch_embeddings,
                )
                total = (
                    text_loss
                    + lambda_prototype * prototype_loss
                    + shared_output.total
                )
                if not torch.isfinite(total):
                    raise RuntimeError(f"Non-finite prototype attack loss at step {step}")
                if step == 0 and lambda_prototype > 0:
                    diagnostics = component_gradient_diagnostics(
                        text_loss,
                        prototype_loss + shared_output.total / lambda_prototype,
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
            artifact_dir = prototype_artifact_dir(
                output_dir,
                objective_tag,
                lambda_prototype,
                shared_clean_weight,
            )
            linf_float, linf_png, adversarial_png = _save_png_batch(
                artifact_dir,
                clean,
                adversarial,
                attack_config.epsilon,
            )
            with torch.no_grad():
                png_output, png_patch_embeddings = proxy.image_embeddings_with_patches(
                    adversarial_png
                )
                png_embeddings = png_output.embeddings.float()
                png_text = proxy.target_loss_from_embeddings(
                    png_embeddings, data_config.target_human_label
                )
                png_prototype = prototype_contrastive_loss(
                    png_embeddings,
                    target_prototype,
                    source_prototype,
                    margin=scan_config.margin,
                    separation_weight=scan_config.separation_weight,
                    clean_embeddings=clean_embeddings,
                    clean_separation_weight=clean_separation_weight,
                )
                png_view_embeddings = proxy.image_embeddings(
                    center_crop_view(adversarial_png, view_scales[0])
                ).embeddings.float()
                png_shared = shared_geometry_loss(
                    png_embeddings,
                    clean_embeddings,
                    png_view_embeddings,
                    clean_weight=shared_clean_weight,
                    view_weight=view_consistency_weight,
                    adversarial_patch_embeddings=png_patch_embeddings,
                    clean_patch_embeddings=clean_patch_embeddings,
                )
            if png_text.class_logits is None:
                raise RuntimeError("Frozen PNG proxy logits are missing")
            target_index = human_label_to_index(data_config.target_human_label)
            target_logits = png_text.class_logits[:, target_index]
            other_mask = torch.arange(10, device="cuda") != target_index
            maximum_other = png_text.class_logits[:, other_mask].max(dim=1).values
            text_margins = target_logits - maximum_other
            text_hits = text_margins > 0
            robust_view_hits = torch.ones_like(text_hits)
            robust_view_denominator = len(source_records)
            if view_consistency_weight > 0:
                robust_view_hits_per_scale = []
                with torch.no_grad():
                    for scale in view_scales:
                        view_embeddings = proxy.image_embeddings(
                            center_crop_view(adversarial_png, scale)
                        ).embeddings
                        view_output = proxy.target_loss_from_embeddings(
                            view_embeddings, data_config.target_human_label
                        )
                        if view_output.class_logits is None:
                            raise RuntimeError("Robust-view proxy logits are missing")
                        robust_view_hits_per_scale.append(
                            view_output.class_logits.argmax(dim=1) == target_index
                        )
                robust_view_hits = torch.stack(robust_view_hits_per_scale).all(dim=0)
            target_gains = png_prototype.target_similarity - clean_target_similarity
            prototype_hits = (
                (target_gains > 0)
                & (png_prototype.target_source_margin > 0)
                & (
                    (png_prototype.target_clean_margin > 0)
                    if clean_separation_weight > 0
                    else torch.ones_like(target_gains, dtype=torch.bool)
                )
            )
            eligible = bool(
                (text_hits & robust_view_hits & prototype_hits).all().item()
            )
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
                    f"robust_views={int(robust_view_hits.sum())}/{len(source_records)}, "
                    f"prototype={int(prototype_hits.sum())}/{len(source_records)}"
                )
            results.append(
                PrototypeAttackResult(
                    lambda_prototype=lambda_prototype,
                    shared_clean_weight=shared_clean_weight,
                    view_consistency_weight=view_consistency_weight,
                    source_image_ids=tuple(record.image_id for record in source_records),
                    target_reference_ids=tuple(
                        record.image_id for record in valid_reference_records
                    ),
                    target_reference_valid_count=len(valid_reference_records),
                    proxy_target_hit_count=int(text_hits.sum().item()),
                    proxy_target_hit_denominator=len(source_records),
                    robust_view_hit_count=int(robust_view_hits.sum().item()),
                    robust_view_hit_denominator=robust_view_denominator,
                    proxy_target_probability=float(png_text.target_probability),
                    proxy_min_text_logit_margin=float(text_margins.min()),
                    mean_target_similarity=float(png_prototype.target_similarity.mean()),
                    minimum_target_similarity_gain=float(target_gains.min()),
                    mean_source_similarity=float(png_prototype.source_similarity.mean()),
                    minimum_target_source_margin=float(png_prototype.target_source_margin.min()),
                    mean_clean_similarity=float(png_prototype.clean_similarity.mean()),
                    minimum_target_clean_margin=float(png_prototype.target_clean_margin.min()),
                    prototype_attraction=float(png_prototype.attraction),
                    prototype_separation=float(png_prototype.separation),
                    prototype_clean_separation=float(png_prototype.clean_separation),
                    loss_text=float(png_text.loss),
                    loss_prototype=float(png_prototype.total),
                    loss_shared_geometry=float(png_shared.total),
                    loss_total=float(
                        png_text.loss
                        + lambda_prototype * png_prototype.total
                        + png_shared.total
                    ),
                    cka_adv_clean_optimized=float(png_shared.clean_alignment),
                    cosine_adv_view=float(png_shared.view_alignment),
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
                    shared_clean_weight=shared_clean_weight,
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


def prototype_artifact_dir(
    output_dir: Path,
    objective_tag: str,
    lambda_prototype: float,
    shared_clean_weight: float,
) -> Path:
    root = output_dir / "attacks" / "DPROTO01" / objective_tag / "batch_00"
    if objective_tag == "prototype_scan" and shared_clean_weight == 0:
        return root / f"lambda_{lambda_prototype:g}"
    return (
        root
        / f"lambda_{lambda_prototype:g}"
        / f"shared_{shared_clean_weight:g}"
    )
