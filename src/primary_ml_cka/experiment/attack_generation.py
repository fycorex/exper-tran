import gc
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from primary_ml_cka.artifacts.png import load_png_tensor, save_png_tensor
from primary_ml_cka.artifacts.schemas import ALL_RESULTS_COLUMNS, ResultRow
from primary_ml_cka.artifacts.writers import write_json
from primary_ml_cka.attack.cka.batches import fixed_reference_batch
from primary_ml_cka.attack.cka.linear import clean_anchored_reference_prototype
from primary_ml_cka.attack.likelihood.contrastive_ce import proxy_target_diagnostics
from primary_ml_cka.attack.losses.component_gradients import (
    calibrate_gradient_ratio,
    component_gradient_diagnostics,
)
from primary_ml_cka.attack.losses.primary import primary_loss
from primary_ml_cka.attack.losses.semantic_contrastive import semantic_representation_loss
from primary_ml_cka.attack.optimization.momentum_pgd import MomentumPGDState, descent_step
from primary_ml_cka.attack.optimization.random_start import shared_random_start
from primary_ml_cka.config.schema import AttackConfig, DataConfig
from primary_ml_cka.data.manifests import ImageRecord, read_manifest
from primary_ml_cka.data.preprocessing import ensure_canvas
from primary_ml_cka.domain.identifiers import MODEL_REVISIONS, ModelPair
from primary_ml_cka.domain.labels import human_label_to_index
from primary_ml_cka.evaluation.attack_metrics import AttackRates
from primary_ml_cka.evaluation.representation_metrics import (
    RepresentationMetrics,
    representation_metrics,
)
from primary_ml_cka.infrastructure.memory import peak_memory, reset_peak_memory
from primary_ml_cka.infrastructure.seeds import seed_everything
from primary_ml_cka.infrastructure.timing import Timer
from primary_ml_cka.models.common.gradients import assert_parameter_gradients_none
from primary_ml_cka.models.proxies.registry import load_proxy
from primary_ml_cka.prompts.classification import CLASSIFICATION_PROMPT


@dataclass(frozen=True, slots=True)
class AttackRunResult:
    pair_id: str
    phase: str
    batch_id: str
    lambda_cka: float
    effective_lambda_cka: float
    gradient_ratio: float | None
    cka_source_weight: float
    cka_target_weight: float
    semantic_target_weight: float
    target_cka_mode: str
    target_alignment_temperature: float
    proxy_tap_path: str
    source_human_label: int
    target_human_label: int
    source_image_ids: tuple[str, ...]
    target_reference_ids: tuple[str, ...]
    proxy_target_nll: float
    proxy_target_probability: float
    proxy_target_hit_count: int
    proxy_target_hit_denominator: int
    proxy_target_all_hit: bool
    # None marks legacy logs where only the aggregate proxy gate was recorded.
    proxy_target_hit_mask: tuple[bool, ...] | None
    proxy_min_target_logit_margin: float
    proxy_min_target_probability: float
    proxy_free_target_hit_count: int
    proxy_max_other_probability: float
    proxy_target_probability_margin: float
    proxy_classification_ce: float
    proxy_margin_loss: float
    loss_cka: float
    loss_total: float
    cka_clean_reference: float
    cka_adv_source: float
    cka_adv_reference: float
    reference_cka_gain: float
    source_cka_drop: float
    source_repulsion_achieved: bool | None
    target_attraction_achieved: bool | None
    proxy_representation_shift: float
    grad_ml_l1: float
    grad_cka_weighted_l1: float
    grad_component_cosine: float
    linf_float: float
    linf_png: float
    elapsed_seconds: float
    peak_allocated_vram_gb: float
    peak_reserved_vram_gb: float
    initial_total: float
    final_total: float
    status: str
    cls_loss_mode: str = "ce_margin"
    lambda_cls: float = 1.0
    semantic_mode: str = "target_only"
    semantic_temperature: float = 0.1
    semantic_target_logit_weight: float = 1.0
    semantic_source_logit_weight: float = 1.0
    representation_type: str = "legacy_projected"
    representation_layer: int = -1
    representation_pooling: str = "mean"
    source_reference_ids: tuple[str, ...] = ()
    target_similarity_clean: float = float("nan")
    target_similarity_adversarial: float = float("nan")
    source_similarity_clean: float = float("nan")
    source_similarity_adversarial: float = float("nan")
    semantic_gap_clean: float = float("nan")
    semantic_gap_adversarial: float = float("nan")
    semantic_gap_gain: float = float("nan")
    semantic_negative_kind: str = "source"
    class_reference_ids: tuple[tuple[str, ...], ...] = ()
    gradient_trace: tuple[dict[str, object], ...] = ()
    failure_reason: str = ""


def _cuda_images(root: Path, records: tuple[ImageRecord, ...], canvas_size: int) -> torch.Tensor:
    decoded = []
    for record in records:
        with Image.open(root / record.relative_path) as image:
            tensor = (
                pil_to_tensor(image.convert("RGB"))
                .cuda(non_blocking=False)
                .float()
                .div(255.0)
                .unsqueeze(0)
            )
        decoded.append(ensure_canvas(tensor, canvas_size).squeeze(0))
    canvas = torch.stack(decoded)
    return canvas.mul(255).round().div(255)


def _manifest_name(model_id: str, phase: str) -> str:
    return f"{model_id.replace('/', '__')}__{phase}.jsonl"


def _detached_embedding_bank(
    proxy: object,
    images: torch.Tensor,
    *,
    chunk_size: int = 8,
    representation_type: str = "legacy_projected",
    layer: int = -1,
    pooling: str = "mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    token_chunks = []
    mask_chunks = []
    semantic_chunks = []
    with torch.no_grad():
        for chunk in images.split(chunk_size):
            output = proxy.image_embeddings(
                chunk,
                representation_type=representation_type,
                layer=layer,
                pooling=pooling,
            )
            token_chunks.append(output.tokens.detach().float())
            mask_chunks.append(output.mask.detach())
            semantic_chunks.append(
                (
                    output.semantic_embeddings
                    if output.semantic_embeddings is not None
                    else output.embeddings
                )
                .detach()
                .float()
            )
    return torch.cat(token_chunks), torch.cat(mask_chunks), torch.cat(semantic_chunks)


def _detached_semantic_bank(
    proxy: object,
    images: torch.Tensor,
    *,
    chunk_size: int = 8,
    representation_type: str = "legacy_projected",
    layer: int = -1,
    pooling: str = "mean",
) -> torch.Tensor:
    chunks = []
    with torch.no_grad():
        for chunk in images.split(chunk_size):
            output = proxy.image_embeddings(
                chunk,
                representation_type=representation_type,
                layer=layer,
                pooling=pooling,
            )
            semantic = (
                output.semantic_embeddings
                if output.semantic_embeddings is not None
                else output.embeddings
            )
            chunks.append(semantic.detach().float())
            del output
    return torch.cat(chunks)


def _detached_multiclass_semantic_bank(
    proxy: object,
    canonical_root: Path,
    class_records: tuple[tuple[ImageRecord, ...], ...],
    canvas_size: int,
    *,
    representation_type: str,
    layer: int,
    pooling: str,
) -> torch.Tensor:
    """Extract an equally sized detached semantic bank for every class."""
    banks = []
    expected_count = len(class_records[0])
    if expected_count < 1 or any(len(records) != expected_count for records in class_records):
        raise ValueError("Every multiclass semantic bank must have the same positive size")
    for records in class_records:
        images = _cuda_images(canonical_root, records, canvas_size)
        banks.append(
            _detached_semantic_bank(
                proxy,
                images,
                representation_type=representation_type,
                layer=layer,
                pooling=pooling,
            )
        )
        del images
    return torch.stack(banks)


def _save_png_batch(
    root: Path,
    clean: torch.Tensor,
    adversarial: torch.Tensor,
    epsilon: float,
) -> tuple[float, float, torch.Tensor]:
    linf_float = float((adversarial - clean).abs().max())
    for index in range(clean.shape[0]):
        save_png_tensor(clean[index], root / f"{index:02d}_clean.png")
        save_png_tensor(adversarial[index], root / f"{index:02d}_adv.png")
    clean_reloaded = torch.stack(
        [load_png_tensor(root / f"{index:02d}_clean.png") for index in range(clean.shape[0])]
    ).cuda()
    adversarial_reloaded = torch.stack(
        [load_png_tensor(root / f"{index:02d}_adv.png") for index in range(clean.shape[0])]
    ).cuda()
    linf_png = float((adversarial_reloaded - clean_reloaded).abs().max())
    if linf_png > epsilon + 1e-7:
        raise RuntimeError(f"Serialized perturbation exceeds epsilon: {linf_png}")
    return linf_float, linf_png, adversarial_reloaded


def _pixel_gradient_stats(gradient: torch.Tensor) -> dict[str, object]:
    value = gradient.detach().float()
    return {
        "mean_abs": float(value.abs().mean()),
        "rms": float(value.square().mean().sqrt()),
        "l2_norm": float(value.norm()),
        "max_abs": float(value.abs().max()),
        "zero_fraction": float(value.eq(0).float().mean()),
        "finite": bool(torch.isfinite(value).all()),
    }


def _gradient_trace_row(
    step: int,
    adversarial: torch.Tensor,
    proxy_output: object,
    representation_loss: torch.Tensor,
) -> dict[str, object]:
    components = {
        "target_token": proxy_output.target_nll,
        "closedset_ce": proxy_output.classification_ce,
        "margin": proxy_output.margin_loss,
        "cls_total": proxy_output.loss,
        "representation": representation_loss,
    }
    gradients = {}
    for name, loss in components.items():
        if loss is None or not loss.requires_grad:
            continue
        gradients[name] = torch.autograd.grad(
            loss,
            adversarial,
            retain_graph=True,
            only_inputs=True,
            allow_unused=False,
        )[0]
    row: dict[str, object] = {"step": step}
    for name, gradient in gradients.items():
        row[name] = _pixel_gradient_stats(gradient)
    cls = gradients.get("cls_total")
    representation = gradients.get("representation")
    if cls is not None and representation is not None:
        left = cls.detach().float().flatten(1)
        right = representation.detach().float().flatten(1)
        row["cls_rep_cosine"] = float(
            torch.nn.functional.cosine_similarity(left, right, dim=1).mean()
        )
    else:
        row["cls_rep_cosine"] = None
    return row


def attack_one_batch(
    pair: ModelPair,
    *,
    project_root: Path,
    output_dir: Path,
    phase: str,
    source_records: tuple[ImageRecord, ...],
    reference_records: tuple[ImageRecord, ...],
    source_reference_records: tuple[ImageRecord, ...] | None = None,
    class_reference_records: tuple[tuple[ImageRecord, ...], ...] | None = None,
    source_batch_index: int,
    lambda_cka: float,
    seed: int,
    steps: int,
    attack_config: AttackConfig,
    data_config: DataConfig,
    reference_batch_index: int | None = None,
    reference_bank_size: int | None = None,
    cka_source_weight: float = 1.0,
    cka_target_weight: float = 1.0,
    semantic_target_weight: float = 0.0,
    target_cka_mode: str = "spatial_index_legacy",
    target_alignment_temperature: float = 0.07,
    gradient_ratio: float | None = None,
    objective_tag: str | None = None,
    early_stop_proxy_gate: bool = False,
    progress_interval: int = 0,
    prompt: str = CLASSIFICATION_PROMPT,
    cls_loss_mode: str = "ce_margin",
    lambda_cls: float = 1.0,
    semantic_mode: str = "target_only",
    semantic_temperature: float = 0.1,
    semantic_target_logit_weight: float = 1.0,
    semantic_source_logit_weight: float = 1.0,
    representation_type: str = "legacy_projected",
    representation_layer: int = -1,
    representation_pooling: str = "mean",
    gradient_trace_steps: tuple[int, ...] = (),
    checkpoint_steps: tuple[int, ...] = (),
) -> AttackRunResult:
    # The public ``seed`` controls the complete attack, including any stochastic
    # work performed while materializing a quantized proxy.  Previously only the
    # random-start tensor used this seed, so otherwise identical NF4 reruns could
    # follow different sign-PGD trajectories.
    seed_everything(seed)
    batch_size = len(source_records)
    if batch_size < 2:
        raise ValueError("An attack/CKA batch must contain at least two images")
    if lambda_cka > 0 and not any((cka_source_weight, cka_target_weight, semantic_target_weight)):
        raise ValueError("A positive auxiliary weight requires at least one objective component")
    if gradient_ratio is not None and lambda_cka <= 0:
        raise ValueError("gradient_ratio requires a positive auxiliary loss")
    if gradient_ratio is not None and lambda_cls <= 0:
        raise ValueError("gradient_ratio requires a positive classification weight")
    if target_cka_mode not in {"spatial_index_legacy", "clean_anchor_soft"}:
        raise ValueError(f"Unknown target CKA mode: {target_cka_mode}")
    if target_alignment_temperature <= 0:
        raise ValueError("target_alignment_temperature must be positive")
    if semantic_mode == "multiclass_prototype":
        if class_reference_records is None or len(class_reference_records) != 10:
            raise ValueError("multiclass_prototype requires ten class reference banks")
    elif semantic_mode != "target_only" and not source_reference_records:
        raise ValueError(f"semantic mode {semantic_mode!r} requires source references")
    resolved_checkpoint_steps = tuple(sorted(set(checkpoint_steps)))
    if any(step < 1 or step > steps for step in resolved_checkpoint_steps):
        raise ValueError("checkpoint_steps must be between 1 and the attack step count")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU attack execution is forbidden")
    canonical_root = output_dir / "canonical_images"
    resolved_reference_batch = (
        source_batch_index if reference_batch_index is None else reference_batch_index
    )
    references = fixed_reference_batch(
        reference_records,
        resolved_reference_batch,
        batch_size,
        reference_count=reference_bank_size,
    )
    source_references = (
        fixed_reference_batch(
            source_reference_records,
            0,
            batch_size,
            reference_count=reference_bank_size,
        )
        if source_reference_records
        else ()
    )
    class_references = (
        tuple(
            fixed_reference_batch(
                records,
                0,
                batch_size,
                reference_count=reference_bank_size,
            )
            for records in class_reference_records
        )
        if class_reference_records is not None
        else ()
    )
    clean = _cuda_images(canonical_root, source_records, attack_config.canvas_size)
    needs_token_reference = cka_source_weight > 0 or cka_target_weight > 0
    uses_multiclass_semantic = semantic_mode == "multiclass_prototype"
    reference_images = (
        _cuda_images(canonical_root, references, attack_config.canvas_size)
        if needs_token_reference or not uses_multiclass_semantic
        else None
    )
    source_reference_images = (
        _cuda_images(canonical_root, source_references, attack_config.canvas_size)
        if source_references and not uses_multiclass_semantic
        else None
    )
    timer = Timer()
    reset_peak_memory()
    proxy = load_proxy(pair.proxy_model, project_root / ".hf-cache", device, attack_config)
    with torch.no_grad():
        clean_representation = proxy.image_embeddings(
            clean,
            representation_type=representation_type,
            layer=representation_layer,
            pooling=representation_pooling,
        )
        clean_semantic = (
            clean_representation.semantic_embeddings
            if clean_representation.semantic_embeddings is not None
            else clean_representation.embeddings
        ).detach().float()
        z_clean = (
            clean_representation.tokens.detach().float()
            if needs_token_reference
            else None
        )
        clean_mask = (
            clean_representation.mask.detach() if needs_token_reference else None
        )
    del clean_representation
    semantic_class_reference = None
    if needs_token_reference:
        assert reference_images is not None
        z_reference, reference_mask, semantic_reference_bank = _detached_embedding_bank(
            proxy,
            reference_images,
            representation_type=representation_type,
            layer=representation_layer,
            pooling=representation_pooling,
        )
    else:
        z_reference = None
        reference_mask = None
        semantic_reference_bank = None
    if uses_multiclass_semantic:
        semantic_class_reference = _detached_multiclass_semantic_bank(
            proxy,
            canonical_root,
            class_references,
            attack_config.canvas_size,
            representation_type=representation_type,
            layer=representation_layer,
            pooling=representation_pooling,
        )
        z_reference = None
        reference_mask = None
        semantic_reference_bank = semantic_class_reference[
            human_label_to_index(data_config.target_human_label)
        ]
    elif semantic_reference_bank is None:
        assert reference_images is not None
        semantic_reference_bank = _detached_semantic_bank(
            proxy,
            reference_images,
            representation_type=representation_type,
            layer=representation_layer,
            pooling=representation_pooling,
        )
    semantic_source_reference = None
    if uses_multiclass_semantic:
        assert semantic_class_reference is not None
        semantic_source_reference = semantic_class_reference[
            human_label_to_index(data_config.source_human_label)
        ]
    elif source_reference_images is not None:
        semantic_source_reference = _detached_semantic_bank(
            proxy,
            source_reference_images,
            representation_type=representation_type,
            layer=representation_layer,
            pooling=representation_pooling,
        )
    aligned_target = None
    aligned_target_mask = None
    if lambda_cka > 0 and cka_target_weight > 0 and target_cka_mode == "clean_anchor_soft":
        aligned_target, aligned_target_mask = clean_anchored_reference_prototype(
            z_clean,
            z_reference,
            clean_mask,
            reference_mask,
            temperature=target_alignment_temperature,
        )
    semantic_reference = semantic_reference_bank if semantic_target_weight > 0 else None
    del reference_images, source_reference_images
    initial = (
        shared_random_start(clean, attack_config.epsilon, seed)
        if attack_config.random_start
        else clean.detach().clone().requires_grad_(True)
    )
    state = MomentumPGDState(initial, torch.zeros_like(initial))
    effective_lambda_cka = lambda_cka
    diagnostics = None
    if lambda_cka > 0 and gradient_ratio is not None:
        calibration_proxy = proxy.target_loss(
            state.adversarial,
            data_config.target_human_label,
            prompt,
            cls_loss_mode,
        )
        grad_ml = torch.autograd.grad(
            lambda_cls * calibration_proxy.loss,
            state.adversarial,
            only_inputs=True,
        )[0]
        del calibration_proxy
        calibration_representation = proxy.image_embeddings(
            state.adversarial,
            representation_type=representation_type,
            layer=representation_layer,
            pooling=representation_pooling,
        )
        calibration_losses = primary_loss(
            state.adversarial.new_zeros(()),
            1.0,
            calibration_representation.tokens,
            z_clean,
            z_reference,
            adv_mask=calibration_representation.mask,
            clean_mask=clean_mask,
            reference_mask=reference_mask,
            source_cka_weight=cka_source_weight,
            target_cka_weight=cka_target_weight,
            semantic_target_weight=semantic_target_weight,
            semantic_adv=(
                calibration_representation.semantic_embeddings
                if calibration_representation.semantic_embeddings is not None
                else calibration_representation.embeddings
            ),
            semantic_reference=semantic_reference,
            semantic_source_reference=semantic_source_reference,
            semantic_class_reference=semantic_class_reference,
            semantic_target_class_index=human_label_to_index(data_config.target_human_label),
            semantic_mode=semantic_mode,
            semantic_temperature=semantic_temperature,
            semantic_target_logit_weight=semantic_target_logit_weight,
            semantic_source_logit_weight=semantic_source_logit_weight,
            lambda_cls=lambda_cls,
            aligned_target=aligned_target,
            aligned_target_mask=aligned_target_mask,
        )
        grad_aux = torch.autograd.grad(calibration_losses.cka, state.adversarial, only_inputs=True)[
            0
        ]
        effective_lambda_cka, calibration_diagnostics = calibrate_gradient_ratio(
            grad_ml, grad_aux, gradient_ratio
        )
        calibration_grad_ml_l1 = calibration_diagnostics.grad_ml_l1
        calibration_grad_aux_weighted_l1 = calibration_diagnostics.grad_cka_weighted_l1
        calibration_grad_cosine = calibration_diagnostics.cosine
        print(
            f"{pair.pair_id} calibrated rho={gradient_ratio:g} "
            f"effective_lambda={effective_lambda_cka:.6g} "
            f"grad_cls={calibration_grad_ml_l1:.6g} "
            f"grad_aux_weighted={calibration_grad_aux_weighted_l1:.6g}",
            flush=True,
        )
        del calibration_representation, calibration_losses, grad_ml, grad_aux
    else:
        calibration_grad_ml_l1 = 0.0
        calibration_grad_aux_weighted_l1 = 0.0
        calibration_grad_cosine = 0.0
    initial_total = float("nan")
    last_losses = None
    last_proxy = None
    gradient_trace_rows: list[dict[str, object]] = []
    checkpoint_adversarials: dict[int, torch.Tensor] = {}
    for step in range(steps):
        proxy_output = proxy.target_loss(
            state.adversarial,
            data_config.target_human_label,
            prompt,
            cls_loss_mode,
        )
        if effective_lambda_cka == 0:
            losses = primary_loss(proxy_output.loss, 0, lambda_cls=lambda_cls)
        else:
            adv_representation = proxy.image_embeddings(
                state.adversarial,
                representation_type=representation_type,
                layer=representation_layer,
                pooling=representation_pooling,
            )
            losses = primary_loss(
                proxy_output.loss,
                effective_lambda_cka,
                adv_representation.tokens,
                z_clean,
                z_reference,
                adv_mask=adv_representation.mask,
                clean_mask=clean_mask,
                reference_mask=reference_mask,
                source_cka_weight=cka_source_weight,
                target_cka_weight=cka_target_weight,
                semantic_target_weight=semantic_target_weight,
                semantic_adv=(
                    adv_representation.semantic_embeddings
                    if adv_representation.semantic_embeddings is not None
                    else adv_representation.embeddings
                ),
                semantic_reference=semantic_reference,
                semantic_source_reference=semantic_source_reference,
                semantic_class_reference=semantic_class_reference,
                semantic_target_class_index=human_label_to_index(
                    data_config.target_human_label
                ),
                semantic_mode=semantic_mode,
                semantic_temperature=semantic_temperature,
                semantic_target_logit_weight=semantic_target_logit_weight,
                semantic_source_logit_weight=semantic_source_logit_weight,
                lambda_cls=lambda_cls,
                aligned_target=aligned_target,
                aligned_target_mask=aligned_target_mask,
            )
        if not torch.isfinite(losses.total):
            raise RuntimeError(f"Non-finite total loss at step {step}")
        if step in gradient_trace_steps:
            gradient_trace_rows.append(
                _gradient_trace_row(step, state.adversarial, proxy_output, losses.cka)
            )
        step_diagnostics = proxy_target_diagnostics(
            proxy_output.class_logits,
            target_index=human_label_to_index(data_config.target_human_label),
            required_margin=attack_config.class_margin,
            required_probability=attack_config.proxy_probability_threshold,
        )
        robust_stop_diagnostics = proxy_target_diagnostics(
            proxy_output.class_logits,
            target_index=human_label_to_index(data_config.target_human_label),
            required_margin=attack_config.class_margin + 0.5,
            required_probability=max(attack_config.proxy_probability_threshold, 0.95),
        )
        if progress_interval and (step == 0 or (step + 1) % progress_interval == 0):
            print(
                f"{pair.pair_id} lambda={lambda_cka:g} step={step + 1}/{steps} "
                f"closed_set={step_diagnostics.hit_count}/{step_diagnostics.denominator} "
                f"min_margin={step_diagnostics.minimum_logit_margin:.4f} "
                f"min_probability={step_diagnostics.minimum_target_probability:.4f}",
                flush=True,
            )
        if early_stop_proxy_gate and step > 0 and robust_stop_diagnostics.all_hit:
            last_losses = losses
            last_proxy = proxy_output
            print(
                f"{pair.pair_id} lambda={lambda_cka:g} early-stop at step {step + 1}: "
                "robust closed-set proxy gate reached",
                flush=True,
            )
            break
        if step == 0:
            initial_total = float(losses.total.detach())
            memory_heavy_diagnostic_proxies = {
                "google/gemma-4-E4B-it",
                "google/siglip2-so400m-patch14-384",
            }
            if (
                lambda_cka > 0
                and gradient_ratio is None
                and pair.proxy_model not in memory_heavy_diagnostic_proxies
            ):
                diagnostics = component_gradient_diagnostics(
                    lambda_cls * losses.ml,
                    losses.cka,
                    state.adversarial,
                    effective_lambda_cka,
                )
        state = descent_step(
            losses.total,
            state,
            clean,
            epsilon=attack_config.epsilon,
            step_size=attack_config.step_size,
            momentum=attack_config.momentum,
        )
        completed_step = step + 1
        if completed_step in resolved_checkpoint_steps:
            # Keep checkpoints off GPU: Gemma runs leave too little VRAM for
            # several full-resolution adversarial batches to remain resident.
            checkpoint_adversarials[completed_step] = state.adversarial.detach().cpu()
        last_losses = losses
        last_proxy = proxy_output
    assert last_losses is not None and last_proxy is not None
    adversarial = state.adversarial.detach()
    with torch.no_grad():
        final_proxy = proxy.target_loss(
            adversarial,
            data_config.target_human_label,
            prompt,
            cls_loss_mode,
        )
        final_representation = proxy.image_embeddings(
            adversarial,
            representation_type=representation_type,
            layer=representation_layer,
            pooling=representation_pooling,
        )
        z_adv_final = final_representation.tokens.float()
        final_losses = primary_loss(
            final_proxy.loss,
            effective_lambda_cka,
            z_adv_final if effective_lambda_cka > 0 else None,
            z_clean if effective_lambda_cka > 0 else None,
            z_reference if effective_lambda_cka > 0 else None,
            adv_mask=final_representation.mask if effective_lambda_cka > 0 else None,
            clean_mask=clean_mask if effective_lambda_cka > 0 else None,
            reference_mask=reference_mask if effective_lambda_cka > 0 else None,
            source_cka_weight=cka_source_weight,
            target_cka_weight=cka_target_weight,
            semantic_target_weight=semantic_target_weight,
            semantic_adv=(
                final_representation.semantic_embeddings
                if final_representation.semantic_embeddings is not None
                else final_representation.embeddings
            ),
            semantic_reference=semantic_reference,
            semantic_source_reference=semantic_source_reference,
            semantic_class_reference=semantic_class_reference,
            semantic_target_class_index=human_label_to_index(data_config.target_human_label),
            semantic_mode=semantic_mode,
            semantic_temperature=semantic_temperature,
            semantic_target_logit_weight=semantic_target_logit_weight,
            semantic_source_logit_weight=semantic_source_logit_weight,
            lambda_cls=lambda_cls,
            aligned_target=aligned_target,
            aligned_target_mask=aligned_target_mask,
        )
    assert_parameter_gradients_none(proxy.model)
    representation = (
        representation_metrics(
            z_clean,
            final_representation.tokens.float(),
            z_reference,
            clean_mask,
            final_representation.mask,
            reference_mask,
            aligned_target,
            aligned_target_mask,
        )
        if z_reference is not None
        else RepresentationMetrics(
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
        )
    )
    adversarial_semantic = (
        final_representation.semantic_embeddings
        if final_representation.semantic_embeddings is not None
        else final_representation.embeddings
    )
    clean_semantic_metrics = semantic_representation_loss(
        clean_semantic,
        semantic_reference_bank,
        semantic_source_reference,
        mode=semantic_mode,
        tau=semantic_temperature,
        target_logit_weight=semantic_target_logit_weight,
        source_logit_weight=semantic_source_logit_weight,
        class_references=semantic_class_reference,
        target_class_index=human_label_to_index(data_config.target_human_label),
    )
    adversarial_semantic_metrics = semantic_representation_loss(
        adversarial_semantic,
        semantic_reference_bank,
        semantic_source_reference,
        mode=semantic_mode,
        tau=semantic_temperature,
        target_logit_weight=semantic_target_logit_weight,
        source_logit_weight=semantic_source_logit_weight,
        class_references=semantic_class_reference,
        target_class_index=human_label_to_index(data_config.target_human_label),
    )
    artifact_dir = output_dir / "attacks" / pair.pair_id / phase / f"batch_{source_batch_index:02d}"
    if objective_tag is not None:
        artifact_dir /= objective_tag
    artifact_dir /= f"lambda_{lambda_cka:g}"
    linf_float, linf_png, adversarial_png = _save_png_batch(
        artifact_dir,
        clean,
        adversarial,
        attack_config.epsilon,
    )
    checkpoint_rows: list[dict[str, object]] = []
    for checkpoint_step, checkpoint_cpu in checkpoint_adversarials.items():
        checkpoint_dir = artifact_dir / f"checkpoint_step_{checkpoint_step:03d}"
        checkpoint_gpu = checkpoint_cpu.to(device)
        checkpoint_linf_float, checkpoint_linf_png, checkpoint_png = _save_png_batch(
            checkpoint_dir,
            clean,
            checkpoint_gpu,
            attack_config.epsilon,
        )
        with torch.no_grad():
            checkpoint_proxy = proxy.target_loss(
                checkpoint_png,
                data_config.target_human_label,
                prompt,
                cls_loss_mode,
            )
        checkpoint_diagnostics = proxy_target_diagnostics(
            checkpoint_proxy.class_logits,
            target_index=human_label_to_index(data_config.target_human_label),
            required_margin=attack_config.class_margin,
            required_probability=attack_config.proxy_probability_threshold,
        )
        checkpoint_free_labels = (
            proxy.free_generate_labels(checkpoint_png, prompt)
            if attack_config.require_proxy_free_generation and checkpoint_diagnostics.all_hit
            else None
        )
        checkpoint_rows.append(
            {
                "step": checkpoint_step,
                "artifact_dir": str(checkpoint_dir),
                "proxy_hit_count": checkpoint_diagnostics.hit_count,
                "proxy_hit_mask": list(checkpoint_diagnostics.hit_mask),
                "proxy_all_hit": checkpoint_diagnostics.all_hit,
                "proxy_min_margin": checkpoint_diagnostics.minimum_logit_margin,
                "proxy_min_probability": checkpoint_diagnostics.minimum_target_probability,
                "proxy_free_hit_count": (
                    sum(
                        label == data_config.target_human_label
                        for label in checkpoint_free_labels
                    )
                    if checkpoint_free_labels is not None
                    else None
                ),
                "linf_float": checkpoint_linf_float,
                "linf_png": checkpoint_linf_png,
            }
        )
        del checkpoint_cpu, checkpoint_gpu, checkpoint_png, checkpoint_proxy
    if checkpoint_rows:
        write_json(artifact_dir / "checkpoint_summary.json", checkpoint_rows)
    with torch.no_grad():
        png_proxy = proxy.target_loss(
            adversarial_png,
            data_config.target_human_label,
            prompt,
            cls_loss_mode,
        )
    if png_proxy.class_logits is None or not torch.isfinite(png_proxy.class_logits).all():
        raise RuntimeError("Frozen PNG proxy class logits are missing or non-finite")
    free_labels = (
        proxy.free_generate_labels(adversarial_png, prompt)
        if attack_config.require_proxy_free_generation
        else None
    )
    proxy_diagnostics = proxy_target_diagnostics(
        png_proxy.class_logits,
        target_index=human_label_to_index(data_config.target_human_label),
        required_margin=attack_config.class_margin,
        required_probability=attack_config.proxy_probability_threshold,
        free_generated_labels=free_labels,
    )
    memory = peak_memory()
    result = AttackRunResult(
        pair_id=pair.pair_id,
        phase=phase,
        batch_id=f"{source_batch_index:02d}",
        lambda_cka=lambda_cka,
        effective_lambda_cka=effective_lambda_cka,
        gradient_ratio=gradient_ratio,
        cka_source_weight=cka_source_weight,
        cka_target_weight=cka_target_weight,
        semantic_target_weight=semantic_target_weight,
        target_cka_mode=target_cka_mode,
        target_alignment_temperature=target_alignment_temperature,
        proxy_tap_path=final_representation.tap.module_path,
        source_human_label=data_config.source_human_label,
        target_human_label=data_config.target_human_label,
        source_image_ids=tuple(record.image_id for record in source_records),
        target_reference_ids=tuple(record.image_id for record in references),
        proxy_target_nll=float(png_proxy.target_nll),
        proxy_target_probability=float(png_proxy.target_probability),
        proxy_target_hit_count=proxy_diagnostics.hit_count,
        proxy_target_hit_denominator=proxy_diagnostics.denominator,
        proxy_target_all_hit=proxy_diagnostics.all_hit,
        proxy_target_hit_mask=proxy_diagnostics.hit_mask,
        proxy_min_target_logit_margin=proxy_diagnostics.minimum_logit_margin,
        proxy_min_target_probability=proxy_diagnostics.minimum_target_probability,
        proxy_free_target_hit_count=(
            sum(label == data_config.target_human_label for label in free_labels)
            if free_labels is not None
            else batch_size
        ),
        proxy_max_other_probability=float(png_proxy.max_other_probability),
        proxy_target_probability_margin=float(
            png_proxy.target_probability - png_proxy.max_other_probability
        ),
        proxy_classification_ce=float(png_proxy.classification_ce),
        proxy_margin_loss=float(png_proxy.margin_loss),
        loss_cka=float(final_losses.cka),
        loss_total=float(final_losses.total),
        cka_clean_reference=representation.cka_clean_reference,
        cka_adv_source=representation.cka_adv_source,
        cka_adv_reference=representation.cka_adv_reference,
        reference_cka_gain=representation.reference_cka_gain,
        source_cka_drop=representation.source_cka_drop,
        source_repulsion_achieved=(
            representation.source_cka_drop > 0 if cka_source_weight > 0 else None
        ),
        target_attraction_achieved=(
            representation.reference_cka_gain > 0 if cka_target_weight > 0 else None
        ),
        proxy_representation_shift=representation.proxy_representation_shift,
        grad_ml_l1=(diagnostics.grad_ml_l1 if diagnostics else calibration_grad_ml_l1),
        grad_cka_weighted_l1=(
            diagnostics.grad_cka_weighted_l1 if diagnostics else calibration_grad_aux_weighted_l1
        ),
        grad_component_cosine=(diagnostics.cosine if diagnostics else calibration_grad_cosine),
        linf_float=linf_float,
        linf_png=linf_png,
        elapsed_seconds=timer.elapsed(),
        peak_allocated_vram_gb=memory.allocated_gb,
        peak_reserved_vram_gb=memory.reserved_gb,
        initial_total=initial_total,
        final_total=float(final_losses.total),
        status="ok" if proxy_diagnostics.all_hit else "proxy_target_not_reached",
        cls_loss_mode=cls_loss_mode,
        lambda_cls=lambda_cls,
        semantic_mode=semantic_mode,
        semantic_temperature=semantic_temperature,
        semantic_target_logit_weight=semantic_target_logit_weight,
        semantic_source_logit_weight=semantic_source_logit_weight,
        representation_type=representation_type,
        representation_layer=representation_layer,
        representation_pooling=representation_pooling,
        source_reference_ids=tuple(record.image_id for record in source_references),
        target_similarity_clean=float(clean_semantic_metrics.target_similarity),
        target_similarity_adversarial=float(adversarial_semantic_metrics.target_similarity),
        source_similarity_clean=float(clean_semantic_metrics.source_similarity),
        source_similarity_adversarial=float(adversarial_semantic_metrics.source_similarity),
        semantic_gap_clean=float(clean_semantic_metrics.semantic_gap),
        semantic_gap_adversarial=float(adversarial_semantic_metrics.semantic_gap),
        semantic_gap_gain=float(
            adversarial_semantic_metrics.semantic_gap - clean_semantic_metrics.semantic_gap
        ),
        semantic_negative_kind=(
            "strongest_non_target"
            if semantic_mode == "multiclass_prototype"
            else "source"
        ),
        class_reference_ids=tuple(
            tuple(record.image_id for record in records) for records in class_references
        ),
        gradient_trace=tuple(gradient_trace_rows),
        failure_reason=(
            ""
            if proxy_diagnostics.all_hit
            else "Frozen PNG proxy target criterion reached for "
            f"{proxy_diagnostics.hit_count}/{proxy_diagnostics.denominator} images"
        ),
    )
    log_name = f"{result.batch_id}_{lambda_cka:g}"
    if objective_tag is not None:
        log_name += f"_{objective_tag}"
    log_path = output_dir / "logs" / pair.pair_id / phase / f"{log_name}.json"
    write_json(log_path, result)
    del proxy
    gc.collect()
    torch.cuda.empty_cache()
    return result


def result_row(
    pair: ModelPair,
    result: AttackRunResult,
    seed: int,
    steps: int,
    rates: AttackRates | None = None,
) -> ResultRow:
    values = {
        "pair_id": pair.pair_id,
        "exp_type": pair.exp_type.value,
        "proxy_model": pair.proxy_model,
        "target_model": pair.target_model,
        "source_human_label": result.source_human_label,
        "target_human_label": result.target_human_label,
        "proxy_revision": MODEL_REVISIONS[pair.proxy_model],
        "target_revision": MODEL_REVISIONS[pair.target_model],
        "proxy_tap_status": "validated",
        "proxy_tap_path": result.proxy_tap_path,
        "phase": result.phase,
        "batch_id": result.batch_id,
        "source_image_ids": "|".join(result.source_image_ids),
        "target_reference_ids": "|".join(result.target_reference_ids),
        "lambda": result.lambda_cka,
        "effective_lambda": result.effective_lambda_cka,
        "gradient_ratio": "" if result.gradient_ratio is None else result.gradient_ratio,
        "cka_source_weight": result.cka_source_weight,
        "cka_target_weight": result.cka_target_weight,
        "semantic_target_weight": result.semantic_target_weight,
        "target_cka_mode": result.target_cka_mode,
        "target_alignment_temperature": result.target_alignment_temperature,
        "seed": seed,
        "steps": steps,
        "clean_valid_count": (
            rates.clean_valid_count if rates is not None else len(result.source_image_ids)
        ),
        "targeted_hit_count": rates.targeted_hit_count if rates is not None else "",
        "tasr_percent": rates.tasr_percent if rates is not None else "",
        "untargeted_hit_count": rates.untargeted_hit_count if rates is not None else "",
        "asr_percent": rates.asr_percent if rates is not None else "",
        "proxy_target_nll": result.proxy_target_nll,
        "proxy_target_probability": result.proxy_target_probability,
        "proxy_target_hit_count": result.proxy_target_hit_count,
        "proxy_target_hit_denominator": result.proxy_target_hit_denominator,
        "proxy_target_all_hit": result.proxy_target_all_hit,
        "proxy_target_hit_mask": (
            ""
            if result.proxy_target_hit_mask is None
            else "|".join("1" if value else "0" for value in result.proxy_target_hit_mask)
        ),
        "proxy_min_target_logit_margin": result.proxy_min_target_logit_margin,
        "proxy_min_target_probability": result.proxy_min_target_probability,
        "proxy_free_target_hit_count": result.proxy_free_target_hit_count,
        "proxy_max_other_probability": result.proxy_max_other_probability,
        "proxy_target_probability_margin": result.proxy_target_probability_margin,
        "proxy_classification_ce": result.proxy_classification_ce,
        "proxy_margin_loss": result.proxy_margin_loss,
        "loss_cka": result.loss_cka,
        "loss_total": result.loss_total,
        "cka_clean_reference": result.cka_clean_reference,
        "cka_adv_source": result.cka_adv_source,
        "cka_adv_reference": result.cka_adv_reference,
        "reference_cka_gain": result.reference_cka_gain,
        "source_cka_drop": result.source_cka_drop,
        "source_repulsion_achieved": result.source_repulsion_achieved,
        "target_attraction_achieved": result.target_attraction_achieved,
        "proxy_representation_shift": result.proxy_representation_shift,
        "grad_ml_l1": result.grad_ml_l1,
        "grad_cka_weighted_l1": result.grad_cka_weighted_l1,
        "grad_component_cosine": result.grad_component_cosine,
        "linf_float": result.linf_float,
        "linf_png": result.linf_png,
        "elapsed_seconds": result.elapsed_seconds,
        "peak_allocated_vram_gb": result.peak_allocated_vram_gb,
        "peak_reserved_vram_gb": result.peak_reserved_vram_gb,
        "status": result.status,
        "failure_reason": result.failure_reason,
    }
    return ResultRow(tuple(values[column] for column in ALL_RESULTS_COLUMNS))


def blocked_result_row(
    pair: ModelPair,
    *,
    phase: str,
    seed: int,
    steps: int,
    error: Exception,
    lambda_cka: float | None = None,
) -> ResultRow:
    values: dict[str, object] = {column: "" for column in ALL_RESULTS_COLUMNS}
    values.update(
        {
            "pair_id": pair.pair_id,
            "exp_type": pair.exp_type.value,
            "proxy_model": pair.proxy_model,
            "target_model": pair.target_model,
            "proxy_revision": MODEL_REVISIONS[pair.proxy_model],
            "target_revision": MODEL_REVISIONS[pair.target_model],
            "phase": phase,
            "lambda": "" if lambda_cka is None else lambda_cka,
            "seed": seed,
            "steps": steps,
            "status": "blocked",
            "failure_reason": repr(error),
        }
    )
    return ResultRow(tuple(values[column] for column in ALL_RESULTS_COLUMNS))


def load_phase_records(output_dir: Path, target_model: str, phase: str) -> tuple[ImageRecord, ...]:
    path = output_dir / "evaluation" / "manifests" / _manifest_name(target_model, phase)
    if not path.is_file():
        raise RuntimeError(f"Screened {phase} manifest missing: {path}")
    return read_manifest(path)
