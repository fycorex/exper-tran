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
from primary_ml_cka.attack.likelihood.contrastive_ce import proxy_target_diagnostics
from primary_ml_cka.attack.losses.component_gradients import component_gradient_diagnostics
from primary_ml_cka.attack.losses.primary import primary_loss
from primary_ml_cka.attack.optimization.momentum_pgd import MomentumPGDState, descent_step
from primary_ml_cka.attack.optimization.random_start import shared_random_start
from primary_ml_cka.config.schema import AttackConfig, DataConfig
from primary_ml_cka.data.manifests import ImageRecord, read_manifest
from primary_ml_cka.data.preprocessing import ensure_canvas
from primary_ml_cka.domain.identifiers import MODEL_REVISIONS, ModelPair
from primary_ml_cka.domain.labels import human_label_to_index
from primary_ml_cka.evaluation.attack_metrics import AttackRates
from primary_ml_cka.evaluation.representation_metrics import representation_metrics
from primary_ml_cka.infrastructure.memory import peak_memory, reset_peak_memory
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
    source_human_label: int
    target_human_label: int
    source_image_ids: tuple[str, ...]
    target_reference_ids: tuple[str, ...]
    proxy_target_nll: float
    proxy_target_probability: float
    proxy_target_hit_count: int
    proxy_target_hit_denominator: int
    proxy_target_all_hit: bool
    proxy_min_target_logit_margin: float
    proxy_max_other_probability: float
    proxy_target_probability_margin: float
    proxy_classification_ce: float
    proxy_rank_loss: float
    proxy_other_suppression_loss: float
    loss_cka: float
    loss_total: float
    cka_clean_reference: float
    cka_adv_source: float
    cka_adv_reference: float
    reference_cka_gain: float
    source_cka_drop: float
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


def attack_one_batch(
    pair: ModelPair,
    *,
    project_root: Path,
    output_dir: Path,
    phase: str,
    source_records: tuple[ImageRecord, ...],
    reference_records: tuple[ImageRecord, ...],
    source_batch_index: int,
    lambda_cka: float,
    seed: int,
    steps: int,
    attack_config: AttackConfig,
    data_config: DataConfig,
    reference_batch_index: int | None = None,
) -> AttackRunResult:
    batch_size = len(source_records)
    if batch_size < 2:
        raise ValueError("An attack/CKA batch must contain at least two images")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU attack execution is forbidden")
    imagenet_root = project_root / "data" / "imagenet_vehicle_official"
    resolved_reference_batch = (
        source_batch_index if reference_batch_index is None else reference_batch_index
    )
    references = fixed_reference_batch(
        reference_records,
        resolved_reference_batch,
        batch_size,
    )
    clean = _cuda_images(imagenet_root, source_records, attack_config.canvas_size)
    reference_images = _cuda_images(imagenet_root, references, attack_config.canvas_size)
    timer = Timer()
    reset_peak_memory()
    proxy = load_proxy(pair.proxy_model, project_root / ".hf-cache", device, attack_config)
    with torch.no_grad():
        z_clean = proxy.image_embeddings(clean).embeddings.detach().float()
        z_reference = proxy.image_embeddings(reference_images).embeddings.detach().float()
    initial = shared_random_start(clean, attack_config.epsilon, seed)
    state = MomentumPGDState(initial, torch.zeros_like(initial))
    initial_total = float("nan")
    diagnostics = None
    last_losses = None
    last_proxy = None
    for step in range(steps):
        proxy_output = proxy.target_loss(
            state.adversarial,
            data_config.target_human_label,
            CLASSIFICATION_PROMPT,
        )
        if lambda_cka == 0:
            losses = primary_loss(proxy_output.loss, 0)
        else:
            z_adv = proxy.image_embeddings(state.adversarial).embeddings
            losses = primary_loss(proxy_output.loss, lambda_cka, z_adv, z_clean, z_reference)
        if not torch.isfinite(losses.total):
            raise RuntimeError(f"Non-finite total loss at step {step}")
        if step == 0:
            initial_total = float(losses.total.detach())
            if lambda_cka > 0:
                diagnostics = component_gradient_diagnostics(
                    losses.ml, losses.cka, state.adversarial, lambda_cka
                )
        state = descent_step(
            losses.total,
            state,
            clean,
            epsilon=attack_config.epsilon,
            step_size=attack_config.step_size,
            momentum=attack_config.momentum,
        )
        last_losses = losses
        last_proxy = proxy_output
    assert last_losses is not None and last_proxy is not None
    adversarial = state.adversarial.detach()
    with torch.no_grad():
        final_proxy = proxy.target_loss(
            adversarial,
            data_config.target_human_label,
            CLASSIFICATION_PROMPT,
        )
        z_adv_final = proxy.image_embeddings(adversarial).embeddings.float()
        final_losses = primary_loss(
            final_proxy.loss,
            lambda_cka,
            z_adv_final if lambda_cka > 0 else None,
            z_clean if lambda_cka > 0 else None,
            z_reference if lambda_cka > 0 else None,
        )
    assert_parameter_gradients_none(proxy.model)
    representation = representation_metrics(z_clean, z_adv_final, z_reference)
    artifact_dir = (
        output_dir
        / "attacks"
        / pair.pair_id
        / phase
        / f"batch_{source_batch_index:02d}"
        / f"lambda_{lambda_cka:g}"
    )
    linf_float, linf_png, adversarial_png = _save_png_batch(
        artifact_dir,
        clean,
        adversarial,
        attack_config.epsilon,
    )
    with torch.no_grad():
        png_proxy = proxy.target_loss(
            adversarial_png,
            data_config.target_human_label,
            CLASSIFICATION_PROMPT,
        )
    if png_proxy.class_logits is None or not torch.isfinite(png_proxy.class_logits).all():
        raise RuntimeError("Frozen PNG proxy class logits are missing or non-finite")
    proxy_diagnostics = proxy_target_diagnostics(
        png_proxy.class_logits,
        target_index=human_label_to_index(data_config.target_human_label),
    )
    memory = peak_memory()
    zero = 0.0
    result = AttackRunResult(
        pair_id=pair.pair_id,
        phase=phase,
        batch_id=f"{source_batch_index:02d}",
        lambda_cka=lambda_cka,
        source_human_label=data_config.source_human_label,
        target_human_label=data_config.target_human_label,
        source_image_ids=tuple(record.image_id for record in source_records),
        target_reference_ids=tuple(record.image_id for record in references),
        proxy_target_nll=float(png_proxy.target_nll),
        proxy_target_probability=float(png_proxy.target_probability),
        proxy_target_hit_count=proxy_diagnostics.hit_count,
        proxy_target_hit_denominator=proxy_diagnostics.denominator,
        proxy_target_all_hit=proxy_diagnostics.all_hit,
        proxy_min_target_logit_margin=proxy_diagnostics.minimum_logit_margin,
        proxy_max_other_probability=float(png_proxy.max_other_probability),
        proxy_target_probability_margin=float(
            png_proxy.target_probability - png_proxy.max_other_probability
        ),
        proxy_classification_ce=float(png_proxy.classification_ce),
        proxy_rank_loss=float(png_proxy.rank_loss),
        proxy_other_suppression_loss=float(png_proxy.other_suppression_loss),
        loss_cka=float(final_losses.cka),
        loss_total=float(final_losses.total),
        cka_clean_reference=representation.cka_clean_reference,
        cka_adv_source=representation.cka_adv_source,
        cka_adv_reference=representation.cka_adv_reference,
        reference_cka_gain=representation.reference_cka_gain,
        source_cka_drop=representation.source_cka_drop,
        proxy_representation_shift=representation.proxy_representation_shift,
        grad_ml_l1=diagnostics.grad_ml_l1 if diagnostics else zero,
        grad_cka_weighted_l1=(diagnostics.grad_cka_weighted_l1 if diagnostics else zero),
        grad_component_cosine=diagnostics.cosine if diagnostics else zero,
        linf_float=linf_float,
        linf_png=linf_png,
        elapsed_seconds=timer.elapsed(),
        peak_allocated_vram_gb=memory.allocated_gb,
        peak_reserved_vram_gb=memory.reserved_gb,
        initial_total=initial_total,
        final_total=float(final_losses.total),
        status="ok" if proxy_diagnostics.all_hit else "proxy_target_not_reached",
        failure_reason=(
            ""
            if proxy_diagnostics.all_hit
            else "Frozen PNG proxy target criterion reached for "
            f"{proxy_diagnostics.hit_count}/{proxy_diagnostics.denominator} images"
        ),
    )
    log_path = output_dir / "logs" / pair.pair_id / phase / f"{result.batch_id}_{lambda_cka:g}.json"
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
    if pair.proxy_model.startswith("Qwen/"):
        proxy_tap_path = "model.visual.merger.norm"
    elif pair.proxy_model.startswith("OpenGVLab/InternVL"):
        proxy_tap_path = "model.vision_tower.layernorm"
    else:
        proxy_tap_path = "get_image_features"
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
        "proxy_tap_path": proxy_tap_path,
        "phase": result.phase,
        "batch_id": result.batch_id,
        "source_image_ids": "|".join(result.source_image_ids),
        "target_reference_ids": "|".join(result.target_reference_ids),
        "lambda": result.lambda_cka,
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
        "proxy_min_target_logit_margin": result.proxy_min_target_logit_margin,
        "proxy_max_other_probability": result.proxy_max_other_probability,
        "proxy_target_probability_margin": result.proxy_target_probability_margin,
        "proxy_classification_ce": result.proxy_classification_ce,
        "proxy_rank_loss": result.proxy_rank_loss,
        "proxy_other_suppression_loss": result.proxy_other_suppression_loss,
        "loss_cka": result.loss_cka,
        "loss_total": result.loss_total,
        "cka_clean_reference": result.cka_clean_reference,
        "cka_adv_source": result.cka_adv_source,
        "cka_adv_reference": result.cka_adv_reference,
        "reference_cka_gain": result.reference_cka_gain,
        "source_cka_drop": result.source_cka_drop,
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
