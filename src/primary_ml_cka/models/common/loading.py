from collections.abc import Iterable
from pathlib import Path

import torch
from safetensors import safe_open

from primary_ml_cka.domain.identifiers import MODEL_REVISIONS


def local_snapshot(hf_home: Path, model_id: str, revision: str | None = None) -> Path:
    revision = revision or MODEL_REVISIONS[model_id]
    model_dir = hf_home / "hub" / f"models--{model_id.replace('/', '--')}"
    snapshot = model_dir / "snapshots" / revision
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Missing pinned local snapshot: {snapshot}")
    return snapshot


def freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return module


def safetensor_files(snapshot: Path) -> tuple[Path, ...]:
    files = tuple(sorted(snapshot.glob("*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors weights in {snapshot}")
    return files


def load_prefixed_weights(
    module: torch.nn.Module,
    files: Iterable[Path],
    prefixes: tuple[str, ...],
    *,
    device: torch.device,
    strict: bool = True,
) -> None:
    if device.type != "cuda":
        raise ValueError("Model weights may only be loaded onto CUDA")
    state: dict[str, torch.Tensor] = {}
    for path in files:
        with safe_open(path, framework="pt", device=str(device)) as handle:
            for key in handle.keys():
                prefix = next((item for item in prefixes if key.startswith(item)), None)
                if prefix is not None:
                    state[key.removeprefix(prefix)] = handle.get_tensor(key)
    if not state:
        raise RuntimeError(f"No weights matched prefixes {prefixes}")
    incompatible = module.load_state_dict(state, strict=False, assign=True)
    if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        raise RuntimeError(
            f"Visual weight mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
