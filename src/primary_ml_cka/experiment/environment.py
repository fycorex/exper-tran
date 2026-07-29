import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from primary_ml_cka.domain.identifiers import MODEL_REVISIONS
from primary_ml_cka.infrastructure.atomic_io import atomic_text_write


def _version(module_name: str) -> str:
    try:
        module = importlib.import_module(module_name)
        return str(getattr(module, "__version__", "installed"))
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}: {exc}"


def _git_commit(project_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    return (
        result.stdout.strip() if result.returncode == 0 else f"unavailable: {result.stderr.strip()}"
    )


def write_environment(project_root: Path, output_dir: Path) -> None:
    gpu = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            gpu.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
            )
    payload = {
        "git_commit": _git_commit(project_root),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": _version("torch"),
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "transformers": _version("transformers"),
        "bitsandbytes": _version("bitsandbytes"),
        "vllm": _version("vllm"),
        "model_revisions": MODEL_REVISIONS,
        "processor_revisions": MODEL_REVISIONS,
        "dtypes": {
            "generative_proxy_compute": "bfloat16",
            "proxy_visual": "bfloat16",
            "cka": "float32",
        },
        "gpu": gpu,
    }
    atomic_text_write(output_dir / "environment.json", json.dumps(payload, indent=2) + "\n")
