import os
from pathlib import Path
from typing import TypeVar

import yaml

T = TypeVar("T")


def _expand(value: object) -> object:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        name = value[2:-1]
        if name not in os.environ:
            raise ValueError(f"Required environment variable is unset: {name}")
        return os.environ[name]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def load_config(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Top-level YAML value must be a mapping: {path}")
    return _expand(loaded)
