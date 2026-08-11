import json
import os
from dataclasses import dataclass
from pathlib import Path

from primary_ml_cka.config.loader import load_config
from primary_ml_cka.config.schema import (
    AlphaScanConfig,
    AttackConfig,
    DataConfig,
    SmokeConfig,
)
from primary_ml_cka.config.validation import (
    validate_alpha_scan_config,
    validate_attack_config,
    validate_data_config,
    validate_smoke_config,
)


@dataclass(frozen=True, slots=True)
class CommandContext:
    project_root: Path
    output_dir: Path
    pair_id: str | None
    resume: bool
    dry_run: bool
    seed: int | None
    config_path: Path | None
    image_count: int | None


def resolve_attack_config(
    context: CommandContext, *, require_canonical_lambda_grid: bool = True
) -> AttackConfig:
    path = context.config_path or (
        context.project_root / "configs" / "attacks" / "primary_ml_cka.yaml"
    )
    raw = load_config(path)
    values = {key: raw[key] for key in AttackConfig.__dataclass_fields__ if key in raw}
    if "lambdas" in values:
        values["lambdas"] = tuple(float(value) for value in values["lambdas"])
    config = AttackConfig(**values)
    validate_attack_config(
        config, require_canonical_lambda_grid=require_canonical_lambda_grid
    )
    return config


def resolve_data_config(context: CommandContext) -> DataConfig:
    configured_path = os.environ.get("PRIMARY_ML_CKA_DATA_CONFIG")
    path = (
        Path(configured_path)
        if configured_path
        else context.project_root / "configs" / "data" / "imagenet_vehicle10.yaml"
    )
    raw = load_config(path)
    values = {key: raw[key] for key in DataConfig.__dataclass_fields__ if key in raw}
    config = DataConfig(**values)
    validate_data_config(config)
    return config


def resolve_smoke_config(
    context: CommandContext,
    attack_config: AttackConfig,
) -> SmokeConfig:
    path = context.project_root / "configs" / "runs" / "smoke.yaml"
    raw = load_config(path)
    values = {key: raw[key] for key in SmokeConfig.__dataclass_fields__ if key in raw}
    if "lambdas" in values:
        values["lambdas"] = tuple(float(value) for value in values["lambdas"])
    config = SmokeConfig(**values)
    validate_smoke_config(config, attack_config)
    return config


def resolve_alpha_scan_config(
    context: CommandContext,
    attack_config: AttackConfig,
) -> AlphaScanConfig:
    path = context.project_root / "configs" / "runs" / "clip_intra_alpha_scan.yaml"
    raw = load_config(path)
    values = {key: raw[key] for key in AlphaScanConfig.__dataclass_fields__ if key in raw}
    if "alphas" in values:
        values["alphas"] = tuple(float(value) for value in values["alphas"])
    config = AlphaScanConfig(**values)
    validate_alpha_scan_config(config, attack_config)
    return config


def require_real_run_ready(context: CommandContext, *, require_taps: bool = True) -> None:
    if context.dry_run:
        return
    test_report = context.output_dir / "summaries" / "test_report.txt"
    if not test_report.is_file() or "passed" not in test_report.read_text(encoding="utf-8"):
        raise RuntimeError("Correctness tests have not passed; run `tests run` first")
    if require_taps:
        taps_dir = context.output_dir / "taps"
        if not taps_dir.is_dir() or not any(taps_dir.glob("*.json")):
            raise RuntimeError(
                "No validated proxy-tap records exist; run `models inspect-taps` first"
            )


def require_proxy_tap(context: CommandContext, proxy_model: str) -> None:
    path = context.output_dir / "taps" / f"{proxy_model.replace('/', '__')}.json"
    if not path.is_file():
        raise RuntimeError(f"Proxy tap record missing for {proxy_model}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") in {None, "blocked", "requires_validation"}:
        raise RuntimeError(
            f"Proxy tap is not validated for {proxy_model}: {payload.get('error', payload)}"
        )
    expected_path = (
        "model.get_image_features.pooler_output"
        if proxy_model.startswith(("Qwen/", "OpenGVLab/InternVL"))
        else "model.embed_vision"
        if proxy_model.startswith("google/gemma")
        else "vision_model.last_hidden_state"
    )
    if payload.get("module_path") != expected_path:
        raise RuntimeError(
            f"Stale proxy tap for {proxy_model}: expected {expected_path}, "
            f"found {payload.get('module_path')}; rerun `models inspect-taps`"
        )
