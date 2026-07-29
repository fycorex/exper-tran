import json
from dataclasses import dataclass
from pathlib import Path

from primary_ml_cka.config.loader import load_config
from primary_ml_cka.config.schema import AttackConfig, DataConfig
from primary_ml_cka.config.validation import validate_attack_config, validate_data_config


@dataclass(frozen=True, slots=True)
class CommandContext:
    project_root: Path
    output_dir: Path
    pair_id: str | None
    resume: bool
    dry_run: bool
    seed: int | None
    config_path: Path | None


def resolve_attack_config(context: CommandContext) -> AttackConfig:
    path = context.config_path or (
        context.project_root / "configs" / "attacks" / "primary_ml_cka.yaml"
    )
    raw = load_config(path)
    values = {key: raw[key] for key in AttackConfig.__dataclass_fields__ if key in raw}
    if "lambdas" in values:
        values["lambdas"] = tuple(float(value) for value in values["lambdas"])
    config = AttackConfig(**values)
    validate_attack_config(config)
    return config


def resolve_data_config(context: CommandContext) -> DataConfig:
    path = context.project_root / "configs" / "data" / "imagenet_vehicle10.yaml"
    raw = load_config(path)
    values = {key: raw[key] for key in DataConfig.__dataclass_fields__ if key in raw}
    config = DataConfig(**values)
    validate_data_config(config)
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
