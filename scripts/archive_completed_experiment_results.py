#!/usr/bin/env python3
"""Archive compact completed experiment artifacts into tracked result bundles."""

from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise FileNotFoundError(source)
    shutil.copytree(source, destination, dirs_exist_ok=True)


def archive_original_token_cka() -> None:
    source_root = ROOT / "outputs/proxy_selector_cka_v2/diagnostics"
    destination = ROOT / "results/original_token_cka_transfer_search"
    stages = (
        "original_token_cka_wide_scan",
        "original_token_cka_mid_transfer",
        "original_token_cka_full100",
        "original_token_cka_reference48",
    )
    for stage in stages:
        stage_source = source_root / stage
        copy_file(stage_source / "summary.csv", destination / stage / "summary.csv")
        copy_tree(stage_source / "trials", destination / stage / "trials")
        target_outputs = stage_source / "target_outputs"
        if target_outputs.is_dir():
            copy_tree(target_outputs, destination / stage / "target_outputs")
    copy_tree(
        source_root / "original_token_cka_generated_configs",
        destination / "selected_configs",
    )
    copy_tree(
        source_root / "original_token_cka_transfer_search_summary",
        destination / "combined",
    )


def archive_scale50() -> None:
    source = ROOT / "outputs/proxy_selector_cka_v2_scale50"
    destination = ROOT / "results/scale50_semantic_all9"
    copy_file(
        source / "summaries/scale_50_semantic_all9.csv",
        destination / "scale_50_semantic_all9.csv",
    )
    copy_file(
        source / "summaries/scale_50_results.csv",
        destination / "last_pair_batch_results.csv",
    )
    copy_file(source / "environment.json", destination / "environment.json")
    copy_file(source / "summaries/test_report.txt", destination / "test_report.txt")
    copy_tree(source / "logs", destination / "attack_logs")
    copy_tree(source / "evaluation/scale_50", destination / "target_evaluations")


def main() -> None:
    archive_original_token_cka()
    archive_scale50()
    print(ROOT / "results/original_token_cka_transfer_search")
    print(ROOT / "results/scale50_semantic_all9")


if __name__ == "__main__":
    main()
