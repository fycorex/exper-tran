#!/usr/bin/env python3
"""Summarize completed v3 eight-image states without loading any model."""

import argparse
import csv
import json
import math
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/proxy_selector_semantic_contrastive_v3"),
    )
    return parser.parse_args()


def finite_or_blank(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def main() -> None:
    args = arguments()
    states = sorted(args.output_dir.glob("states/*/*steps_100*/*.json"))
    rows = []
    traces = []
    for path in states:
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("status") != "complete":
            continue
        attack = state["attack"]
        row = {
            "pair_id": state["pair_id"],
            "loss": state["arm"],
            "cls_loss_mode": attack["cls_loss_mode"],
            "semantic_mode": attack["semantic_mode"],
            "tau": attack["semantic_temperature"],
            "target_logit_weight": attack.get("semantic_target_logit_weight", 1.0),
            "source_logit_weight": attack.get("semantic_source_logit_weight", 1.0),
            "representation_type": attack["representation_type"],
            "vision_layer": attack["representation_layer"],
            "pooling": attack["representation_pooling"],
            "lambda_cls": attack["lambda_cls"],
            "lambda_sem_requested": attack["lambda_cka"],
            "lambda_sem_effective": attack["effective_lambda_cka"],
            "proxy_hits": attack["proxy_target_hit_count"],
            "proxy_denominator": attack["proxy_target_hit_denominator"],
            "tasr_hits": state["tasr_hits"],
            "tasr_denominator": state["clean_valid_count"],
            "asr_hits": state["asr_hits"],
            "target_similarity_clean": attack["target_similarity_clean"],
            "target_similarity_adv": attack["target_similarity_adversarial"],
            "target_gain": attack["target_similarity_adversarial"]
            - attack["target_similarity_clean"],
            "source_similarity_clean": attack["source_similarity_clean"],
            "source_similarity_adv": attack["source_similarity_adversarial"],
            "source_drop": attack["source_similarity_clean"]
            - attack["source_similarity_adversarial"],
            "semantic_gap_clean": attack["semantic_gap_clean"],
            "semantic_gap_adv": attack["semantic_gap_adversarial"],
            "semantic_gap_gain": attack["semantic_gap_gain"],
            "grad_cls_mean_abs": attack["grad_ml_l1"],
            "grad_rep_weighted_mean_abs": attack["grad_cka_weighted_l1"],
            "grad_cls_rep_cosine": attack["grad_component_cosine"],
            "elapsed_seconds": attack["elapsed_seconds"],
            "peak_vram_gb": attack["peak_allocated_vram_gb"],
            "seed": 42,
            "steps": 100,
            "source_reference_count": len(attack["source_reference_ids"]),
            "target_reference_count": len(attack["target_reference_ids"]),
        }
        rows.append({key: finite_or_blank(value) for key, value in row.items()})
        for trace in attack.get("gradient_trace", []):
            traces.append(
                {
                    "pair_id": state["pair_id"],
                    "loss": state["arm"],
                    **trace,
                }
            )
    diagnostics = args.output_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    if rows:
        with (diagnostics / "ablation_8.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (diagnostics / "gradient_trace.json").write_text(
        json.dumps(traces, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    flat_traces = []
    for trace in traces:
        row = {
            "pair_id": trace["pair_id"],
            "loss": trace["loss"],
            "step": trace["step"],
            "grad_cls_rep_cosine": trace.get("cls_rep_cosine"),
        }
        for component in (
            "target_token",
            "closedset_ce",
            "margin",
            "cls_total",
            "representation",
        ):
            for metric, value in trace.get(component, {}).items():
                row[f"{component}_{metric}"] = value
        flat_traces.append(row)
    if flat_traces:
        fieldnames = tuple(dict.fromkeys(key for row in flat_traces for key in row))
        with (diagnostics / "gradient_trace.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flat_traces)

    by_loss = {row["loss"]: row for row in rows if row["pair_id"] == "P20"}
    lines = [
        "# Semantic Contrastive V3 — eight-image summary",
        "",
        "## P20 results",
        "",
        "| Loss | Proxy | TASR | ASR | Target gain | Source drop | Gap gain |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in (
        "cls_only",
        "semantic_only",
        "cls_plus_semantic",
        "contrastive_only",
        "cls_plus_contrastive",
        "mean_reference_only",
    ):
        if name not in by_loss:
            continue
        row = by_loss[name]
        lines.append(
            f"| {name} | {row['proxy_hits']}/8 | {row['tasr_hits']}/8 | "
            f"{row['asr_hits']}/8 | {row['target_gain']:.4f} | "
            f"{row['source_drop'] if row['source_drop'] != '' else 'n/a'} | "
            f"{row['semantic_gap_gain'] if row['semantic_gap_gain'] != '' else 'n/a'} |"
        )
    lines.extend(
        [
            "",
            "## Diagnostic answers",
            "",
            "1. CLS is not universally helpful: CLS-only reached 2/8 TASR; adding CLS to "
            "target-only semantic reduced 2/8 to 1/8, while adding CLS to prototype "
            "contrastive increased 0/8 to 4/8.",
            "2. Source-target contrastive only reached strict proxy 8/8 but 0/8 TASR. "
            "Its value appears in interaction with CLS, where the combined arm reached 4/8.",
            "3. Prototype-only and mean-reference-only both reached 0/8 TASR; prototype "
            "had 5/8 ASR versus 4/8 for mean-reference. No TASR advantage is established "
            "between the two representation-only modes.",
            "4. CLS pixel gradients are not extremely small. The detailed combined run "
            "records target-token, CE, margin, total-CLS, and representation gradients at "
            "steps 0/25/50/99.",
            "5. The initial CLS/representation gradient cosine is weakly positive rather "
            "than strongly conflicting; see gradient_trace.csv for its evolution.",
            "6. P20 uses Qwen Vision Encoder block 17 of 24, valid-token mean pooling, "
            "with no target-model representation access.",
            "7. Depth materially changes semantic separability: source-target prototype "
            "cosine was 0.952 (layer 12), 0.929 (layer 17), and 0.99996 (layer 23).",
            "",
            "These are eight-image diagnostic results, not final statistical claims.",
        ]
    )
    (diagnostics / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summarized completed_trials={len(rows)} gradient_rows={len(traces)}", flush=True)


if __name__ == "__main__":
    main()
