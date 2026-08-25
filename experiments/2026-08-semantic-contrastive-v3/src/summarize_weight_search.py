#!/usr/bin/env python3
"""Summarize completed P21/P22 weighted-contrastive screening trials."""

import argparse
import csv
import json
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/proxy_selector_semantic_contrastive_v3"),
    )
    parser.add_argument("--steps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    rows: list[dict[str, object]] = []
    pattern = f"states/*/*steps_{args.steps}_*/*.json"
    for path in sorted(args.output_dir.glob(pattern)):
        state = json.loads(path.read_text(encoding="utf-8"))
        if state.get("status") != "complete" or not str(state.get("arm", "")).startswith("ws_"):
            continue
        attack = state["attack"]
        rows.append(
            {
                "pair_id": state["pair_id"],
                "arm": state["arm"],
                "rho": attack["gradient_ratio"],
                "effective_lambda_sem": attack["effective_lambda_cka"],
                "lambda_cls": attack["lambda_cls"],
                "target_logit_weight": attack.get("semantic_target_logit_weight", 1.0),
                "source_logit_weight": attack.get("semantic_source_logit_weight", 1.0),
                "proxy_hits": attack["proxy_target_hit_count"],
                "tasr_hits": state["tasr_hits"],
                "asr_hits": state["asr_hits"],
                "target_gain": attack["target_similarity_adversarial"]
                - attack["target_similarity_clean"],
                "source_drop": attack["source_similarity_clean"]
                - attack["source_similarity_adversarial"],
                "gap_gain": attack["semantic_gap_gain"],
                "min_proxy_margin": attack["proxy_min_target_logit_margin"],
                "min_proxy_probability": attack["proxy_min_target_probability"],
                "elapsed_seconds": attack["elapsed_seconds"],
                "peak_vram_gb": attack["peak_allocated_vram_gb"],
            }
        )
    rows.sort(
        key=lambda row: (
            str(row["pair_id"]),
            -int(row["tasr_hits"]),
            -int(row["proxy_hits"]),
            -float(row["target_gain"]),
            -int(row["asr_hits"]),
        )
    )
    diagnostics = args.output_dir / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    csv_path = diagnostics / f"weight_search_{args.steps}.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        f"# Weighted contrastive search ({args.steps} steps)",
        "",
        "Rank order is TASR, proxy hits, target attraction, then ASR. Screening results are not final confirmations.",
        "",
        "| Pair | Arm | rho | target/source | Proxy | TASR | ASR | Target gain | Source drop | Gap gain |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['pair_id']} | {row['arm']} | {row['rho']} | "
            f"{row['target_logit_weight']}/{row['source_logit_weight']} | "
            f"{row['proxy_hits']}/8 | {row['tasr_hits']}/8 | {row['asr_hits']}/8 | "
            f"{float(row['target_gain']):.4f} | {float(row['source_drop']):.4f} | "
            f"{float(row['gap_gain']):.4f} |"
        )
    markdown_path = diagnostics / f"weight_search_{args.steps}.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"summarized weight-search trials={len(rows)} to {csv_path}", flush=True)


if __name__ == "__main__":
    main()
