#!/usr/bin/env python3
"""Create presentation-ready proxy-selector figures using Pillow only."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = REPO_ROOT / "outputs" / "proxy_selector_cka_v2"
FIGURE_ROOT = REPO_ROOT / "results" / "proxy_selector_ppt_figures"

INK = "#18212F"
MUTED = "#5F6B7A"
GRID = "#DCE2EA"
BLUE = "#246BCE"
ORANGE = "#E58B2A"
LIGHT_BLUE = "#CFE0F7"
WHITE = "#FFFFFF"

SOURCE_NAMES = [
    "0001_ILSVRC2012_val_00015879.png",
    "0003_ILSVRC2012_val_00035897.png",
    "0004_ILSVRC2012_val_00026503.png",
    "0005_ILSVRC2012_val_00025354.png",
    "0006_ILSVRC2012_val_00039508.png",
    "0007_ILSVRC2012_val_00037136.png",
    "0009_ILSVRC2012_val_00049612.png",
    "0012_ILSVRC2012_val_00010012.png",
]

PAIR_LABELS = {
    "P02": "Qwen 4B → Gemma E4B",
    "P06": "CLIP-L → InternVL 2B",
    "P11": "SigLIP2 → Gemma E2B",
    "P14": "Qwen 2B → 4B",
    "P16": "InternVL 2B → 4B",
    "P19": "Gemma E2B → E4B",
    "P20": "Qwen 4B → 2B",
    "P21": "InternVL 4B → 2B",
    "P22": "Gemma E4B → E2B",
}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / filename
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    fill: str = INK,
) -> None:
    box = draw.multiline_textbbox((0, 0), text, font=text_font, align="center")
    width = box[2] - box[0]
    height = box[3] - box[1]
    draw.multiline_text(
        (xy[0] - width / 2, xy[1] - height / 2),
        text,
        font=text_font,
        fill=fill,
        align="center",
        spacing=8,
    )


def save(image: Image.Image, name: str) -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    image.save(FIGURE_ROOT / name, optimize=True)


def make_source_contact_sheet() -> None:
    source_root = OUTPUT_ROOT / "canonical_images" / "source_candidates"
    canvas = Image.new("RGB", (1920, 1080), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text((90, 45), "Controlled Source Images", font=font(52, True), fill=INK)
    draw.text(
        (90, 115),
        "Same 8 clean pickup-truck images used across all 9 proxy–target pairs",
        font=font(27),
        fill=MUTED,
    )
    tile, gap_x, gap_y = 340, 75, 90
    x0, y0 = 115, 188
    for index, name in enumerate(SOURCE_NAMES):
        path = source_root / name
        if not path.exists():
            raise FileNotFoundError(path)
        image = ImageOps.fit(
            Image.open(path).convert("RGB"),
            (tile, tile),
            method=Image.Resampling.LANCZOS,
        )
        col, row = index % 4, index // 4
        x = x0 + col * (tile + gap_x)
        y = y0 + row * (tile + gap_y)
        canvas.paste(image, (x, y))
        image_id = name.split("_")[-1].replace(".png", "")
        draw.text(
            (x, y + tile + 13),
            f"Image {index + 1}  ·  {image_id}",
            font=font(22),
            fill=INK,
        )
    draw.text(
        (90, 1035),
        "Clean criterion: all participating models predict class 8 (pickup truck).",
        font=font(21),
        fill=MUTED,
    )
    save(canvas, "01_controlled_8_sources.png")


def make_attack_triptych() -> None:
    attack_root = (
        OUTPUT_ROOT
        / "attacks/P20/objective_split_all9v2_common48_rho03/batch_00"
        / "semantic_only_rho_0.3/lambda_1"
    )
    clean_image = Image.open(attack_root / "00_clean.png").convert("RGB")
    adv_image = Image.open(attack_root / "00_adv.png").convert("RGB")
    clean = np.asarray(clean_image, dtype=np.float32) / 255.0
    adv = np.asarray(adv_image, dtype=np.float32) / 255.0
    amplified = np.clip(0.5 + 8.0 * (adv - clean), 0.0, 1.0)
    perturbation = Image.fromarray(np.uint8(np.round(amplified * 255)))

    canvas = Image.new("RGB", (1920, 1080), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw_centered(
        draw,
        (960, 70),
        "Successful Targeted Transfer Example: Qwen 4B → Qwen 2B",
        font(46, True),
    )
    draw_centered(
        draw,
        (960, 132),
        "Semantic auxiliary · same frozen 224×224 input · target prediction changes 8 → 7",
        font(25),
        MUTED,
    )
    size = 470
    x_positions = [95, 725, 1355]
    images = [clean_image, adv_image, perturbation]
    headings = ["Clean image", "Adversarial image", "Perturbation ×8"]
    subtitles = [
        "Target: 8 (pickup truck)",
        "Target: 7 (garbage truck)",
        "Gray = zero change",
    ]
    for x, image, heading, subtitle in zip(x_positions, images, headings, subtitles):
        fitted = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
        canvas.paste(fitted, (x, 245))
        draw_centered(draw, (x + size / 2, 765), heading, font(30, True))
        draw_centered(draw, (x + size / 2, 817), subtitle, font(23), MUTED)
    draw.rounded_rectangle((290, 900, 1630, 990), radius=20, fill="#F5F7FA")
    draw_centered(
        draw,
        (960, 944),
        "ρ₀ = 0.3  ·  100 PGD steps  ·  ε = 16/255  ·  displayed PNG remains within the L∞ budget",
        font(23),
        MUTED,
    )
    save(canvas, "02_p20_targeted_transfer_example.png")


def selector_rows() -> list[dict[str, str]]:
    path = (
        OUTPUT_ROOT
        / "diagnostics/selector_analysis_all9v2_common48_rho03"
        / "semantic_only/pair_summary.csv"
    )
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def make_all9_transfer_chart() -> None:
    rows = selector_rows()
    by_pair = {row["pair_id"]: row for row in rows}
    order = ["P20", "P19", "P14", "P22", "P16", "P21", "P02", "P06", "P11"]
    canvas = Image.new("RGB", (1920, 1080), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.text((80, 42), "All-9 Controlled Transfer Results", font=font(48, True), fill=INK)
    draw.text(
        (80, 108),
        "Semantic auxiliary (ρ₀ = 0.3); same 8 clean images across every model pair",
        font=font(25),
        fill=MUTED,
    )
    x0, x1 = 655, 1810
    chart_width = x1 - x0
    for tick in range(0, 101, 20):
        x = x0 + chart_width * tick / 100
        draw.line((x, 190, x, 930), fill=GRID, width=2)
        draw_centered(draw, (x, 965), f"{tick}%", font(19), MUTED)
    y0, step = 212, 80
    bar_h = 22
    for index, pair in enumerate(order):
        row = by_pair[pair]
        tasr = float(row["tasr_percent"])
        asr = 12.5 * float(row["untargeted_hits"])
        y = y0 + index * step
        draw.text((80, y + 8), f"{pair}  {PAIR_LABELS[pair]}", font=font(22), fill=INK)
        asr_end = x0 + chart_width * asr / 100
        tasr_end = x0 + chart_width * tasr / 100
        draw.rectangle((x0, y, asr_end, y + bar_h), fill=LIGHT_BLUE, outline=BLUE, width=2)
        draw.rectangle((x0, y + 31, tasr_end, y + 31 + bar_h), fill=ORANGE)
        draw.text((asr_end + 10, y - 2), f"{asr:.1f}%", font=font(18), fill=INK)
        draw.text((tasr_end + 10, y + 27), f"{tasr:.1f}%", font=font(18), fill=INK)
    draw.rectangle((1210, 92, 1242, 114), fill=LIGHT_BLUE, outline=BLUE, width=2)
    draw.text((1255, 86), "ASR: any departure from class 8", font=font(19), fill=INK)
    draw.rectangle((1210, 128, 1242, 150), fill=ORANGE)
    draw.text((1255, 122), "TASR: target class 7", font=font(19), fill=INK)
    draw.text(
        (80, 1025),
        "Cross-family pairs: P02, P06, P11. Percentages use the common 8-image denominator.",
        font=font(20),
        fill=MUTED,
    )
    save(canvas, "03_all9_semantic_tasr_asr.png")


def make_cka_relationship_chart() -> None:
    rows = selector_rows()
    canvas = Image.new("RGB", (1920, 1080), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw_centered(
        draw,
        (960, 65),
        "Representation Similarity and Targeted Transfer",
        font(47, True),
    )
    draw_centered(
        draw,
        (960, 125),
        "Pair-level association under the semantic attack; N = 9 model pairs",
        font(25),
        MUTED,
    )
    panels = [
        ((105, 210, 910, 900), "Targeted success rate", "TASR (%)", "tasr_percent", 70, 0.818),
        ((1010, 210, 1815, 900), "Target-class gap closure", "Mean gap closure (%)", "gap", 115, 0.869),
    ]
    label_offsets = {
        ("tasr_percent", "P16"): (12, -34),
        ("tasr_percent", "P21"): (12, 5),
        ("gap", "P16"): (12, -34),
        ("gap", "P21"): (12, 5),
        ("gap", "P19"): (-67, -34),
    }
    x_min, x_max = 0.68, 1.01
    for bounds, title, ylabel, field, y_max, rho in panels:
        left, top, right, bottom = bounds
        draw_centered(draw, ((left + right) / 2, top - 38), title, font(29, True))
        for tick in [0.7, 0.8, 0.9, 1.0]:
            x = left + (tick - x_min) / (x_max - x_min) * (right - left)
            draw.line((x, top, x, bottom), fill=GRID, width=2)
            draw_centered(draw, (x, bottom + 28), f"{tick:.1f}", font(18), MUTED)
        y_ticks = list(range(0, y_max + 1, 20))
        for tick in y_ticks:
            y = bottom - tick / y_max * (bottom - top)
            draw.line((left, y, right, y), fill=GRID, width=2)
            draw.text((left - 53, y - 12), f"{tick}", font=font(17), fill=MUTED)
        draw.line((left, top, left, bottom), fill=INK, width=2)
        draw.line((left, bottom, right, bottom), fill=INK, width=2)
        draw_centered(draw, ((left + right) / 2, bottom + 70), "Proxy–target global linear CKA", font(21))
        draw.text((left + 18, top + 18), f"Spearman ρ = {rho:.3f}", font=font(21, True), fill=INK)
        draw.text((left + 18, top + 51), "N = 9 pairs", font=font(18), fill=MUTED)
        for row in rows:
            px = float(row["global_cka"])
            py = (
                float(row["tasr_percent"])
                if field == "tasr_percent"
                else 100.0 * float(row["mean_gap_closure"])
            )
            x = left + (px - x_min) / (x_max - x_min) * (right - left)
            y = bottom - py / y_max * (bottom - top)
            if row["pair_type"] == "Intra-Family":
                draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=BLUE, outline=WHITE, width=2)
            else:
                draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=WHITE, outline=ORANGE, width=4)
            dx, dy = label_offsets.get((field, row["pair_id"]), (12, -22))
            draw.text((x + dx, y + dy), row["pair_id"], font=font(17, True), fill=INK)
        draw.text((left - 83, top - 3), ylabel, font=font(18), fill=MUTED)
    draw.ellipse((1260, 955, 1280, 975), fill=BLUE)
    draw.text((1290, 950), "Intra-family", font=font(19), fill=INK)
    draw.ellipse((1460, 955, 1480, 975), fill=WHITE, outline=ORANGE, width=4)
    draw.text((1490, 950), "Cross-family", font=font(19), fill=INK)
    draw.text(
        (105, 1018),
        "Descriptive association for one 8→7 transition; CKA alone does not guarantee that a pair crosses the target boundary.",
        font=font(20),
        fill=MUTED,
    )
    save(canvas, "04_cka_vs_targeted_transfer.png")


def write_metadata() -> None:
    target_path = (
        OUTPUT_ROOT
        / "diagnostics/objective_split_all9v2_common48_rho03/target_outputs"
        / "P20__original__lambda_1__alpha_0__beta_1__objective_semantic_only__rho_0.3.json"
    )
    target = json.loads(target_path.read_text())
    metadata = {
        "source_contact_sheet_images": SOURCE_NAMES,
        "triptych_pair": "P20",
        "triptych_image_index": 0,
        "triptych_clean_label": target["clean_outputs"][0]["parsed_label"],
        "triptych_adversarial_label": target["adversarial_outputs"][0]["parsed_label"],
        "all9_objective": "semantic_only",
        "all9_image_count": 8,
    }
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    (FIGURE_ROOT / "figure_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    make_source_contact_sheet()
    make_attack_triptych()
    make_all9_transfer_chart()
    make_cka_relationship_chart()
    write_metadata()
    for path in sorted(FIGURE_ROOT.glob("*.png")):
        print(path.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
