"""Recreate the Visual Genome quality–latency figure from the results CSV."""

import argparse
import csv
from html import escape
from math import ceil, floor, isfinite
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "baseline": ("Baseline", 9, -8),
    "fixed_detail": ("Fixed detail", 8, 15),
    "current_best": ("Previous best", 8, -12),
    "D_asymmetric": ("Quality", -8, -12),
    "G_selective_boundary": ("Balanced", -9, 20),
    "C_visual_dispersion": ("Efficient", -5, -12),
    "B_best_large_fallback": ("Large-target fallback", -10, -9),
    "size_only": ("Size-aware only", 8, 16),
    "boundary_only": ("Boundary–semantic only", 8, -12),
}
HIGHLIGHTS = {"baseline", "D_asymmetric", "G_selective_boundary", "C_visual_dispersion", "B_best_large_fallback"}


def render(rows):
    """Return a standalone SVG including the exact values in each point's tooltip."""
    quality_counts = {int(row["quality_regions"]) for row in rows}
    timing_counts = {int(row["timing_regions"]) for row in rows}
    if len(quality_counts) != 1 or len(timing_counts) != 1:
        raise ValueError("Expected one shared quality and timing cohort in the CSV")
    quality_count = quality_counts.pop()
    timing_count = timing_counts.pop()
    points = [(row, float(row["mean_online_ms_per_region"]), float(row["CIDEr"])) for row in rows]
    if any(not (isfinite(ms) and isfinite(score) and ms > 0 and score >= 0) for _, ms, score in points):
        raise ValueError("Expected finite, positive latency and finite, nonnegative CIDEr")
    first_ms = 20 * floor(min(ms for _, ms, _ in points) / 20)
    last_ms = max(first_ms + 20, 20 * ceil(max(ms for _, ms, _ in points) / 20))
    first_score = floor(100 * min(score for _, _, score in points))
    last_score = ceil(100 * max(score for _, _, score in points))

    def x(value):
        return 72 + (value - first_ms) / (last_ms - first_ms) * 635

    def y(value):
        return 371 - (100 * value - first_score + 0.5) / (last_score - first_score + 1) * 277

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 758 449" role="img" aria-labelledby="title desc">',
        '<title id="title">Visual Genome test quality and latency</title>',
        f'<desc id="desc">CIDEr scores for {quality_count:,} held-out regions versus online latency from the {timing_count:,}-region timing audit.</desc>',
        '<rect width="758" height="449" fill="#fff"/>',
        '<g font-family="system-ui, sans-serif" font-size="13" fill="#333f4d">',
        '<text x="32" y="36" font-size="22" font-weight="650" fill="#172338">Visual Genome test quality and latency</text>',
        f'<text x="32" y="61" fill="#53657a">{quality_count:,} held-out regions · {timing_count:,}-region online timing audit</text>',
    ]
    for ms in range(first_ms, last_ms + 1, 20 * max(1, ceil((last_ms - first_ms) / 120))):
        position = x(ms)
        svg.extend(
            (f'<path d="M{position:.1f} 94 V371" stroke="#e6ebf0"/>',
             f'<text x="{position:.1f}" y="392" text-anchor="middle">{ms}</text>')
        )
    for tick in range(first_score, last_score + 1, max(1, ceil((last_score - first_score) / 6))):
        score = tick / 100
        position = y(score)
        svg.extend(
            (f'<path d="M72 {position:.1f} H707" stroke="#e6ebf0"/>',
             f'<text x="62" y="{position + 4:.1f}" text-anchor="end">{score:.2f}</text>')
        )
    svg.extend((
        '<path d="M72 94 V371 H707" fill="none" stroke="#95a3b3"/>',
        '<text x="390" y="425" text-anchor="middle">Mean inference time per region (ms)</text>',
        '<text x="22" y="231" text-anchor="middle" transform="rotate(-90,22,231)">CIDEr</text>',
    ))
    for row, ms, score in points:
        method = row["method_id"]
        label, dx, dy = LABELS.get(method, (row["method"], 8, -12))
        px, py = x(ms), y(score)
        color = "#13675d" if method in HIGHLIGHTS else "#65798e"
        anchor = "end" if dx < 0 else "start"
        svg.extend((
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}" stroke="#fff">'
            f'<title>{escape(label)}: {score:.5f} CIDEr, {ms:.1f} ms</title></circle>',
            f'<text x="{px + dx:.1f}" y="{py + dy:.1f}" text-anchor="{anchor}" font-size="11" fill="{color}">{escape(label)}</text>',
        ))
    return "\n".join([*svg, "</g>", "</svg>", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "main_results.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "quality_latency.svg")
    args = parser.parse_args()
    try:
        with args.input.open(encoding="utf-8-sig", newline="") as stream:
            svg = render(list(csv.DictReader(stream)))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(svg, encoding="utf-8", newline="\n")
    except (OSError, KeyError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
