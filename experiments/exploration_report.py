"""Summarize completed Patch-ioner exploration stages without running inference."""

import argparse
from collections import defaultdict
import csv
from html import escape
import json
import math
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile

import numpy as np


GROUPS = ("all", "small", "medium", "large")
METRIC_FIELDS = ("CIDEr", "BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4", "METEOR", "ROUGE_L", "RefPAC-S")
GROUP_LABELS = {"all": "overall", "small": "small", "medium": "medium", "large": "large"}
FAMILY_COLORS = dict(zip("ABCDEFGHIJ", (
    "#0064a4", "#0f766e", "#9348ae", "#bf5425", "#3353b5",
    "#9e2d66", "#116b40", "#78522b", "#736000", "#9d2933",
)))
FACTOR_DEFAULTS = {
    "weight": 0.3, "weight_rule": "fixed", "detail_scale": 1.2, "detail_resolution": 518,
    "detail_pool": "overlap", "original_pool": "official", "context_weight": 0,
    "context_scale": 2.0, "context_pool": "uniform", "context_medium_only": False,
    "size_signal": "patch_count", "scale_rule": "mean", "gate": None, "gate_threshold": None,
}
CONFIG_LABELS = {"id", "family", "categories", "label", "description"}


def read_input(argument):
    if "=" in argument:
        stage, filename = argument.split("=", 1)
        path = Path(filename)
    else:
        stage = None
        path = Path(argument)
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    return stage or document.get("metadata", {}).get("stage") or path.parent.name, document


def config_for(configs, mode):
    config = configs.get(mode, {})
    if not isinstance(config, dict):
        raise ValueError(f"The configuration for {mode} must be a mapping")
    family = config.get("family") or ("baseline" if mode == "baseline" else "other")
    return family, config


def event_rate(rows):
    flags = [row.get("detail_active") for row in rows]
    if not flags or any(flag is None for flag in flags):
        return None
    return mean(bool(flag) for flag in flags)


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def cached_decode_mean(rows):
    durations = [row.get("cache_decode_ms") for row in rows]
    if not durations or not all(is_number(duration) for duration in durations):
        return None
    return mean(durations)


def index_rows(rows):
    indexed = defaultdict(dict)
    for row in rows:
        key = str(row["region_id"])
        mode = row["mode"]
        if key in indexed[mode]:
            raise ValueError(f"Duplicate prediction for {mode}/{key}")
        indexed[mode][key] = row
    return indexed


def rank_stage(stage, document, online_measurements):
    metadata = document.get("metadata", {})
    configs = metadata.get("configs", {})
    latency_kind = metadata.get("latency_kind", "online" if "timing" in metadata else "unspecified")
    rows_by_mode = index_rows(document.get("rows", []))
    if "baseline" not in rows_by_mode:
        raise ValueError(f"{stage} requires per-region baseline predictions")
    if any(rows.keys() != rows_by_mode["baseline"].keys() for rows in rows_by_mode.values()):
        raise ValueError(f"Unpaired per-region predictions in {stage}")
    grouped_predictions = defaultdict(list)
    for mode, rows in rows_by_mode.items():
        for row in rows.values():
            grouped_predictions[mode, "all"].append(row)
            grouped_predictions[mode, row["area_group"]].append(row)
    baselines = {row["area_group"]: row for row in document["summary"] if row["mode"] == "baseline"}
    grouped = defaultdict(list)
    seen = set()
    for row in document["summary"]:
        mode, group = row["mode"], row["area_group"]
        if (mode, group) in seen or group not in GROUPS:
            raise ValueError(f"Duplicate or unsupported summary for {stage}/{mode}/{group}")
        seen.add((mode, group))
        predictions = grouped_predictions[mode, group]
        if row.get("regions", len(predictions)) != len(predictions):
            raise ValueError(f"Region count does not match per-region predictions for {stage}/{mode}/{group}")
        family, config = config_for(configs, mode)
        measured = online_measurements.get((stage, mode, group), row if latency_kind == "online" else {})
        cached_ms = cached_decode_mean(predictions)
        if cached_ms is None and latency_kind == "cached":
            cached_ms = row.get("mean_inference_ms")
        activation = measured.get("detail_activation_rate", event_rate(predictions))
        base_score = baselines.get(group, {}).get("CIDEr")
        delta = row["CIDEr"] - base_score if is_number(row.get("CIDEr")) and is_number(base_score) else None
        grouped[group].append({
            "stage": stage,
            "area_group": group,
            "mode": mode,
            "family": family,
            "regions": len(predictions),
            **{metric: row[metric] for metric in METRIC_FIELDS if metric in row},
            "delta_CIDEr": delta,
            "detail_activation_rate": activation,
            "cached_decode_ms": cached_ms,
            "cached_peak_allocated_mib": row.get("peak_allocated_mib") if latency_kind == "cached" else None,
            "online_mean_ms": measured.get("mean_inference_ms"),
            "online_peak_allocated_mib": measured.get("peak_allocated_mib"),
            "online_regions": measured.get("regions", len(predictions)) if measured else None,
            "config": config,
        })
    ordered = []
    for group in GROUPS:
        missing = set(rows_by_mode) - {row["mode"] for row in grouped[group]}
        if missing:
            raise ValueError(f"Missing scored {group} rows for {stage}: {', '.join(sorted(missing))}")
        rows = sorted(grouped[group], key=lambda row: (
            -(row["CIDEr"] if is_number(row.get("CIDEr")) else -math.inf), row["mode"]
        ))
        ordered.extend({"rank": i, **row} for i, row in enumerate(rows, start=1))
    return ordered, rows_by_mode


def paired_bootstrap(stage, ranked, rows_by_mode, seed, draws, top):
    """Resample images as clusters, carrying every selected region on an image."""
    baseline = rows_by_mode["baseline"]
    result = []
    for offset, group in enumerate(GROUPS):
        keys = [key for key, row in baseline.items() if group == "all" or row["area_group"] == group]
        images = sorted({str(baseline[key]["image_id"]) for key in keys})
        if not images:
            continue
        modes = [row["mode"] for row in ranked if row["area_group"] == group
                 and row["mode"] != "baseline" and is_number(row.get("CIDEr"))]
        if top:
            modes = modes[:top]
        if not modes:
            continue
        if any(rows_by_mode[mode].keys() != baseline.keys() for mode in modes):
            raise ValueError(f"Unpaired region keys in {stage}/{group}")
        image_index = {image: i for i, image in enumerate(images)}
        selected = np.array([image_index[str(baseline[key]["image_id"])] for key in keys])
        count = np.bincount(selected, minlength=len(images))
        base_scores = np.array([baseline[key]["CIDEr"] for key in keys])
        sums = []
        for mode in modes:
            differences = np.array([rows_by_mode[mode][key]["CIDEr"] for key in keys]) - base_scores
            sums.append(np.bincount(selected, weights=differences, minlength=len(images)))
        rng = np.random.default_rng(seed + offset)
        samples = rng.multinomial(len(images), np.full(len(images), 1 / len(images)), size=draws)
        estimates = (samples @ np.asarray(sums).T) / (samples @ count)[:, None]
        lower, upper = np.percentile(estimates, [2.5, 97.5], axis=0)
        for i, mode in enumerate(modes):
            result.append({
                "stage": stage,
                "area_group": group,
                "mode": mode,
                "images": len(images),
                "regions": len(keys),
                "delta_CIDEr": float(sums[i].sum() / len(keys)),
                "ci95_lower": float(lower[i]),
                "ci95_upper": float(upper[i]),
                "resamples": draws,
                "seed": seed + offset,
            })
    return result


def find_examples(stage, rows_by_mode, per_direction):
    baseline = rows_by_mode["baseline"]
    examples = []
    for mode, rows in rows_by_mode.items():
        if mode == "baseline":
            continue
        paired = []
        for key, row in rows.items():
            base = baseline[key]
            if row["caption"] == base["caption"] or not all(
                is_number(item.get("CIDEr")) for item in (row, base)
            ):
                continue
            paired.append((row["CIDEr"] - base["CIDEr"], key, row, base))
        for direction, reverse in (("improvement", True), ("regression", False)):
            valid = [item for item in paired if (item[0] > 0 if reverse else item[0] < 0)]
            for delta, key, row, base in sorted(valid, key=lambda item: (item[0], item[1]), reverse=reverse)[:per_direction]:
                examples.append({
                    "stage": stage, "mode": mode, "example_type": direction, "region_id": key,
                    "image_id": row["image_id"], "area_group": row["area_group"],
                    "image_path": row.get("image_path"), "bbox_xywh": row.get("bbox_xywh"),
                    "references": row.get("references"), "baseline_caption": base["caption"],
                    "caption": row["caption"], "baseline_CIDEr": base["CIDEr"],
                    "CIDEr": row["CIDEr"], "delta_CIDEr": delta,
                })
    return examples


def single_factor_sweeps(leaderboard):
    """Find compared settings that vary exactly one known configuration field."""
    buckets = defaultdict(list)
    for row in leaderboard:
        if not row["config"] or not is_number(row.get("CIDEr")):
            continue
        config = {**FACTOR_DEFAULTS, **{k: v for k, v in row["config"].items() if k not in CONFIG_LABELS}}
        families = {row["family"], *row["config"].get("categories", [])}
        if config["weight"] == 0.3 and row["mode"] == "A_w0p3":
            families.update(("B", "G"))  # The grid shares this default-detail anchor.
        if row["mode"] == "baseline":
            families.update(("F", "G"))  # Original pooling and zero background anchors.
        for parameter, setting in config.items():
            fixed = json.dumps({key: value for key, value in config.items() if key != parameter}, sort_keys=True)
            for family in sorted(families):
                buckets[row["stage"], row["area_group"], family, parameter, fixed].append((row, setting))

    output = []
    for (stage, group, family, parameter, fixed), entries in buckets.items():
        if len({json.dumps(setting, sort_keys=True) for _, setting in entries}) < 2:
            continue
        available = [row["CIDEr"] for row, _ in entries]
        for row, setting in sorted(entries, key=lambda pair: (-pair[0]["CIDEr"], pair[0]["mode"])):
            output.append({"stage": stage, "area_group": group, "family": family, "parameter": parameter,
                           "setting": json.dumps(setting, ensure_ascii=False), "other_settings": fixed,
                           "mode": row["mode"], "CIDEr": row["CIDEr"], "delta_CIDEr": row["delta_CIDEr"],
                           "CIDEr_range": max(available) - min(available), "is_best": row["CIDEr"] == max(available)})
    return output


def write_text(path, render):
    with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        render(stream)
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(key for row in rows for key in row))

    def render(stream):
        # A UTF-8 signature lets Windows Excel recognize non-ASCII descriptions.
        stream.write("\ufeff")
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})

    write_text(path, render)


def quality_cost(rows):
    """Render separate stage and timing facets directly as vector graphics."""
    facets = defaultdict(list)
    for row in rows:
        if row["area_group"] == "all" and is_number(row.get("CIDEr")):
            for timing, title in (("online_mean_ms", "Online inference"), ("cached_decode_ms", "Cached decoding")):
                if is_number(row.get(timing)) and row[timing] > 0:
                    facets[row["stage"], timing, title].append(row)
    width = 840
    if not facets:
        return '<svg xmlns="http://www.w3.org/2000/svg" width="840" height="80"><text x="20" y="45">No latency measurements were supplied.</text></svg>\n'
    height = 65 + 340 * len(facets)
    output = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
              '<rect width="100%" height="100%" fill="#fff"/>',
              '<g font-family="Arial, sans-serif" fill="#18263c">',
              '<text x="28" y="29" font-size="21" font-weight="bold">Patch-ioner quality and latency</text>',
              '<text x="28" y="47" font-size="11" fill="#4b6077">Each panel has its own axes. Hover over a point for its exact configuration.</text>']
    for index, ((stage, timing, title), panel) in enumerate(facets.items()):
        top = 75 + 340 * index
        left, right = 85, 804
        first_y, last_y = top + 28, top + 261
        xmax, xmin = max(row[timing] for row in panel), min(row[timing] for row in panel)
        ymax, ymin = max(row["CIDEr"] for row in panel), min(row["CIDEr"] for row in panel)
        xpad = max((xmax - xmin) * 0.08, 0.1)
        ypad = max((ymax - ymin) * 0.08, 0.001)
        xmin, xmax = max(0, xmin - xpad), xmax + xpad
        ymin, ymax = max(0, ymin - ypad), ymax + ypad
        output.append(f'<text x="28" y="{top + 12}" font-size="15" font-weight="bold">{escape(stage)} · {title}</text>')
        for i in range(6):
            x = left + (right - left) * i / 5
            y = last_y - (last_y - first_y) * i / 5
            x_tick = xmin + (xmax - xmin) * i / 5
            y_tick = ymin + (ymax - ymin) * i / 5
            output.extend((
                f'<path d="M {left} {y:.2f} H {right} M {x:.2f} {first_y} V {last_y}" stroke="#e3e9ef"/>',
                f'<text x="{left - 9}" y="{y + 4:.2f}" text-anchor="end" font-size="11">{y_tick:.3f}</text>',
                f'<text x="{x:.2f}" y="{last_y + 18}" text-anchor="middle" font-size="11">{x_tick:.1f}</text>',
            ))
        output.extend((
            f'<path d="M {left} {first_y} V {last_y} H {right}" fill="none" stroke="#708296"/>',
            f'<text x="{(left + right) / 2}" y="{last_y + 40}" text-anchor="middle" font-size="12">Milliseconds per region</text>',
            f'<text x="24" y="{(first_y + last_y) / 2}" transform="rotate(-90 24 {(first_y + last_y) / 2})" text-anchor="middle" font-size="12">CIDEr</text>',
        ))
        for row in panel:
            x = left + (right - left) * (row[timing] - xmin) / (xmax - xmin)
            y = last_y - (last_y - first_y) * (row["CIDEr"] - ymin) / (ymax - ymin)
            color = FAMILY_COLORS.get(str(row["family"]).upper(), "#26364b")
            tip = escape(f'{row["mode"]}: {row["CIDEr"]:.5f} CIDEr, {row[timing]:.2f} ms')
            radius = 6 if row["mode"] == "baseline" else 4
            output.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}" stroke="#fff"><title>{tip}</title></circle>')
            if row["mode"] == "baseline" or row["rank"] == 1:
                label = row["mode"]
                anchor = "end" if x > 690 else "start"
                label_x = x - 7 if anchor == "end" else x + 7
                output.append(f'<text x="{label_x:.2f}" y="{y - 6:.2f}" text-anchor="{anchor}" font-size="11">{escape(label)}</text>')
    output.append("</g></svg>\n")
    return "\n".join(output)


def build_report(score_inputs, online_inputs, output, seed, draws, bootstrap_top, example_count):
    online = {}
    for stage, document in online_inputs:
        for item in document["summary"]:
            key = (stage, item["mode"], item.get("area_group", "all"))
            if key in online:
                raise ValueError(f"Duplicate online measurement for {key}")
            online[key] = item
    leaderboard, intervals, examples = [], [], []
    seen_stages = set()
    for stage, document in score_inputs:
        if stage in seen_stages:
            raise ValueError(f"Use a single merged scores file for stage {stage}")
        seen_stages.add(stage)
        ranked, predictions = rank_stage(stage, document, online)
        leaderboard.extend(ranked)
        intervals.extend(paired_bootstrap(stage, ranked, predictions, seed, draws, bootstrap_top))
        examples.extend(find_examples(stage, predictions, example_count))

    ci_by_key = {(row["stage"], row["area_group"], row["mode"]): row for row in intervals}
    for row in leaderboard:
        bounds = ci_by_key.get((row["stage"], row["area_group"], row["mode"]))
        row["ci95_lower"] = bounds["ci95_lower"] if bounds else None
        row["ci95_upper"] = bounds["ci95_upper"] if bounds else None
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "leaderboard.csv", leaderboard)
    for group, label in GROUP_LABELS.items():
        write_csv(output / f"leaderboard_{label}.csv", [row for row in leaderboard if row["area_group"] == group])
    sensitivity = sorted(leaderboard, key=lambda row: (row["stage"], row["family"], row["mode"], GROUPS.index(row["area_group"])))
    write_csv(output / "sensitivity.csv", [
        {**row, **{f"parameter.{key}": value for key, value in row["config"].items() if key not in CONFIG_LABELS}}
        for row in sensitivity
    ])
    write_csv(output / "sensitivity_sweeps.csv", single_factor_sweeps(leaderboard), (
        "stage", "area_group", "family", "parameter", "setting", "other_settings", "mode", "CIDEr",
        "delta_CIDEr", "CIDEr_range", "is_best",
    ))
    write_csv(output / "paired_bootstrap.csv", intervals, (
        "stage", "area_group", "mode", "images", "regions", "delta_CIDEr", "ci95_lower", "ci95_upper", "resamples", "seed",
    ))
    write_csv(output / "examples.csv", examples, (
        "stage", "mode", "example_type", "region_id", "image_id", "area_group", "image_path", "bbox_xywh", "references",
        "baseline_caption", "caption", "baseline_CIDEr", "CIDEr", "delta_CIDEr",
    ))
    write_text(output / "leaderboard.json", lambda stream: json.dump({
        "comparison": "Each ranking uses the regions and IDF fitted within its own stage."
                      " Paired confidence intervals resample images and hold IDF fixed.",
        "bootstrap_top_per_group": bootstrap_top,
        "rows": leaderboard,
    }, stream, ensure_ascii=False, allow_nan=False, indent=2))
    write_text(output / "quality_cost.svg", lambda stream: stream.write(quality_cost(leaderboard)))
    return leaderboard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", metavar="[STAGE=]PATH", required=True, action="append")
    parser.add_argument("--online", metavar="[STAGE=]PATH", action="append", default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-top", type=int, default=5, help="Candidates per stage and size group, or 0 for all")
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--examples-per-direction", type=int, default=2)
    args = parser.parse_args()
    if args.bootstrap_resamples < 1 or min(args.bootstrap_top, args.examples_per_direction, args.seed) < 0:
        parser.error("Use positive resamples and nonnegative other numeric arguments")
    rows = build_report([read_input(arg) for arg in args.scores], [read_input(arg) for arg in args.online],
                        args.output_dir, args.seed, args.bootstrap_resamples, args.bootstrap_top, args.examples_per_direction)
    print(f"Wrote {len(rows)} stage/method/size comparisons to {args.output_dir}")


if __name__ == "__main__":
    main()
