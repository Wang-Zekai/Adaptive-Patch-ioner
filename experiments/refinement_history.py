"""Audit earlier Patch-ioner predictions and score a fixed offline fallback."""

import _paths  # Make the shared source modules importable.

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from vg_metrics import AREA_GROUPS, paired_rows, score_predictions, write_csv


STAGES = ("screen", "validation", "combinations", "test")
GROUPS = ("all", *AREA_GROUPS)
BEST = "J_dynamic_boundary"
FIXED = "A_w0p3"
SIZE_AWARE = "C_piecewise_mixed_support"
BOUNDARY = "G_detail_overlap_semantic"
METRICS = ("CIDEr", "BLEU", "ROUGE_L", "METEOR")


def load_stage(root, stage):
    file = root / stage / "scores.json"
    source = json.loads(file.read_bytes())
    modes, ids = paired_rows(source["rows"])
    recorded = {(r["mode"], r["area_group"]): r for r in source["summary"]}
    return source, modes, ids, recorded


def mean_ranks(x):
    order = np.argsort(x, kind="stable")
    result = np.empty(len(x), dtype=float)
    first = 0
    while first < len(order):
        last = first + 1
        while last < len(order) and x[order[first]] == x[order[last]]:
            last += 1
        result[order[first:last]] = (first + last - 1) / 2
        first = last
    return result


def relation(x, delta):
    if not len(x) or np.all(x == x[0]) or np.all(delta == delta[0]):
        return {"pearson": None, "spearman": None}
    return {"pearson": float(np.corrcoef(x, delta)[0, 1]),
            "spearman": float(np.corrcoef(mean_ranks(x), mean_ranks(delta))[0, 1])}


def delta_record(stage, mode, group, rows):
    deltas = np.asarray([r["delta_CIDEr"] for r in rows])
    changed = np.asarray([r["caption"] != r["baseline_caption"] for r in rows])
    return {"stage": stage, "mode": mode, "area_group": group, "regions": len(rows),
            "images": len({r['image_id'] for r in rows}), "mean_delta_CIDEr": float(deltas.mean()),
            "caption_change_rate": float(changed.mean()),
            "CIDEr_positive_rate": float((deltas > 1e-9).mean()),
            "CIDEr_negative_rate": float((deltas < -1e-9).mean()),
            "CIDEr_tie_rate": float((np.abs(deltas) <= 1e-9).mean())}


def stage_deltas(stage, modes, ids, summary):
    baseline = modes["baseline"]
    output = []
    for mode, regions in modes.items():
        rows = [{**regions[key], "baseline_caption": baseline[key]["caption"],
                 "delta_CIDEr": regions[key]["CIDEr"] - baseline[key]["CIDEr"]} for key in ids]
        for group in GROUPS:
            selected = [row for row in rows if group == "all" or row["area_group"] == group]
            result = delta_record(stage, mode, group, selected)
            expected = summary[mode, group]["CIDEr"] - summary["baseline", group]["CIDEr"]
            if not np.isclose(expected, result["mean_delta_CIDEr"], atol=1e-12, rtol=1e-9):
                raise ValueError(f"Per-region and corpus CIDEr differ in {stage}, {mode}, {group}")
            output.append({**result, "CIDEr": summary[mode, group]["CIDEr"]})
    return output


def signal_rows(root, stages):
    comparisons = []
    examples = []
    for stage in ("combinations", "test"):
        source, modes, keys, _ = stages[stage]
        metadata = source["metadata"]
        name = f"{metadata['manifest_sha256'][:12]}-{metadata['runner_version'][:12]}"
        signals = torch.load(root / 'features' / name / 'signals.pt', map_location='cpu', weights_only=True)
        candidates = []
        for key in keys:
            row, base = modes[BEST][key], modes["baseline"][key]
            geometry = signals[key]
            width, height = row["bbox_xywh"][2:]
            patch_support, short_support = geometry["patch_count"], geometry["short_side"]
            mixed_support = (min(patch_support / 150, 1) * min(short_support / 10, 1)) ** .5
            candidates.append({"stage": stage, "region_id": key, "image_id": row["image_id"],
                               "area_group": row["area_group"], "reference": row["references"],
                               "caption": row["caption"], "baseline_caption": base["caption"],
                               "CIDEr": row["CIDEr"], "baseline_CIDEr": base["CIDEr"],
                               "delta_CIDEr": row["CIDEr"] - base["CIDEr"], **geometry,
                               "aspect_ratio": max(width, height) / min(width, height),
                               "mixed_support": mixed_support})
        for group in GROUPS:
            selected = [r for r in candidates if group == 'all' or r['area_group'] == group]
            delta = np.array([row["delta_CIDEr"] for row in selected])
            for signal in ("area_fraction", "patch_count", "short_side", "aspect_ratio", "coherence", "mixed_support"):
                x = np.array([row[signal] for row in selected])
                comparisons.append({"stage": stage, "area_group": group, "signal": signal,
                                    "regions": len(x), **relation(x, delta)})
            changed = [row for row in selected if row['caption'] != row['baseline_caption']]
            for direction, rev in (("improvement", True), ("regression", False)):
                matching = [r for r in changed if r['delta_CIDEr'] > 0] if rev else [r for r in changed if r['delta_CIDEr'] < 0]
                ranked = sorted(matching, key=lambda r: r['delta_CIDEr'], reverse=rev)
                seen = set()
                for row in ranked:
                    if row['image_id'] in seen:
                        continue
                    examples.append({**row, "comparison_group": group, "direction": direction})
                    seen.add(row['image_id'])
                    if len(seen) == 4:
                        break
    return comparisons, examples


def factorial(stages):
    val = stages['validation'][3]
    combined = stages['combinations'][3]
    table = []
    for group in GROUPS:
        scores = {name: val[mode, group]['CIDEr'] for name, mode in
                  (("baseline", "baseline"), ("fixed", FIXED), ("size_only", SIZE_AWARE),
                   ("semantic_only", "G_detail_semantic"), ("boundary_semantic_only", BOUNDARY))}
        scores["size_and_boundary_semantic"] = combined[BEST, group]['CIDEr']
        table.append({"area_group": group, **scores,
                      "size_gain_vs_fixed": scores["size_only"] - scores["fixed"],
                      "boundary_semantic_gain_vs_fixed": scores["boundary_semantic_only"] - scores["fixed"],
                      "overlap_gain_given_semantic": scores["boundary_semantic_only"] - scores["semantic_only"],
                      "interaction": (scores["size_and_boundary_semantic"] - scores["size_only"]
                                      - scores["boundary_semantic_only"] + scores["fixed"])})
    return table


def fallback(stage, modes, ids):
    baseline = modes["baseline"]
    order = [baseline[k] for k in ids]
    active = {k: baseline[k]["area_group"] != "large" for k in ids}
    for key in ids:
        final = modes[BEST][key] if active[key] else baseline[key]
        order.append({**final, "mode": "simulated_large_fallback", "detail_active": active[key]})
    for key in ids:
        order.append(modes[BEST][key])
    output = score_predictions({"rows": order, "metadata": {"source_stage": stage,
                               "simulation_only": True, "online_latency_measured": False}}, METRICS, "ptb")
    output['metadata']['cider_idf_corpus'] = f"all paired regions in historical {stage}"
    for row in output["summary"]:
        row["stage"] = stage
        if row["mode"] == "simulated_large_fallback":
            row["detail_activation_rate"] = (sum(active.values()) / len(active) if row["area_group"] == "all"
                                             else float(row["area_group"] != "large"))
    return output


def fallback_intervals(simulations, seed=20261125, draws=10000):
    rng = np.random.default_rng(seed)
    records = []
    for stage, output in simulations.items():
        modes, ids = paired_rows(output['rows'])
        baseline = modes['baseline']
        for group in GROUPS:
            selected = [key for key in ids if group == 'all' or baseline[key]['area_group'] == group]
            image_ids = sorted({baseline[key]['image_id'] for key in selected})
            image_index = {image_id: index for index, image_id in enumerate(image_ids)}
            positions = [image_index[baseline[key]['image_id']] for key in selected]
            count = np.bincount(positions, minlength=len(image_ids))
            weights = rng.multinomial(len(image_ids), np.full(len(image_ids), 1 / len(image_ids)), size=draws)
            for control in ('baseline', BEST):
                delta = np.array([modes['simulated_large_fallback'][key]['CIDEr'] - modes[control][key]['CIDEr']
                                  for key in selected])
                totals = np.bincount(positions, weights=delta, minlength=len(image_ids))
                reps = (weights @ totals) / (weights @ count)
                lo, hi = np.quantile(reps, (0.025, 0.975))
                records.append({'stage': stage, 'area_group': group, 'control': control,
                                'images': len(image_ids), 'regions': len(selected), 'draws': draws,
                                'delta_CIDEr': float(delta.mean()), 'ci95_lower': float(lo), 'ci95_upper': float(hi)})
    return records


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--previous-root', type=Path, default=Path('results/exploration'))
    cli.add_argument('--output', type=Path, default=Path('results/refinement/history'))
    args = cli.parse_args()
    stages = {stage: load_stage(args.previous_root, stage) for stage in STAGES}
    total = sum(len(stage[0]['rows']) for stage in stages.values())
    all_deltas = [row for stage, (_, modes, ids, summary) in stages.items()
                  for row in stage_deltas(stage, modes, ids, summary)]
    signal_comparisons, examples = signal_rows(args.previous_root, stages)
    components = factorial(stages)
    simulations = {stage: fallback(stage, stages[stage][1], stages[stage][2]) for stage in ("combinations", "test")}
    args.output.mkdir(parents=True, exist_ok=True)
    results = {
        "prior_prediction_records": total, "stage_record_counts": {key: len(v[0]['rows']) for key, v in stages.items()},
        "stage_group_counts": {key: dict(Counter(v[1]['baseline'][k]['area_group'] for k in v[2])) for key, v in stages.items()},
        "locked_configuration": stages['test'][0]['metadata']['configs'][BEST],
        "effective_defaults": {"crop_scale": 1.2, "resolution": 518, "semantic_temperature": 0.25,
                               "original_pool": "official"},
        "notes": ["Archived tests are historical diagnostics only.", "Existing geometry caches are used on CPU.",
                  "Historical coherence is mean pairwise cosine; no baseline token probabilities were saved.",
                  "A fixed fallback chooses baseline for area fraction at least 0.10; its quality is rescored with Stanford PTB.",
                  "Fallback replay has no online latency measurement."],
    }
    (args.output / 'audit.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    for name, rows in (("all_prior_method_deltas", all_deltas), ("validation_factorial", components),
                       ("signal_correlations", signal_comparisons), ("selected_caption_examples", examples),
                       ("simulated_fallback_summary", [r for obj in simulations.values() for r in obj['summary']]),
                       ("simulated_fallback_paired_bootstrap", fallback_intervals(simulations))):
        path = args.output / f'{name}.csv'
        temp = args.output / f'{name}.tmp'
        temp.unlink(missing_ok=True)
        write_csv(temp, rows)
        temp.replace(path)
    for stage, output in simulations.items():
        (args.output / f'{stage}_fallback_scores.json').write_text(
            json.dumps(output, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    print(f"Audited {total:,} records and rescored two historical fallback simulations")


if __name__ == '__main__':
    main()
