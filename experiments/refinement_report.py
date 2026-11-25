"""Export refinement rankings, paired intervals, selective budgets, and online Pareto fronts."""

import _paths  # Make the shared source modules importable.

import argparse
from html import escape
from pathlib import Path

import numpy as np

from explore import read_json
from exploration_report import build_report, paired_bootstrap, rank_stage, read_input, write_csv
from region_smoke import ROOT


def pareto(stage, scores, online):
    by_mode = {row['mode']: row for row in scores['summary'] if row['area_group'] == 'all'}
    timed = [row for row in online['summary'] if row['area_group'] == 'all']
    base = next(row for row in timed if row['mode'] == 'baseline')
    rows = [{'stage': stage, 'mode': row['mode'], 'CIDEr': by_mode[row['mode']]['CIDEr'],
             'online_ms': row['mean_inference_ms'], 'peak_gpu_MiB': row['peak_allocated_mib'],
             'detail_rate_full': by_mode[row['mode']]['detail_activation_rate'], 'detail_rate_timed': row['detail_activation_rate'],
             'relative_time': row['mean_inference_ms'] / base['mean_inference_ms'],
             'confidence_ms': row.get('mean_confidence_ms')}
            for row in timed]
    for row in rows:
        row['pareto_optimal'] = not any(other['CIDEr'] >= row['CIDEr'] and other['online_ms'] <= row['online_ms'] and
                                         (other['CIDEr'] > row['CIDEr'] or other['online_ms'] < row['online_ms']) for other in rows)
    return sorted(rows, key=lambda row: (row['online_ms'], -row['CIDEr']))


def paired_timing(stage, scores, online, seed, draws):
    images = {r['region_id']: r['image_id'] for r in scores['rows'] if r['mode'] == 'baseline'}
    ids = online['metadata']['sample_region_ids']
    clusters = sorted({images[rid] for rid in ids})
    positions = {image: i for i, image in enumerate(clusters)}
    index = [positions[images[rid]] for rid in ids]
    groups = {}
    for row in online['rows']:
        groups.setdefault(row['mode'], {})[row['region_id']] = row['online_ms']
    totals = {name: np.bincount(index, weights=[items[rid] for rid in ids], minlength=len(clusters))
              for name, items in groups.items()}
    rng = np.random.default_rng(seed)
    draw = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=draws)
    base = draw @ totals['baseline']
    rows = []
    for name, elapsed in totals.items():
        lower, upper = np.percentile(((draw @ elapsed) / base - 1) * 100, [2.5, 97.5])
        rows.append({'stage': stage, 'mode': name, 'images': len(clusters), 'regions': len(ids),
                     'online_ms': float(elapsed.sum() / len(ids)),
                     'relative_overhead_pct': float((elapsed.sum() / totals['baseline'].sum() - 1) * 100),
                     'ci95_lower_pct': float(lower), 'ci95_upper_pct': float(upper)})
    return rows


def pareto_svg(stages, roles):
    width, height = 1040, 385 * len(stages)
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#f9fafb"/>',
            '<g font-family="Arial, sans-serif" font-size="13" fill="#1f2937">']
    for index, (stage, points) in enumerate(stages.items()):
        top = index * 385
        x0, y0, plot_w, plot_h = 80, top + 65, 720, 245
        xmin, xmax = 0.88 * min(p['online_ms'] for p in points), 1.12 * max(p['online_ms'] for p in points)
        ymin = min(p['CIDEr'] for p in points) - 0.003
        ymax = max(p['CIDEr'] for p in points) + 0.003
        def pos(x, y):
            return x0 + plot_w * (x - xmin) / (xmax - xmin), y0 + plot_h * (ymax - y) / (ymax - ymin)
        body.append(f'<text x="80" y="{top+32}" font-size="20" font-weight="bold">{escape(stage.title())} · CIDEr vs actual online latency</text>')
        for i in range(6):
            xvalue, yvalue = xmin + i * (xmax - xmin) / 5, ymin + i * (ymax - ymin) / 5
            x, y = pos(xvalue, yvalue)
            body.extend([f'<line x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y0+plot_h}" stroke="#e1e6ed"/>',
                         f'<line x1="{x0}" y1="{y:.1f}" x2="{x0+plot_w}" y2="{y:.1f}" stroke="#e1e6ed"/>',
                         f'<text x="{x:.1f}" y="{y0+plot_h+22}" text-anchor="middle">{xvalue:.0f}</text>',
                         f'<text x="{x0-10}" y="{y+4:.1f}" text-anchor="end">{yvalue:.3f}</text>'])
        front = [pos(p['online_ms'], p['CIDEr']) for p in points if p['pareto_optimal']]
        if len(front) > 1:
            coordinates = ' '.join(f'{x:.2f},{y:.2f}' for x, y in front)
            body.append(f'<polyline points="{coordinates}" fill="none" stroke="#157f66" stroke-width="2.5"/>')
        short = {identifier: role for role, identifier in roles.items()}
        for point in points:
            name = point['mode']
            x, y = pos(point['online_ms'], point['CIDEr'])
            color = '#157f66' if point['pareto_optimal'] else '#91a1b3'
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{color}"><title>{escape(name)}: {point["CIDEr"]:.4f} / {point["online_ms"]:.1f} ms</title></circle>')
            if name in short or name in ('baseline', 'current_best') or (stage == 'validation' and name.startswith('C_budget_')):
                label = short.get(name, name.replace('C_budget_', 'Budget '))
                body.append(f'<text x="{x+8:.1f}" y="{y-6:.1f}" font-size="11">{escape(label)}</text>')
        body.extend([f'<text x="430" y="{y0+plot_h+56}" text-anchor="middle">Mean milliseconds per region</text>',
                     f'<circle cx="842" cy="{y0+10}" r="5" fill="#157f66"/><text x="852" y="{y0+14}">Pareto-optimal</text>',
                     f'<circle cx="842" cy="{y0+37}" r="5" fill="#91a1b3"/><text x="852" y="{y0+41}">Dominated</text>'])
    return '\n'.join([*body, '</g>', '</svg>'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'results' / 'refinement')
    parser.add_argument('--stages', nargs='+', default=['screen', 'validation', 'test'])
    args = parser.parse_args()
    lock_path = args.root / 'locked_final_configs.json'
    lock = read_json(lock_path) if lock_path.is_file() else {'roles': {}, 'bootstrap_seed': 20261126, 'bootstrap_resamples': 10000}
    out = args.root / 'report'
    stages = {stage: args.root / stage / 'scores.json' for stage in args.stages}
    online = {stage: path.parent / 'online.json' for stage, path in stages.items() if (path.parent / 'online.json').is_file()}
    build_report([read_input(f'{stage}={path}') for stage, path in stages.items()],
                 [read_input(f'{stage}={path}') for stage, path in online.items()], out,
                 lock['bootstrap_seed'], lock['bootstrap_resamples'], 0, 2)
    comparisons = {stage: pareto(stage, read_json(stages[stage]), read_json(path)) for stage, path in online.items()}
    if 'test' in stages:
        test_scores = read_json(stages['test'])
        ranked, predictions = rank_stage('test', test_scores, {})
        # The shared bootstrap uses the name baseline for its paired control.
        aliases = {'current_best': 'baseline', 'baseline': 'official_baseline'}
        comparator = {aliases.get(name, name): rows for name, rows in predictions.items()}
        ranks = [{**row, 'mode': aliases.get(row['mode'], row['mode'])} for row in ranked]
        intervals = paired_bootstrap('test', ranks, comparator, lock['bootstrap_seed'], lock['bootstrap_resamples'], 0)
        write_csv(out / 'paired_vs_previous_best.csv', [
            {**row, 'mode': 'baseline' if row['mode'] == 'official_baseline' else row['mode'],
             'control': 'current_best'} for row in intervals
        ])
        timed = {'test': online['test']} if 'test' in online else {}
        audit = args.root / 'test_timing_audit' / 'online.json'
        if audit.is_file():
            timed['test_timing_audit'] = audit
        timing_rows = []
        for stage, path in timed.items():
            actual = read_json(path)
            timing_rows.extend(paired_timing(stage, test_scores, actual, lock['bootstrap_seed'], lock['bootstrap_resamples']))
            if stage not in comparisons:
                comparisons[stage] = pareto(stage, test_scores, actual)
        if timing_rows:
            write_csv(out / 'paired_online_timing.csv', timing_rows)
    if comparisons:
        rows = [point for points in comparisons.values() for point in points]
        write_csv(out / 'pareto_online.csv', rows)
        (out / 'pareto_online.svg').write_text(pareto_svg(comparisons, lock['roles']), encoding='utf-8')
        budgets = [point for point in rows if point['mode'].startswith('C_budget_')]
        if budgets:
            write_csv(out / 'selective_budgets.csv', budgets)
    print(out)


if __name__ == '__main__':
    main()
