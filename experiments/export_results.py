"""Build publication CSVs from refinement scores, paired intervals, and online timing."""

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GROUPS = ('all', 'small', 'medium', 'large')
METRICS = ('CIDEr', 'BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 'ROUGE_L', 'METEOR')
CONTROLS = {
    'baseline': 'Patch-ioner baseline',
    'fixed_detail': 'Fixed detail fusion',
    'size_only': 'Size-aware fusion',
    'boundary_only': 'Boundary-semantic pooling',
    'current_best': 'Previous best (size-aware and boundary-semantic)',
}
FALLBACK = {'B_best_large_fallback': 'Large-region fallback'}
ABLATIONS = {
    **CONTROLS,
    'B_size_select': 'Size-aware fusion with Patch-count selector',
    'B_boundary_select': 'Boundary-semantic pooling with Patch-count selector',
    'B_size_boundary_select': 'Size-aware and boundary-semantic with Patch-count selector',
    'B_semantic_only': 'Size-aware fusion with semantic-only pooling',
    **FALLBACK,
}
TIMING_STAGE = 'test_timing_audit'


def read_json(path):
    return json.loads(path.read_bytes())


def index_rows(rows, *keys):
    index = {}
    for row in rows:
        key = tuple(row[field] for field in keys)
        if key in index:
            raise ValueError(f'Duplicate entry for {key}')
        index[key] = row
    return index


def read_table(path, *keys):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return index_rows(csv.DictReader(stream), *keys)


def score_fields(score, manifest_sha):
    return {
        'quality_manifest_sha256': manifest_sha,
        **{metric: score[metric] for metric in METRICS},
    }


def interval_fields(intervals, stage, group, method):
    interval = {} if method == 'baseline' else intervals[stage, group, method]
    return {
        'delta_CIDEr_vs_baseline': interval.get('delta_CIDEr', 0),
        'delta_CIDEr_ci95_lower': interval.get('ci95_lower'),
        'delta_CIDEr_ci95_upper': interval.get('ci95_upper'),
        'bootstrap_images': interval.get('images'),
        'bootstrap_draws': interval.get('resamples'),
    }


def test_rows(test, audit, roles, intervals, timing):
    test_meta, audit_meta = test['metadata'], audit['metadata']
    sha = test_meta['manifest_sha256']
    if audit_meta['manifest_sha256'] != sha or audit_meta['latency_kind'] != 'online':
        raise ValueError('The timing audit must measure the same independent test split online')

    tests = index_rows(test['summary'], 'mode', 'area_group')
    times = index_rows(audit['summary'], 'mode', 'area_group')
    baseline_total = tests['baseline', 'all']['CIDEr']
    images = {row['image_id'] for row in test['rows'] if row['mode'] == 'baseline'}
    role_names = {roles[role]: role for role in ('Quality', 'Balanced', 'Efficient')}
    methods = {**CONTROLS, **role_names, **FALLBACK}
    if set(methods) != set(test_meta['configs']):
        raise ValueError('The test scores do not match the locked configurations')

    for method, name in methods.items():
        score = tests[method, 'all']
        audit_row = times[method, 'all']
        timing_row = timing[TIMING_STAGE, method]
        sample_size = audit_row['regions']
        if sample_size != int(timing_row['regions']) or sample_size != audit_meta['sample_size']:
            raise ValueError(f'Timing sample sizes disagree for {method}')

        yield {
            'method_id': method,
            'method': name,
            'quality_split': 'independent_test',
            'quality_regions': score['regions'],
            'quality_images': len(images),
            **score_fields(score, sha),
            **interval_fields(intervals, 'test', 'all', method),
            'relative_CIDEr_gain_pct': 100 * (score['CIDEr'] / baseline_total - 1),
            **{f'CIDEr_{group}': tests[method, group]['CIDEr'] for group in GROUPS[1:]},
            **{f'delta_CIDEr_{group}_vs_baseline': tests[method, group]['CIDEr'] - tests['baseline', group]['CIDEr']
               for group in GROUPS[1:]},
            'timing_source': f'independent_test_{sample_size}_region_online_recheck',
            'timing_regions': sample_size,
            'timing_images': timing_row['images'],
            'mean_online_ms_per_region': timing_row['online_ms'],
            'online_overhead_vs_baseline_pct': timing_row['relative_overhead_pct'],
            'online_overhead_ci95_lower_pct': timing_row['ci95_lower_pct'],
            'online_overhead_ci95_upper_pct': timing_row['ci95_upper_pct'],
            'peak_cuda_allocated_mib': audit_row['peak_allocated_mib'],
            'online_caption_agreement_with_quality_predictions': audit_row['caption_agreement'],
        }


def selection_rule(config):
    gate = config.get('gate', {})
    if gate.get('fixed_signal') == 'area_fraction':
        return 'large_region_area_limit'
    components = gate.get('components', [])
    return '+'.join(f'{signal}_score' for signal, _ in components)


def ablation_rows(validation, intervals):
    metadata = validation['metadata']
    scores = index_rows(validation['summary'], 'mode', 'area_group')
    prediction_sources = {}
    for row in validation['rows']:
        prediction_sources.setdefault(row['mode'], set()).add(row['prediction_source'])

    for method, name in ABLATIONS.items():
        config = metadata['configs'][method]
        pool = config.get('detail_pool', {}).get('method')
        sources = prediction_sources[method]
        if sources == {'previous_validation'}:
            basis = 'reused_identical_validation_predictions'
        elif 'previous_validation' in sources:
            basis = 'partially_reused_validation_predictions'
        else:
            basis = 'refinement_validation_inference'
        for group in GROUPS:
            score = scores[method, group]
            yield {
                'method_id': method,
                'method': name,
                'quality_split': 'validation',
                'area_group': group,
                'quality_regions': score['regions'],
                **score_fields(score, metadata['manifest_sha256']),
                **interval_fields(intervals, 'validation', group, method),
                'previous_experiment_config_id': config.get('history_id'),
                'prediction_basis': basis,
                'weighting_rule': config.get('weight_rule', 'fixed'),
                'detail_pooling_override': pool,
                'selection_rule': selection_rule(config),
                'config_json': json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(',', ':')),
            }


def write_table(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'results' / 'refinement', help='Refinement results directory')
    parser.add_argument('--output-dir', type=Path, help='Output directory (default: <results-root>/publication)')
    args = parser.parse_args()
    source = args.root
    output = args.output_dir or source / 'publication'
    test_path, ablation_path = output / 'main_results.csv', output / 'ablation.csv'
    if test_path.exists() or ablation_path.exists():
        parser.error('Output CSVs already exist. Choose a different --output-dir.')

    try:
        test = read_json(source / 'test' / 'scores.json')
        validation = read_json(source / 'validation' / 'scores.json')
        audit = read_json(source / TIMING_STAGE / 'online.json')
        roles = read_json(source / 'locked_final_configs.json')['roles']
        intervals = read_table(source / 'report' / 'paired_bootstrap.csv', 'stage', 'area_group', 'mode')
        timing = read_table(source / 'report' / 'paired_online_timing.csv', 'stage', 'mode')
        test_table = list(test_rows(test, audit, roles, intervals, timing))
        ablation_table = list(ablation_rows(validation, intervals))
        output.mkdir(parents=True, exist_ok=True)
        write_table(test_path, test_table)
        write_table(ablation_path, ablation_table)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(2, f'Could not export results: {error}\n')
    print(f'Wrote {len(test_table)} test summaries and {len(ablation_table)} ablation summaries to {output}.')


if __name__ == '__main__':
    main()
