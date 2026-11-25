"""Measure screening overhead or audit frozen test timings on fresh regions."""

from _paths import CONFIGS, ROOT

import argparse
from pathlib import Path

from explore import read_json, stratified
from refine import Refinement
from refinement_online import measure_online


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-timing-audit', action='store_true',
                        help='Remeasure frozen test methods on 96 different regions')
    parser.add_argument('--root', type=Path, default=ROOT / 'results' / 'refinement')
    parser.add_argument('--config', type=Path, default=CONFIGS / 'refinement.json')
    args = parser.parse_args()
    settings = read_json(args.config)
    output = args.root
    runner = Refinement(settings, output)
    if args.test_timing_audit:
        score = read_json(output / 'test' / 'scores.json')
        initial = read_json(output / 'test' / 'online.json')
        previous = set(initial['metadata']['sample_region_ids'])
        records, paths, _ = runner.dataset('test')
        previous_images = {r['image_id'] for r in records if r['region_id'] in previous}
        sample = stratified([r for r in records if r['image_id'] not in previous_images], 96, settings['seed'] + 3)
        locked = read_json(output / 'locked_final_configs.json')['configs']
        runner.model()
        measure_online(runner, 'test_timing_audit', output / 'test_timing_audit', sample, paths, locked, score)
        return
    folder = output / 'screen'
    score = read_json(folder / 'scores.json')
    candidates = score['metadata']['configs']
    names = ('baseline', 'current_best', 'C_conf_mean_token_logprob', 'C_conf_token_entropy',
             'C_conf_min_token_logprob', 'C_conf_memory_max_cosine', 'C_joint_with_confidence',
             'E_res336', 'E_support_resolution')
    runner.model()
    regions, paths, _ = runner.dataset('screen')
    sample = stratified(regions, settings['online_sample_size'], settings['seed'] + 2)
    measure_online(runner, 'screen', folder, sample, paths, [candidates[name] for name in names], score)


if __name__ == '__main__':
    main()
