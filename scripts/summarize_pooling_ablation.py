"""Summarize paired CIDEr differences using evaluate.py's per-region scores."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np


MODES = ('uniform_detail', 'size_only', 'current_best')
GROUPS = ('small', 'medium', 'large')
DRAWS = 10_000


def read_scores(path):
    rows = json.loads(path.read_text(encoding='utf-8'))['rows']
    by_mode = {mode: {} for mode in MODES}
    for row in rows:
        if row['mode'] not in by_mode:
            continue
        predictions = by_mode[row['mode']]
        region = str(row['region_id'])
        if region in predictions:
            raise ValueError(f'Duplicate region {region} in {row["mode"]}')
        group = row['area_group']
        if group not in GROUPS or row['CIDEr'] is None:
            raise ValueError(f'Expected a size group and CIDEr score for region {region}')
        score = float(row['CIDEr'])
        if not np.isfinite(score):
            raise ValueError(f'Non-finite CIDEr for region {region}')
        predictions[region] = (str(row['image_id']), group, score)

    uniform = by_mode[MODES[0]]
    if not uniform or any(by_mode[mode].keys() != uniform.keys() for mode in MODES[1:]):
        raise ValueError('All three modes need predictions for the same region IDs')

    paired = defaultdict(list)
    for region, (image, group, score_a) in uniform.items():
        image_b, group_b, score_b = by_mode[MODES[1]][region]
        image_c, group_c, score_c = by_mode[MODES[2]][region]
        if (image, group) != (image_b, group_b) or (image, group) != (image_c, group_c):
            raise ValueError(f'Image or size group differs for region {region}')
        paired[image, group].append((score_a, score_b, score_c))
    return paired


def summarize(paired, group, rng):
    by_image = defaultdict(list)
    for (image, size), regions in paired.items():
        if group == 'all' or size == group:
            by_image[image].extend(regions)
    if not by_image:
        return [group, 0, 0, *([''] * 9)]

    counts = np.array([len(regions) for regions in by_image.values()])
    totals = np.array([np.sum(regions, axis=0) for regions in by_image.values()])
    averages = totals.sum(axis=0) / counts.sum()
    differences = np.diff(averages)
    intervals = [''] * 4
    if len(counts) > 1:
        # CIDEr document frequencies remain fixed in the saved per-region scores.
        difference_totals = np.diff(totals, axis=1)
        batches = []
        for start in range(0, DRAWS, 256):
            picks = rng.integers(len(counts), size=(min(256, DRAWS - start), len(counts)))
            batches.append(difference_totals[picks].sum(axis=1) / counts[picks].sum(axis=1)[:, None])
        intervals = np.percentile(np.concatenate(batches), [2.5, 97.5], axis=0).T.ravel()
    return [group, int(counts.sum()), len(counts), *averages, *differences, *intervals]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scores', type=Path, help='scores.json from the pooling ablation run')
    args = parser.parse_args()
    try:
        paired = read_scores(args.scores)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(2, f'Error: {error}\n')

    rng = np.random.default_rng(42)
    writer = csv.writer(sys.stdout, lineterminator='\n')
    writer.writerow(['area_group', 'regions', 'images', 'CIDEr_A', 'CIDEr_B', 'CIDEr_C',
                     'B_minus_A', 'C_minus_B', 'B_minus_A_low', 'B_minus_A_high',
                     'C_minus_B_low', 'C_minus_B_high'])
    for group in ('all', *GROUPS):
        row = summarize(paired, group, rng)
        writer.writerow([*row[:3], *(f'{value:.8f}' if value != '' else '' for value in row[3:])])


if __name__ == '__main__':
    main()
