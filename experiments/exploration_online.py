"""Measure end-to-end, single-region online inference on a fixed stratified sample."""

from _paths import CONFIGS

import argparse
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from explore import read_json, stratified
from exploration_decode import ExplorationDecoder
from exploration_features import VisualFeatures, image_rgb, tensor_signals
from exploration_methods import candidate_grid, compose, detail_keys, required_features
from multiscale import clip_box, load_captioner
from multiscale_experiment import read_manifest, write_json
from region_smoke import MIB, ROOT
from vg_metrics import AREA_GROUPS


class OnlineRegion:
    def __init__(self, visual, image, bbox):
        self.visual = visual
        self.image = image
        self.box = clip_box(bbox, image.size)
        self.original = None
        self.signals = None
        self.pools = {}
        self.crops = {}

    def feature(self, key):
        if key not in self.pools:
            kind, *parts = key.split(':')
            if kind == 'detail':
                scale, size, pool = parts
                layout = float(scale), int(size)
                if layout not in self.crops:
                    tensor, target = self.visual.prepare_detail(self.image, self.box, *layout)
                    patches, shape = self.visual.encode([tensor])
                    self.crops[layout] = (patches, shape, target)
                patches, shape, target = self.crops[layout]
                result = self.visual.pool(patches, target, shape, self.visual.patch_size, pool)
            else:
                patches, shape = self.full()
                if kind == 'original':
                    result = self.visual.pool_original(patches, shape, self.image, self.box, parts[0])
                else:
                    result = self.visual.pool_context(patches, shape, self.image, self.box, float(parts[0]), parts[1])
            self.pools[key] = result
        return self.pools[key]

    def full(self):
        if self.original is None:
            side = self.visual.captioner.model.resize_dim
            self.original = self.visual.encode([self.visual.transform(self.image, side)])
        return self.original

    def vector(self, config):
        if config['id'] == 'detail_only':
            key = detail_keys(config)[0]
            return self.feature(key)
        if self.signals is None:
            original, shape = self.full()
            self.signals = tensor_signals([self.visual.get_signals(original, shape, self.image, self.box)])
        # Model-derived gate signals are available before any detail feature call.
        keys = required_features(config, self.signals)
        features = {key: self.feature(key) for key in keys}
        for key in set(required_features(config)).difference(keys):
            if key.startswith('context:'):
                features[key] = features[f"original:{config.get('original_pool', 'official')}"].new_zeros((1, self.visual.dimension))
        return compose(config, features, self.signals)[0]


@torch.inference_mode()
def online_predict(captioner, visual, decoder, image, region, config, sources):
    name = config['id']
    if name == 'baseline':
        value = captioner.predict(image, region['bbox_xywh'], 'baseline')['caption']
        return value.strip(), 0
    online = OnlineRegion(visual, image, region['bbox_xywh'])
    if config['family'] == 'H' or config.get('rank_policy'):
        choice_set = config.get('source_configs', sources)
        vectors = torch.cat([online.vector(c) for c in choice_set])
        candidates = decoder.decode(vectors, cached=False)
        result = decoder.rank([candidates], vectors[:1], vectors[1:2], source_vectors=vectors[None], cached=False)
        caption = candidates[result[config.get('rank_policy', name)][0]]
    else:
        vector = online.vector(config)
        caption = decoder.decode(vector, cached=False)[0]
    return caption.strip(), len(online.crops)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'results' / 'exploration')
    parser.add_argument('--config', type=Path, default=CONFIGS / 'exploration.json')
    parser.add_argument('--stages', nargs='+', choices=('screen', 'validation', 'combinations', 'test'), required=True)
    parser.add_argument('--paired', action='store_true', help='Rotate the order of policies within each region')
    args = parser.parse_args()
    settings = read_json(args.config)
    captioner = load_captioner()
    visual = VisualFeatures(captioner, 1)
    decoder = ExplorationDecoder(captioner, projection_batch_size=settings['decode_batch_size'])
    grid = {config['id']: config for config in candidate_grid()}
    sources = [grid[name] for name in settings['h_sources']]
    for stage in args.stages:
        folder = args.root / stage
        results = read_json(folder / 'scores.json')
        manifest = ROOT / settings['test_manifest' if stage == 'test' else 'validation_manifest']
        data, images, sha = read_manifest(manifest, manifest.parent)
        if sha != results['metadata']['manifest_sha256']:
            raise ValueError(f"Online sample source disagrees with the {stage} experiment")
        available = data['regions']
        if stage == 'screen':
            screen_ids = {r['region_id'] for r in read_json(args.root / 'screen_manifest.json')['regions']}
            available = [r for r in available if r['region_id'] in screen_ids]
        regions = stratified(available, settings['online_sample_size'], settings['seed'] + 21)
        name_to_config = results['metadata']['configs']
        if stage == 'screen':
            name_to_config = {name: config for name, config in name_to_config.items()
                              if config['family'] in ('B', 'I') or name in ('baseline', 'A_w0p3')}
        cached = {(r['mode'], r['region_id']): r for r in results['rows']}
        report = folder / ('online_paired.json' if args.paired else 'online.json')
        if report.exists():
            continue
        partial = folder / ('online_parts_paired' if args.paired else 'online_parts')
        partial.mkdir(exist_ok=True)

        def measured(config, region, image):
            name = config['id']
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = perf_counter()
            caption, crops = online_predict(captioner, visual, decoder, image, region, config, sources)
            torch.cuda.synchronize()
            return {'mode': name, 'region_id': region['region_id'], 'area_group': region['area_group'],
                    'online_ms': (perf_counter() - start) * 1000,
                    'peak_allocated_mib': torch.cuda.max_memory_allocated() / MIB,
                    'crop_encoder_calls': crops, 'detail_active': bool(crops), 'caption': caption,
                    'same_as_cached': caption == cached[name, region['region_id']]['caption']}

        if args.paired:
            pending = [c for c in name_to_config.values() if not (partial / f"{c['id']}.json").exists()]
            first = image_rgb(images[regions[0]['region_id']])
            for config in pending:
                for _ in range(2):
                    measured(config, regions[0], first)
            partial_rows = {c['id']: [] for c in pending}
            for index, region in enumerate(regions):
                if not pending:
                    break
                offset = index % len(pending)
                image = image_rgb(images[region['region_id']])
                for config in pending[offset:] + pending[:offset]:
                    partial_rows[config['id']].append(measured(config, region, image))
                if (index + 1) % 4 == 0:
                    print(f"Paired online {stage}: {index + 1}/{len(regions)} regions", flush=True)
            for name, rows in partial_rows.items():
                write_json(partial / f'{name}.json', rows)

        for config in name_to_config.values():
            name = config['id']
            if (partial / f'{name}.json').exists():
                continue
            rows = []
            for index, region in enumerate([regions[0], regions[0], *regions]):
                image = image_rgb(images[region['region_id']])
                row = measured(config, region, image)
                if index >= 2:
                    rows.append(row)
            write_json(partial / f'{name}.json', rows)
            print(f"Online {stage}/{name}: {mean(r['online_ms'] for r in rows):.1f} ms", flush=True)
        all_rows = [r for name in name_to_config for r in read_json(partial / f'{name}.json')]
        summary = []
        for name in name_to_config:
            for group in ('all', *AREA_GROUPS):
                rows = [r for r in all_rows if r['mode'] == name and (group == 'all' or r['area_group'] == group)]
                summary.append({'mode': name, 'area_group': group, 'regions': len(rows),
                                'mean_inference_ms': mean(r['online_ms'] for r in rows),
                                'peak_allocated_mib': max(r['peak_allocated_mib'] for r in rows),
                                'detail_activation_rate': mean(r['detail_active'] for r in rows),
                                'mean_crop_encoder_calls': mean(r['crop_encoder_calls'] for r in rows),
                                'caption_agreement': mean(r['same_as_cached'] for r in rows)})
        write_json(report, {'metadata': {'stage': stage, 'latency_kind': 'online', 'interleaved': args.paired,
                                         'manifest_sha256': sha, 'sample_size': len(regions)},
                            'summary': summary, 'rows': all_rows})


if __name__ == '__main__':
    main()
