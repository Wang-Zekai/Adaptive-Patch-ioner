"""Resumable staged evaluation of frozen Patch-ioner policies."""

from _paths import CONFIGS, EXPERIMENTS

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

import torch

from exploration_decode import ExplorationDecoder
from exploration_features import FeatureCache, VisualFeatures, image_rgb
from exploration_methods import (SIZE_BINS, candidate_grid, compose, detail_keys, gate_mask,
                                 piecewise_candidate, required_features, size_bins)
from multiscale import load_captioner
from multiscale_experiment import read_manifest, write_json
from region_smoke import ROOT, MIB, MODEL_REVISION
from vg_metrics import AREA_GROUPS, make_scorers, text_scores, tokenizer, write_csv


def read_json(path):
    return json.loads(path.read_bytes())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stratified(records, count, seed):
    rng = random.Random(seed)
    result = []
    for index, group in enumerate(AREA_GROUPS):
        members = [r for r in records if r['area_group'] == group]
        quota = count // 3 + (index < count % 3)
        if len(members) < quota:
            raise ValueError(f"Too few {group} regions for the fixed sample")
        result.extend(rng.sample(members, quota))
    chosen = {r['region_id'] for r in result}
    return [r for r in records if r['region_id'] in chosen]


class StageScores:
    def __init__(self, folder, records, metadata, metrics):
        self.folder = folder
        self.folder.mkdir(parents=True, exist_ok=True)
        self.parts = self.folder / 'parts'
        self.parts.mkdir(exist_ok=True)
        self.records = records
        self.metadata = metadata
        self.metrics = metrics
        self.scorers = None
        self.refs = None
        self.groups = {name: [r['region_id'] for r in records if r['area_group'] == name] for name in AREA_GROUPS}

    def path(self, name):
        return self.parts / f'{name}.json'

    def existing(self, config):
        path = self.path(config['id'])
        if not path.exists():
            return None
        record = read_json(path)
        expected = digest([config, self.metadata, self.metrics])
        if record['run_signature'] != expected:
            raise ValueError(f"Saved experiment has different settings: {path}")
        return record

    def score(self, config, predictions, baseline):
        if self.scorers is None:
            self.tokenize = tokenizer('ptb')
            self.refs = self.tokenize({r['region_id']: [{'caption': s} for s in r['references']] for r in self.records})
            self.scorers = make_scorers(self.metrics)
        name = config['id']
        captions = {r['region_id']: [{'caption': predictions[r['region_id']]['caption']}] for r in self.records}
        corpus, instance = text_scores(self.scorers, self.refs, self.tokenize(captions), self.groups)
        rows = []
        for region in self.records:
            region_id = region['region_id']
            rows.append({**region, **predictions[region_id], **instance[region_id], 'mode': name,
                         'baseline_caption': baseline[region_id]['caption']})
        summary = []
        for group, metrics in corpus.items():
            selected = [r for r in rows if group == 'all' or r['area_group'] == group]
            times = [r['cache_decode_ms'] for r in selected if r.get('cache_decode_ms') is not None]
            allocations = [r['cache_peak_allocated_mib'] for r in selected if r.get('cache_peak_allocated_mib') is not None]
            summary.append({'mode': name, 'area_group': group, 'regions': len(selected), **metrics,
                            'mean_inference_ms': mean(times) if times else None,
                            'peak_allocated_mib': max(allocations, default=None),
                            'detail_activation_rate': mean(r['detail_active'] for r in selected)})
        record = {'run_signature': digest([config, self.metadata, self.metrics]), 'config': config,
                  'summary': summary, 'rows': rows}
        write_json(self.path(name), record)
        print(f"[{self.metadata['stage']}] {name}: CIDEr {corpus['all']['CIDEr']:.5f}", flush=True)
        return record

    def export(self):
        entries = [read_json(path) for path in sorted(self.parts.glob('*.json'))]
        if not entries:
            return
        summary = [row for entry in entries for row in entry['summary']]
        rows = [row for entry in entries for row in entry['rows']]
        metadata = {**self.metadata, 'latency_kind': 'cached', 'metrics': self.metrics,
                    'configs': {entry['config']['id']: entry['config'] for entry in entries}}
        write_json(self.folder / 'scores.json', {'metadata': metadata, 'summary': summary, 'rows': rows})
        for name, values in [('summary.csv', summary), ('per_region.csv', rows)]:
            path = self.folder / name
            temp = path.with_suffix('.csv.tmp')
            if temp.is_file():
                temp.unlink()
            write_csv(temp, values)
            temp.replace(path)


class Exploration:
    def __init__(self, settings, args):
        self.settings = settings
        self.args = args
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=True)
        self.grid = {c['id']: c for c in candidate_grid()}
        self.prediction_root = self.output / 'predictions'
        self.captioner = None
        self.visual = None
        self.decoder = None
        files = ('explore.py', 'exploration_features.py', 'exploration_methods.py', 'exploration_decode.py')
        self.version = digest([MODEL_REVISION, args.kv_cache, *(file_digest(EXPERIMENTS / file) for file in files)])
        self.prior = None

    def load_model(self):
        if self.captioner is None:
            self.captioner = load_captioner()
            self.visual = VisualFeatures(self.captioner, self.settings['image_batch_size'])
            self.decoder = ExplorationDecoder(self.captioner, kv_cache=self.args.kv_cache,
                                              projection_batch_size=self.settings['decode_batch_size'])

    def dataset(self, stage):
        path = ROOT / self.settings['test_manifest' if stage == 'test' else 'validation_manifest']
        doc, images, sha = read_manifest(path, path.parent)
        rows = doc['regions']
        if stage == 'test':
            validation = read_json(ROOT / self.settings['validation_manifest'])
            if doc.get('split') != 'test' or {r['image_id'] for r in rows} & {r['image_id'] for r in validation['regions']}:
                raise ValueError('The test manifest must be disjoint from validation')
        if stage == 'screen':
            rows = stratified(rows, self.settings['screen_size'], self.settings['seed'])
            write_json(self.output / 'screen_manifest.json', {**doc, 'regions': rows, 'source_manifest_sha256': sha})
        return rows, images, sha

    def cache(self, rows, images, sha):
        self.load_model()
        return FeatureCache(self.output / 'features' / f'{sha[:12]}-{self.version[:12]}', self.visual,
                            {r['region_id']: r for r in rows}, images)

    def prediction_file(self, config, sha):
        folder = self.prediction_root / f'{sha[:12]}-{self.version[:12]}'
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{config['id']}-{digest(config)[:12]}.json"

    def existing_predictions(self, config, sha):
        path = self.prediction_file(config, sha)
        return read_json(path) if path.exists() else {}

    def reuse_prior(self, config, sha, ids, active):
        original = ROOT / 'results' / 'vg_val_1000' / 'results.json'
        aliases = dict(zip(self.settings['prior_controls'], ('baseline', 'detail_only', 'detail_fusion', 'context_fusion', 'multiscale_fusion')))
        if config['id'] not in aliases or not original.is_file():
            return {}
        if self.prior is None:
            self.prior = read_json(original)
        if self.prior['metadata']['manifest_sha256'] != sha:
            return {}
        rows = {r['region_id']: r for r in self.prior['rows'] if r['mode'] == aliases[config['id']]}
        return {rid: {'caption': rows[rid]['caption'], 'detail_active': bool(flag),
                      'cache_decode_ms': None, 'cache_peak_allocated_mib': None,
                      'prediction_source': 'previous_online_run'} for rid, flag in zip(ids, active) if rid in rows}

    def predict(self, cache, config, ids, sha):
        features, signals = cache.load(required_features(config), ids)
        fused, weights = compose(config, features, signals)
        if not torch.isfinite(fused).all() or (torch.linalg.vector_norm(fused, dim=-1) <= 0).any():
            raise ValueError(f"Invalid feature for {config['id']}")
        active = gate_mask(config, signals).tolist()
        path = self.prediction_file(config, sha)
        predictions = self.existing_predictions(config, sha)
        if not predictions:
            predictions = self.reuse_prior(config, sha, ids, active)
        missing = [idx for idx, rid in enumerate(ids) if rid not in predictions]
        batch = self.settings['decode_batch_size']
        for start in range(0, len(missing), batch):
            indices = missing[start:start + batch]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tick = perf_counter()
            outputs = self.decoder.decode(fused[indices])
            torch.cuda.synchronize()
            each_ms = (perf_counter() - tick) * 1000 / len(indices)
            peak = torch.cuda.max_memory_allocated() / MIB
            for index, caption in zip(indices, outputs, strict=True):
                predictions[ids[index]] = {'caption': caption.strip(), 'detail_active': bool(active[index]),
                                           'detail_weight': weights[index].item(), 'cache_decode_ms': each_ms,
                                           'cache_peak_allocated_mib': peak, 'prediction_source': 'cached_features'}
            if start and start % (batch * 8) == 0:
                write_json(path, predictions)
        write_json(path, predictions)
        return {rid: predictions[rid] for rid in ids}, fused

    def predict_h(self, cache, configs, ids, sha):
        sources = configs[0].get('source_configs') or [self.grid[name] for name in self.settings['h_sources']]
        input_results = [self.predict(cache, config, ids, sha) for config in sources]
        vectors = torch.stack([features for _, features in input_results], dim=1)
        candidates = [[pred[rid]['caption'] for pred, _ in input_results] for rid in ids]
        output = {c['id']: self.existing_predictions(c, sha) for c in configs}
        missing = [index for index, rid in enumerate(ids) if any(rid not in saved for saved in output.values())]
        if not missing:
            return output
        for start in range(0, len(missing), 12):
            indices = missing[start:start + 12]
            source = vectors[indices]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tick = perf_counter()
            batch = self.decoder.rank([candidates[i] for i in indices], source[:, 0], source[:, 1], source_vectors=source)
            torch.cuda.synchronize()
            each_ms = (perf_counter() - tick) * 1000 / len(indices)
            peak = torch.cuda.max_memory_allocated() / MIB
            for config in configs:
                name = config['id']
                saved = output[name]
                for i, choice in zip(indices, batch[config.get('rank_policy', name)], strict=True):
                    saved[ids[i]] = {'caption': candidates[i][choice], 'detail_active': True,
                                     'candidate_source': sources[choice]['id'], 'cache_decode_ms': each_ms,
                                     'cache_peak_allocated_mib': peak, 'prediction_source': 'rank_only'}
            if start and start % 96 == 0:
                for config in configs:
                    write_json(self.prediction_file(config, sha), output[config['id']])
        for config in configs:
            write_json(self.prediction_file(config, sha), output[config['id']])
        return output

    def run_candidates(self, scorer, cache, configs, ids, sha, baseline=None):
        pending = [c for c in configs if scorer.existing(c) is None]
        if not pending:
            return baseline
        rank_configs = [c for c in pending if c['family'] == 'H' or c.get('rank_policy')]
        feature_configs = [c for c in pending if c not in rank_configs]
        rank_groups = defaultdict(list)
        for config in rank_configs:
            rank_groups[digest(config.get('source_configs'))].append(config)
        rank_sources = [source for configs in rank_groups.values() for source in
                        (configs[0].get('source_configs') or [self.grid[name] for name in self.settings['h_sources']])]
        keys = {key for c in (*feature_configs, *rank_sources) for key in required_features(c)}
        if any(cache.needed(key, ids) for key in keys):
            self.decoder.clear_projection_cache()
        cache.fill(keys, ids)
        for config in feature_configs:
            prediction, _ = self.predict(cache, config, ids, sha)
            if config['id'] == 'baseline':
                baseline = prediction
            scorer.score(config, prediction, baseline)
        for group in rank_groups.values():
            ranked = self.predict_h(cache, group, ids, sha)
            for config in group:
                scorer.score(config, ranked[config['id']], baseline)
        scorer.export()
        return baseline

    def derived_screen(self, scorer, cache, ids):
        configs = []
        controls = [c for c in self.grid.values() if c['family'] == 'A']
        rows = {c['id']: scorer.existing(c) for c in controls}
        signals = cache.load((), ids)[1]
        for signal in SIZE_BINS:
            bins = size_bins(signals, signal).tolist()
            winners = []
            for group in (0, 1, 2):
                scores = {name: [r['CIDEr'] for k, r in enumerate(data['rows']) if bins[k] == group]
                          for name, data in rows.items()}
                eligible = [c for c in controls if scores[c['id']]]
                winners.append(max(eligible, key=lambda c: mean(scores[c['id']]))['weight'] if eligible else 0.3)
            configs.append(piecewise_candidate(signal, winners))
        crop_candidates = [c for c in self.grid.values() if c['id'].startswith('B_crop')]
        winner = max(crop_candidates, key=lambda c: scorer.existing(c)['summary'][0]['CIDEr'])
        for weight in (0.1, 0.5, 0.7):
            configs.append({**winner, 'id': f"{winner['id']}_w{str(weight).replace('.', 'p')}", 'weight': weight})
        best_size = max((c for c in self.grid.values() if c['family'] == 'C'),
                        key=lambda c: scorer.existing(c)['summary'][0]['CIDEr'])
        configs.append({**best_size, 'id': 'D_best_size_semantic', 'family': 'D',
                        'size_rule': best_size['weight_rule'], 'weight_rule': 'size_semantic'})
        return configs

    def selections(self, stage):
        former = 'screen' if stage == 'validation' else 'validation'
        previous = read_json(self.output / former / 'scores.json')
        configs = previous['metadata']['configs']
        rows = sorted((r for r in previous['summary'] if r['area_group'] == 'all'), key=lambda r: -r['CIDEr'])
        families = defaultdict(list)
        for row in rows:
            config = configs[row['mode']]
            families[config['family']].append(config)
        if stage == 'validation':
            picked = [configs[name] for name in self.settings['prior_controls']]
            present = {c['id'] for c in picked}
            for family in 'ABCDEFGHI':
                picked.extend([c for c in families[family] if c['id'] not in present][:self.settings['survivors_per_family']])
            return picked
        best_fusion = next(configs[row['mode']] for row in rows if configs[row['mode']]['family'] in 'BCDEG')
        return self.combine(families, configs, best_fusion)

    @staticmethod
    def combine(families, configs, best_fusion):
        weight = next((c for c in families['A'] if c['id'].startswith('A_')), configs['A_w0p3'])
        crop = families['B'][0]
        dynamic = families['C'][0]
        boundary = next((c for c in families['G'] if 'original_pool' in c or 'detail_pool' in c), None)
        semantic = families['D'][0]
        multiscale = next((c for c in families['E'] if not c.get('scale_selector')), families['E'][0])
        gate = families['I'][0]
        pool = {key: boundary[key] for key in ('original_pool', 'detail_pool') if boundary and key in boundary}
        scale = {key: crop[key] for key in ('detail_scale', 'detail_resolution') if key in crop}
        rule = {k: v for k, v in semantic.items() if k not in ('id', 'family')}
        def joint(name, *parts):
            return {**{k: v for part in parts for k, v in part.items()}, 'id': name, 'family': 'J'}
        result = [configs['baseline'], configs['A_w0p3'], joint('J_crop_fixed', weight, scale),
                joint('J_crop_dynamic', dynamic, scale), joint('J_dynamic_boundary', dynamic, pool),
                joint('J_multiscale_semantic', multiscale, rule),
                joint('J_crop_selective', weight, scale, {k: gate[k] for k in ('gate', 'gate_threshold')})]
        if families['H']:
            result.append({'id': 'J_fusion_rerank', 'family': 'J', 'rank_policy': families['H'][0]['id'],
                           'source_configs': [configs['baseline'], configs['detail_only'], configs['A_w0p3'], best_fusion]})
        unique = {}
        for config in result:
            signature = digest({k: v for k, v in config.items() if k not in ('id', 'family', 'categories')})
            unique.setdefault(signature, config)
        return list(unique.values())

    def freeze(self):
        frozen = self.output / 'locked_test_configs.json'
        if frozen.exists():
            return read_json(frozen)['configs']
        tables = [read_json(self.output / stage / 'scores.json') for stage in ('validation', 'combinations')]
        scores = [(summary['CIDEr'], table['metadata']['configs'][summary['mode']]) for table in tables
                  for summary in table['summary'] if summary['area_group'] == 'all']
        best = max((item for item in scores if item[1]['id'] != 'baseline'), key=lambda x: x[0])[1]
        picked = [self.grid['baseline'], self.grid['A_w0p3'], best]
        unique = list({c['id']: c for c in picked}.values())
        write_json(frozen, {'configs': unique, 'validation_sources': ['validation', 'combinations'],
                             'selection_metric': 'overall CIDEr', 'test_unseen_during_selection': True})
        return unique

    def stage(self, stage):
        if stage == 'test':
            candidates = self.freeze()
        elif stage == 'screen':
            candidates = [*self.grid.values(), *({'id': name, 'family': 'H'} for name in self.settings['h_policies'])]
        else:
            candidates = self.selections(stage)
        rows, images, sha = self.dataset(stage)
        metadata = {'stage': stage, 'manifest_sha256': sha, 'region_ids_sha256': digest([r['region_id'] for r in rows]),
                    'seed': self.settings['seed'], 'model_revision': MODEL_REVISION, 'runner_version': self.version}
        folder = self.output / stage
        scorer = StageScores(folder, rows, metadata, self.settings['metrics'])
        cache = self.cache(rows, images, sha)
        ids = [r['region_id'] for r in rows]
        baseline = self.existing_predictions(self.grid['baseline'], sha)
        for group in ('ABCDEFGHIJ' if stage == 'screen' else ('all',)):
            part = [c for c in candidates if group == 'all' or c['family'] == group or (group == 'A' and c['family'] == 'control')]
            if part:
                baseline = self.run_candidates(scorer, cache, part, ids, sha, baseline)
        if stage == 'screen':
            baseline = self.run_candidates(scorer, cache, self.derived_screen(scorer, cache, ids), ids, sha, baseline)
        scorer.export()
        write_json(folder / 'feature_cache_run.json', {'new_image_or_crop_encodes': cache.encodes,
                                                       'cache_build_seconds': cache.seconds})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('screen', 'validation', 'combinations', 'test', 'all'), default='all')
    parser.add_argument('--config', type=Path, default=CONFIGS / 'exploration.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'exploration')
    parser.add_argument('--kv-cache', action='store_true', help='Reuse the decoder KV state only after parity verification')
    args = parser.parse_args()
    settings = read_json(args.config)
    random.seed(settings['seed'])
    torch.manual_seed(settings['seed'])
    task = Exploration(settings, args)
    for stage in ('screen', 'validation', 'combinations', 'test') if args.stage == 'all' else (args.stage,):
        task.stage(stage)


if __name__ == '__main__':
    main()
