"""Resumable, training-free policy evaluation on disjoint validation and final splits."""

from _paths import CONFIGS, EXPERIMENTS

import argparse
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import random
from statistics import mean
from time import perf_counter

import torch

from explore import StageScores, file_digest, read_json, stratified
from exploration_features import image_rgb, tensor_signals
from multiscale import clip_box, load_captioner
from multiscale_experiment import read_manifest, write_json
from refinement_cache import OFFICIAL, RefinementCache, signature
from refinement_confidence import RefinementDecoder
from refinement_features import RefinementFeatures
from refinement_policies import calibrate, candidate_grid, compose, detail_specs, original_specs, semantic_signature, uses_confidence
from region_smoke import MIB, MODEL_REVISION, ROOT
from vg_metrics import AREA_GROUPS


META = {'id', 'family', 'history_id', 'group', 'description', 'notes', 'role'}


def config_key(config):
    if config.get('rank_policy'):
        return signature([config['rank_policy'], [config_key(c) for c in config['sources']]])
    return signature(semantic_signature({key: value for key, value in config.items() if key not in META}))


def rank_table(data):
    return sorted((r for r in data['summary'] if r['area_group'] == 'all'), key=lambda row: (-row['CIDEr'], row['mode']))


def first_by_history(configs, history):
    return next(config for config in configs if config.get('history_id') == history)


class Refinement:
    def __init__(self, settings, output):
        self.settings = settings
        self.output = output
        output.mkdir(parents=True, exist_ok=True)
        self.previous = ROOT / 'results' / 'exploration'
        self.best = read_json(CONFIGS / 'exploration_published_lock.json')['configs'][-1]
        self.raw_configs = candidate_grid(self.best)
        self.history = None
        self.captioner = None
        self.decoder = None
        self.visual = None
        self.version = signature([MODEL_REVISION, *(file_digest(EXPERIMENTS / name) for name in (
            'refine.py', 'refinement_cache.py', 'refinement_confidence.py', 'refinement_features.py',
            'refinement_policies.py', 'refinement_selection.py')),
            file_digest(ROOT / 'src/multiscale.py'), file_digest(ROOT / 'src/vg_metrics.py')])

    def model(self):
        if self.captioner is None:
            self.captioner = load_captioner()
            self.visual = RefinementFeatures(self.captioner, self.settings['image_batch_size'])
            self.decoder = RefinementDecoder(self.captioner, projection_batch_size=self.settings['decode_batch_size'])

    def dataset(self, stage):
        final = stage == 'test'
        path = ROOT / self.settings['final_test_manifest' if final else 'validation_manifest']
        data, paths, sha = read_manifest(path, path.parent)
        records = data['regions']
        if final:
            validation = read_json(ROOT / self.settings['validation_manifest'])
            old_test = read_json(ROOT / self.settings['previous_test_manifest'])
            used = {r['image_id'] for doc in (validation, old_test) for r in doc['regions']}
            if data['split'] != 'test' or len(records) < self.settings['minimum_test_regions'] or used & {r['image_id'] for r in records}:
                raise ValueError('The locked test must be official, sufficiently large and image-disjoint')
        if stage == 'screen':
            chosen = {r['region_id'] for r in read_json(ROOT / self.settings['screen_manifest'])['regions']}
            records = [r for r in records if r['region_id'] in chosen]
            if len(records) != len(chosen) or not 200 <= len(records) <= 300:
                raise ValueError('Screen manifest must be a fixed subset of validation')
        return records, paths, sha

    def feature_cache(self, records, paths, sha):
        self.model()
        previous = sorted((self.previous / 'features').glob(f'{sha[:12]}-*'))
        return RefinementCache(self.output / 'features' / f'{sha[:12]}-{self.version[:12]}',
                               previous[0] if len(previous) == 1 else None, self.visual, self.decoder, records, paths)

    def prediction_file(self, sha, config):
        directory = self.output / 'predictions' / f'{sha[:12]}-{self.version[:12]}'
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f'{config_key(config)[:32]}.json'

    def from_history(self, sha, config):
        name = config.get('history_id')
        if not name:
            return {}
        if self.history is None:
            self.history = {}
            for stage in ('validation', 'combinations'):
                path = self.previous / stage / 'scores.json'
                if not path.is_file():
                    continue
                data = read_json(path)
                origin = data['metadata']['manifest_sha256']
                for row in data['rows']:
                    if row['mode'] in ('baseline', 'A_w0p3', 'C_piecewise_mixed_support', 'G_detail_overlap_semantic', 'J_dynamic_boundary'):
                        self.history.setdefault((origin, row['mode']), {})[row['region_id']] = row['caption']
        return self.history.get((sha, name), {})

    def plans(self, cache, ids, configs):
        want_conf = any(uses_confidence(c) for c in configs)
        cache.prepare_base(ids, want_conf)
        signals = cache.signals(ids, want_conf)
        layouts = {c['id']: (original_specs(c, signals), detail_specs(c, signals)) for c in configs}
        cache.fill(ids, layouts.values())
        return layouts, signals

    def vectors(self, cache, ids, config, layouts, signals):
        orig, detail = cache.vectors(ids, *layouts[config['id']])
        combined, weights = compose(config, orig, detail, signals)
        if not torch.isfinite(combined).all().item() or not (combined.norm(dim=-1) > 0).all().item():
            raise RuntimeError(f"Invalid composed features for {config['id']}")
        return combined, weights, detail

    def predict(self, cache, ids, config, sha, layouts, signals, baseline):
        vector, weights, detail = self.vectors(cache, ids, config, layouts, signals)
        o_specs, d_specs = layouts[config['id']]
        path = self.prediction_file(sha, config)
        saved = read_json(path) if path.is_file() else {}
        history = self.from_history(sha, config)
        missing = []
        for i, rid in enumerate(ids):
            if rid in saved:
                continue
            active = d_specs[i] is not None
            common = {'detail_active': active, 'detail_layout': d_specs[i], 'detail_weight': weights[i].item(),
                      'cache_decode_ms': None, 'cache_peak_allocated_mib': None}
            if rid in history:
                saved[rid] = {**common, 'caption': history[rid], 'prediction_source': 'previous_validation'}
            elif baseline is not None and rid in baseline and o_specs[i] == OFFICIAL and not active:
                saved[rid] = {**common, 'caption': baseline[rid]['caption'], 'prediction_source': 'exact_baseline_fallback'}
            else:
                missing.append(i)
        step = self.decoder.batch_size
        for start in range(0, len(missing), step):
            indices = missing[start:start + step]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tick = perf_counter()
            captions = self.decoder.decode(vector[indices])
            torch.cuda.synchronize()
            each = (perf_counter() - tick) * 1000 / len(indices)
            peak = torch.cuda.max_memory_allocated() / MIB
            for i, caption in zip(indices, captions, strict=True):
                saved[ids[i]] = {'caption': caption.strip(), 'detail_active': d_specs[i] is not None,
                                 'detail_layout': d_specs[i], 'detail_weight': weights[i].item(),
                                 'cache_decode_ms': each, 'cache_peak_allocated_mib': peak,
                                 'prediction_source': 'cached_features'}
            if start % (step * 8) == 0:
                write_json(path, saved)
        write_json(path, saved)
        return {rid: saved[rid] for rid in ids}, vector, detail

    def predict_ranks(self, cache, ids, configs, sha, baseline):
        parents = configs[0]['sources']
        layouts, signals = self.plans(cache, ids, parents)
        outputs = [self.predict(cache, ids, c, sha, layouts, signals, baseline) for c in parents]
        text = [[p[0][rid]['caption'] for p in outputs] for rid in ids]
        vectors = torch.stack([p[1] for p in outputs], dim=1)
        candidates = {c['id']: (read_json(self.prediction_file(sha, c)) if self.prediction_file(sha, c).is_file() else {}) for c in configs}
        missing = [i for i, rid in enumerate(ids) if any(rid not in rows for rows in candidates.values())]
        methods = sorted({c['rank_policy'] for c in configs})
        for start in range(0, len(missing), 16):
            indices = missing[start:start + 16]
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            tick = perf_counter()
            picks = self.decoder.rank([text[i] for i in indices], vectors[indices, 0], outputs[2][2][indices],
                                      source_vectors=vectors[indices], methods=methods)
            torch.cuda.synchronize()
            each = (perf_counter() - tick) * 1000 / len(indices)
            peak = torch.cuda.max_memory_allocated() / MIB
            for config in configs:
                for i, chosen in zip(indices, picks[config['rank_policy']], strict=True):
                    candidates[config['id']][ids[i]] = {'caption': text[i][chosen], 'detail_active': True,
                        'picked_source': parents[chosen]['id'], 'cache_decode_ms': each,
                        'cache_peak_allocated_mib': peak, 'prediction_source': 'reranking_only'}
            if start % (16 * 8) == 0:
                for c in configs:
                    write_json(self.prediction_file(sha, c), candidates[c['id']])
        for c in configs:
            write_json(self.prediction_file(sha, c), candidates[c['id']])
        return candidates

    def _score_stage(self, stage, cache, records, sha, configs):
        ids = [r['region_id'] for r in records]
        metadata = {'stage': stage, 'manifest_sha256': sha, 'region_ids_sha256': signature(ids), 'runner_version': self.version,
                    'model_revision': MODEL_REVISION, 'seed': self.settings['seed'], 'tokenizer': 'ptb'}
        folder = self.output / stage
        scorer = StageScores(folder, records, metadata, self.settings['metrics'])
        base_config = first_by_history(configs, 'baseline')
        if base_config['id'] != 'baseline':
            raise ValueError('Ranking/metrics require the baseline ID to be baseline')
        old = scorer.existing(base_config)
        baseline = {row['region_id']: row for row in old['rows']} if old else None
        pending = [c for c in configs if scorer.existing(c) is None]
        regular = [c for c in pending if not c.get('rank_policy')]
        if regular:
            layouts, signals = self.plans(cache, ids, regular)
            for config in regular:
                predictions, _, _ = self.predict(cache, ids, config, sha, layouts, signals, baseline)
                if config['id'] == 'baseline':
                    baseline = predictions
                scorer.score(config, predictions, baseline)
        ranked = defaultdict(list)
        for c in pending:
            if c.get('rank_policy'):
                ranked[signature(c['sources'])].append(c)
        for group in ranked.values():
            predictions = self.predict_ranks(cache, ids, group, sha, baseline)
            for config in group:
                scorer.score(config, predictions[config['id']], baseline)
        scorer.export()
        write_json(folder / 'feature_cache_run.json', {'new_image_or_crop_encodes': cache.encode_calls,
                   'prior_exploration_feature_rows_loaded': cache.prior_hits, 'feature_build_seconds': cache.feature_seconds,
                   'confidence_build_seconds': cache.confidence_seconds})
        return read_json(folder / 'scores.json')

    def _reranks(self, configs, data, config_id=None):
        from refinement_selection import choose_selective

        selected = config_id or choose_selective(data)
        source = [first_by_history(configs, key) for key in ('baseline', 'A_w0p3', 'J_dynamic_boundary')]
        candidate = data['metadata']['configs'][selected]
        if not any(config_key(candidate) == config_key(c) for c in source):
            source.append(candidate)
        return [{'id': 'F_' + method.removeprefix('H_').removeprefix('F_'), 'family': 'F', 'rank_policy': method,
                 'sources': source} for method in self.settings['rerank_methods']]

    def _screen(self, cache, records, sha):
        ids = [r['region_id'] for r in records]
        path = self.output / 'calibrated_grid.json'
        if path.is_file():
            configs = read_json(path)
        else:
            cache.prepare_base(ids, True)
            signals = cache.signals(ids, True)
            configs = [calibrate(c, signals) for c in self.raw_configs]
            write_json(path, configs)
        data = self._score_stage('screen', cache, records, sha, configs)
        return self._score_stage('screen', cache, records, sha, [*configs, *self._reranks(configs, data)])

    def stage(self, stage):
        from refinement_selection import combine_candidates, validation_candidates

        if stage == 'test':
            lock = read_json(self.output / 'locked_final_configs.json')
            if lock['runner_version'] != self.version or lock['test_manifest_sha256'] != file_digest(ROOT / self.settings['final_test_manifest']):
                raise ValueError('Locked test inputs have changed')
        records, paths, sha = self.dataset(stage)
        cache = self.feature_cache(records, paths, sha)
        if stage == 'screen':
            return self._screen(cache, records, sha)
        if stage == 'validation':
            screened = read_json(self.output / 'screen' / 'scores.json')
            configs = validation_candidates(screened, self.settings['validation_survivors_per_family'])
            data = self._score_stage(stage, cache, records, sha, configs)
            extra = combine_candidates(data, screened)
            return self._score_stage(stage, cache, records, sha, [*configs, *extra])
        return self._score_stage(stage, cache, records, sha, lock['configs'])

    def online_predict(self, image, region, config):
        from refinement_online import RegionSession

        session = RegionSession(self.captioner, self.visual, self.decoder, image, region['bbox_xywh'])
        return session.predict(config)

    def profile(self, stage):
        from refinement_online import measure_online

        self.model()
        self.decoder.clear_projection_cache()
        records, paths, _ = self.dataset(stage)
        data = read_json(self.output / stage / 'scores.json')
        sample = stratified(records, self.settings['online_sample_size'], self.settings['seed'] + 2)
        configs = list(data['metadata']['configs'].values())
        if stage == 'validation':
            from refinement_selection import online_shortlist

            configs = online_shortlist(data)
        return measure_online(self, stage, self.output / stage, sample, paths, configs, data)

    def lock_final(self, use_published=False):
        from refinement_selection import final_roles

        path = self.output / 'locked_final_configs.json'
        published = read_json(CONFIGS / 'refinement_published_lock.json') if use_published else None
        if path.exists():
            record = read_json(path)
            if published and any(record[k] != published[k] for k in ('configs', 'roles', 'test_manifest_sha256')):
                raise ValueError('The existing run uses a different lock. Choose a new --output directory.')
            return record
        if published:
            if file_digest(ROOT / self.settings['final_test_manifest']) != published['test_manifest_sha256']:
                raise ValueError('The prepared manifest differs from the 5,000-region published test manifest.')
            if (self.settings['metrics'], self.settings['bootstrap_seed'], self.settings['bootstrap_resamples']) != (
                published['secondary_metrics'], published['bootstrap_seed'], published['bootstrap_resamples']
            ):
                raise ValueError('The active metric or resampling settings differ from the published settings.')
            record = {**published, 'runner_version': self.version, 'lock_source': 'published'}
            write_json(path, record)
            print('Loaded published final roles:', record['roles'], flush=True)
            return record
        val = read_json(self.output / 'validation' / 'scores.json')
        online = read_json(self.output / 'validation' / 'online.json')
        roles, selected, decisions = final_roles(val, online, self.settings['extra_latency_target'])
        record = {'roles': roles, 'configs': selected, 'decision': decisions, 'runner_version': self.version,
                  'primary_metric': 'CIDEr', 'secondary_metrics': self.settings['metrics'],
                  'test_manifest_sha256': file_digest(ROOT / self.settings['final_test_manifest']),
                  'val_scores_sha256': file_digest(self.output / 'validation' / 'scores.json'),
                  'bootstrap_unit': 'image', 'paired': True, 'bootstrap_seed': self.settings['bootstrap_seed'],
                  'bootstrap_resamples': self.settings['bootstrap_resamples'], 'test_unseen_during_selection': True}
        write_json(path, record)
        print('Locked final roles:', roles, flush=True)
        return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('screen', 'validation', 'test', 'all'), default='all')
    parser.add_argument('--config', type=Path, default=CONFIGS / 'refinement.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'refinement')
    parser.add_argument('--use-published-policies', action='store_true',
                        help='Run the exact released test policies without repeating validation')
    args = parser.parse_args()
    if args.use_published_policies and args.stage != 'test':
        parser.error('--use-published-policies requires --stage test')
    settings = read_json(args.config)
    random.seed(settings['seed'])
    torch.manual_seed(settings['seed'])
    runner = Refinement(settings, args.output)
    stages = ('screen', 'validation', 'test') if args.stage == 'all' else (args.stage,)
    for stage in stages:
        if stage == 'test':
            runner.lock_final(args.use_published_policies)
        runner.stage(stage)
        if stage in ('validation', 'test'):
            runner.profile(stage)


if __name__ == '__main__':
    main()
