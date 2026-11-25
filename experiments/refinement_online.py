"""Paired online inference with pre-crop gating and per-region feature reuse."""

import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import torch

from exploration_features import image_rgb, tensor_signals
from multiscale import clip_box
from multiscale_experiment import write_json
from refinement_cache import OFFICIAL, canonical, signature
from refinement_confidence import MEMORY_SIGNALS, TOKEN_SIGNALS
from refinement_policies import compose, detail_specs, detail_weight, original_specs
from region_smoke import MIB
from vg_metrics import AREA_GROUPS


GEOMETRY_SIGNALS = {'area_fraction', 'patch_count', 'short_side', 'aspect_ratio', 'mixed_support'}


def _gate_signals(config):
    gate = config.get('gate') or {}
    if 'fixed_signal' in gate:
        return {gate['fixed_signal']}
    return {name for name, _ in gate.get('components', [])}


class RegionSession:
    """One requested region; intermediate computations never cross a timed call."""

    def __init__(self, captioner, visual, decoder, image, bbox):
        self.captioner = captioner
        self.visual = visual
        self.decoder = decoder
        self.image = image
        self.bbox = tuple(bbox)
        self.box = clip_box(bbox, image.size)
        self.geometry = None
        self.full = None
        self.visual_signals = None
        self.confidence = {}
        self.originals = {}
        self.crops = {}
        self.detail_pools = {}
        self.prefix = None
        self.original_caption = None
        self.confidence_projection_ms = 0.0
        self.confidence_decode_ms = 0.0
        self.visual_signal_ms = 0.0
        self.candidate_generation_ms = 0.0
        self.rerank_ms = 0.0

    def _sync(self):
        device = torch.device(self.captioner.device)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def _full(self):
        if self.full is None:
            side = self.captioner.model.resize_dim
            self.full = self.visual.encode([self.visual.transform(self.image, side)])
        return self.full

    def _original(self, spec):
        key = canonical(spec)
        if key not in self.originals:
            patches, shape = self._full()
            self.originals[key] = self.visual.pool_original_variant(patches, shape, self.image, self.box, spec)
        return self.originals[key]

    def _detail(self, address):
        scale, resolution, preserve, spec = address
        layout = (scale, resolution, preserve)
        if layout not in self.crops:
            tensor, target = self.visual.prepare_refined_detail(
                self.image, self.box, scale, resolution, 'letterbox' if preserve else 'square'
            )
            patches, shape = self.visual.encode([tensor])
            self.crops[layout] = (patches, shape, target)
        key = canonical(address)
        if key not in self.detail_pools:
            patches, shape, target = self.crops[layout]
            fraction = self.box.w * self.box.h / (self.image.width * self.image.height)
            self.detail_pools[key] = self.visual.pool_variant(patches, target, shape, spec, area_fraction=fraction)
        return self.detail_pools[key]

    def _collect_confidence(self, needed):
        missing = needed.difference(self.confidence)
        memory_keys = tuple(name for name in MEMORY_SIGNALS if name in missing)
        token_keys = tuple(name for name in TOKEN_SIGNALS if name in missing)
        if not memory_keys and not token_keys:
            return
        if memory_keys or self.prefix is None:
            original = self._original(OFFICIAL)
            self._sync()
            start = perf_counter()
            self.prefix, signals = self.decoder.project_with_signals(original, cached=False, signals=memory_keys)
            self._sync()
            self.confidence_projection_ms += (perf_counter() - start) * 1000
            self.confidence.update(signals)
        if token_keys:
            self._sync()
            start = perf_counter()
            values, signals = self.decoder.decode_prefix_with_signals(self.prefix, cached=False, signals=token_keys)
            self._sync()
            self.confidence_decode_ms += (perf_counter() - start) * 1000
            self.confidence.update(signals)
            self.original_caption = values[0]

    def _signals(self, config):
        if self.geometry is None:
            self.geometry = tensor_signals([self.visual.get_geometry_signals(self.image, self.box)])
        needed = _gate_signals(config)
        if not needed or not detail_weight(config, self.geometry).any().item():
            return self.geometry
        visual_names = needed.difference(GEOMETRY_SIGNALS, MEMORY_SIGNALS, TOKEN_SIGNALS)
        if visual_names and self.visual_signals is None:
            patches, shape = self._full()
            self._sync()
            start = perf_counter()
            self.visual_signals = tensor_signals([self.visual.get_signals(patches, shape, self.image, self.box)])
            self._sync()
            self.visual_signal_ms += (perf_counter() - start) * 1000
        self._collect_confidence(needed)
        return {**(self.visual_signals or self.geometry), **self.confidence}

    def _region(self, config):
        signals = self._signals(config)
        pool = original_specs(config, signals)[0]
        address = detail_specs(config, signals)[0]
        original = self._original(pool)
        detail = None if address is None else self._detail(address)
        result, weights = compose(config, original, detail, signals)
        return result, detail, pool, address, weights.item()

    def _caption(self, vector, pool, detail):
        if detail is None and pool == OFFICIAL:
            if self.original_caption is None:
                if self.prefix is None:
                    values = self.decoder.decode(vector, cached=False)
                else:
                    values, _ = self.decoder.decode_prefix_with_signals(self.prefix, cached=False, signals=())
                self.original_caption = values[0]
            return self.original_caption
        return self.decoder.decode(vector, cached=False)[0]

    def _rerank(self, config):
        self._sync()
        start = perf_counter()
        # Collect confidence first so a baseline candidate can reuse its full decode.
        for source in config['sources']:
            self._signals(source)
        regions = [self._region(source) for source in config['sources']]
        candidates = [self._caption(vector, pool, detail) for vector, detail, pool, _, _ in regions]
        self._sync()
        self.candidate_generation_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        # The sorted source contract is baseline, fixed, previous best, selected policy.
        # Obtain pure detail from the best policy before blending with the full image.
        best_config = config['sources'][2]
        detail = regions[2][1]
        if detail is None:
            unrestricted = {key: val for key, val in best_config.items() if key != 'gate'}
            address = detail_specs(unrestricted, self._signals(unrestricted))[0]
            if address is not None:
                detail = self._detail(address)
        if detail is None:
            raise ValueError('Reranking needs a true detail feature from the preceding best method')
        method = config['rank_policy']
        source_vectors = torch.cat([r[0] for r in regions]).unsqueeze(0)
        ranked = self.decoder.rank([candidates], regions[0][0], detail, source_vectors=source_vectors,
                                   methods=(method,), cached=False)
        self._sync()
        self.rerank_ms = (perf_counter() - start) * 1000
        index = ranked[method][0]
        return candidates[index], config['sources'][index]['id']

    @torch.inference_mode()
    def predict(self, config):
        source = None
        address = None
        weight = None
        if config.get('rank_policy'):
            caption, source = self._rerank(config)
        elif config['id'] == 'baseline':
            # The baseline timing uses the original upstream singleton call.
            caption = self.captioner.predict(self.image, self.bbox, 'baseline')['caption']
            weight = 0.0
        else:
            vector, detail, pool, address, weight = self._region(config)
            caption = self._caption(vector, pool, detail)
        return {
            'caption': caption.strip(), 'detail_active': bool(self.crops), 'crop_encoder_calls': len(self.crops),
            'detail_layout': address, 'detail_weight': weight, 'picked_source': source,
            'confidence_ms': self.confidence_projection_ms + self.confidence_decode_ms,
            'confidence_projection_ms': self.confidence_projection_ms, 'confidence_decode_ms': self.confidence_decode_ms,
            'visual_signal_ms': self.visual_signal_ms, 'candidate_generation_ms': self.candidate_generation_ms,
            'rerank_ms': self.rerank_ms,
        }


def measure_online(runner, stage, folder, sample, paths, configs, data):
    """Rotate the method order for each region and checkpoint completed pairs."""
    folder = Path(folder)
    save_path = folder / 'online.json'
    partial = folder / 'online_parts'
    partial.mkdir(parents=True, exist_ok=True)
    sample_ids = [r['region_id'] for r in sample]
    config_ids = [c['id'] for c in configs]
    selector = signature([stage, data['metadata']['manifest_sha256'], data['metadata'].get('runner_version'),
                          sample_ids, configs])
    metadata = {'stage': stage, 'latency_kind': 'online', 'interleaved': True, 'sample_size': len(sample),
                'manifest_sha256': data['metadata']['manifest_sha256'], 'sample_region_ids': sample_ids,
                'config_ids': config_ids, 'source_signature': selector,
                'timing_scope': 'Preprocessed region geometry, GPU encoding, pooling, projection and decode; excludes JPEG read and model load.',
                'confidence_timing_scope': 'Required original projection/decoding and confidence extraction, including reusable computations.'}
    if save_path.exists():
        previous = json.loads(save_path.read_bytes())
        if previous['metadata']['source_signature'] != selector:
            raise ValueError(f'Online output belongs to different inputs: {save_path}')
        return previous

    if not sample or not configs or len(set(config_ids)) != len(configs):
        raise ValueError('Online measurement requires a nonempty sample and unique policy configurations')
    captions = {(r['mode'], r['region_id']): r['caption'] for r in data['rows']}
    device = runner.captioner.device

    def measured(region, image, config):
        name = config['id']
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = perf_counter()
        result = runner.online_predict(image, region, config)
        torch.cuda.synchronize(device)
        return {**result, 'mode': name, 'region_id': region['region_id'], 'area_group': region['area_group'],
                'online_ms': (perf_counter() - start) * 1000,
                'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / MIB,
                'peak_reserved_mib': torch.cuda.max_memory_reserved(device) / MIB,
                'same_as_cached': result['caption'] == captions[(name, region['region_id'])]}

    first_image = image_rgb(paths[sample_ids[0]])
    for config in configs:
        for _ in range(2):
            measured(sample[0], first_image, config)

    rows = []
    for index, region in enumerate(sample):
        part = partial / f'{selector[:12]}_{index:04d}.json'
        if part.is_file():
            completed = json.loads(part.read_bytes())
        else:
            image = image_rgb(paths[region['region_id']])
            offset = index % len(configs)
            completed = [measured(region, image, c) for c in configs[offset:] + configs[:offset]]
            write_json(part, completed)
        rows.extend(completed)
        if (index + 1) % 4 == 0:
            print(f'Online {stage}: {index + 1}/{len(sample)} paired regions', flush=True)

    summary = []
    for name in config_ids:
        for group in ('all', *AREA_GROUPS):
            selection = [r for r in rows if r['mode'] == name and (group == 'all' or group == r['area_group'])]
            if not selection:
                continue
            summary.append({
                'mode': name, 'area_group': group, 'regions': len(selection),
                'mean_inference_ms': mean(r['online_ms'] for r in selection),
                'median_inference_ms': median(r['online_ms'] for r in selection),
                'peak_allocated_mib': max(r['peak_allocated_mib'] for r in selection),
                'peak_reserved_mib': max(r['peak_reserved_mib'] for r in selection),
                'detail_activation_rate': mean(r['detail_active'] for r in selection),
                'mean_crop_encoder_calls': mean(r['crop_encoder_calls'] for r in selection),
                'caption_agreement': mean(r['same_as_cached'] for r in selection),
                'mean_confidence_ms': mean(r['confidence_ms'] for r in selection),
                'mean_confidence_projection_ms': mean(r['confidence_projection_ms'] for r in selection),
                'mean_confidence_decode_ms': mean(r['confidence_decode_ms'] for r in selection),
                'mean_visual_signal_ms': mean(r['visual_signal_ms'] for r in selection),
                'mean_candidate_generation_ms': mean(r['candidate_generation_ms'] for r in selection),
                'mean_rerank_ms': mean(r['rerank_ms'] for r in selection),
            })
    output = {'metadata': metadata, 'summary': summary, 'rows': rows}
    write_json(save_path, output)
    return output
