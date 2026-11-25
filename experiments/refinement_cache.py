"""Cache refinement features by actual geometry, pool parameters and region ID."""

from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from exploration_features import image_rgb, save_atomic, tensor_signals
from multiscale import clip_box


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def signature(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def original_key(spec):
    return 'original:' + canonical(spec)


def detail_key(spec):
    return 'detail:' + canonical(spec)


OFFICIAL = {'method': 'official'}


class RefinementCache:
    def __init__(self, folder, prior_folder, visual, decoder, records, image_paths):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.prior_folder = prior_folder
        self.visual = visual
        self.decoder = decoder
        self.records = {r['region_id']: r for r in records}
        self.image_paths = image_paths
        self.load_image = lru_cache(maxsize=24)(image_rgb)
        self.tables = {}
        self.prior_hits = 0
        self.encode_calls = 0
        self.feature_seconds = 0.0
        self.confidence_seconds = 0.0

    def _path(self, key):
        return self.folder / f'{key.partition(":")[0]}_{signature(key)[:24]}.pt'

    def _legacy_name(self, key):
        kind, _, parameters = key.partition(':')
        if kind == 'original':
            spec = json.loads(parameters)
            if isinstance(spec, dict) and set(spec) == {'method'} and spec['method'] in ('official', 'overlap', 'semantic', 'overlap_semantic'):
                return f"original_{spec['method']}.pt"
        if kind == 'detail':
            scale, side, preserve, spec = json.loads(parameters)
            if not preserve and isinstance(spec, dict) and set(spec) == {'method'} and spec['method'] in ('overlap', 'semantic', 'overlap_semantic', 'center'):
                return f"detail_{scale:g}_{side}_{spec['method']}.pt"
        return None

    def table(self, key):
        if key not in self.tables:
            table = {}
            path = self._path(key)
            if path.is_file():
                table = torch.load(path, map_location='cpu', weights_only=True)
            elif self.prior_folder is not None:
                legacy = self._legacy_name(key)
                if legacy and (path := self.prior_folder / legacy).is_file():
                    table = torch.load(path, map_location='cpu', weights_only=True)
                    self.prior_hits += len(table)
            self.tables[key] = table
        return self.tables[key]

    def checkpoint(self, keys):
        for key in keys:
            save_atomic(self.table(key), self._path(key))

    def _put(self, key, rid, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().reshape(self.visual.dimension)
            if not torch.isfinite(value).all().item():
                raise ValueError(f'Nonfinite feature for {rid}/{key}')
        self.table(key)[rid] = value

    def _view(self, rid):
        image = self.load_image(self.image_paths[rid])
        bbox = clip_box(self.records[rid]['bbox_xywh'], image.size)
        return image, bbox

    def _area(self, image, box):
        return box.w * box.h / (image.width * image.height)

    def _originals(self, pending):
        grouped = defaultdict(set)
        for ids in pending.values():
            for rid in ids:
                grouped[self.image_paths[rid]].add(rid)
        if not grouped:
            return
        side = self.visual.captioner.model.resize_dim
        for index, (path, image_regions) in enumerate(sorted(grouped.items()), start=1):
            image = self.load_image(path)
            patches, shape = self.visual.encode([self.visual.transform(image, side)])
            self.encode_calls += 1
            for rid in sorted(image_regions):
                _, box = self._view(rid)
                for key, wanted in pending.items():
                    if rid not in wanted:
                        continue
                    if key == 'signals':
                        data = self.visual.get_signals(patches, shape, image, box)
                    else:
                        data = self.visual.pool_original_variant(patches, shape, image, box, json.loads(key.split(':', 1)[1]))
                    self._put(key, rid, data)
            if index % 32 == 0:
                self.checkpoint(pending)
        self.checkpoint(pending)
        print(f'Cached original features/signals from {len(grouped)} images', flush=True)

    @torch.inference_mode()
    def prepare_base(self, ids, with_confidence):
        started = perf_counter()
        key = original_key(OFFICIAL)
        pending = {name: set(ids).difference(self.table(name)) for name in ('signals', key)}
        if any(pending.values()):
            self.decoder.clear_projection_cache()
            self._originals(pending)
        self.feature_seconds += perf_counter() - started
        if with_confidence:
            self.prepare_confidence(ids)

    def prepare_confidence(self, ids):
        cached = self.table('confidence')
        missing = [rid for rid in ids if rid not in cached]
        started = perf_counter()
        step = self.decoder.batch_size
        for start in range(0, len(missing), step):
            group = missing[start:start + step]
            vectors = torch.stack([self.table(original_key(OFFICIAL))[rid] for rid in group])
            captions, signals = self.decoder.decode_with_signals(vectors)
            for i, rid in enumerate(group):
                self._put('confidence', rid, {**{name: float(value[i]) for name, value in signals.items()},
                                              'baseline_caption': captions[i].strip()})
            if start % (step * 8) == 0:
                self.checkpoint(('confidence',))
        if missing:
            self.checkpoint(('confidence',))
            print(f'Cached baseline confidence for {len(missing)} regions', flush=True)
        self.confidence_seconds += perf_counter() - started

    @torch.inference_mode()
    def fill(self, ids, variants):
        """All requested pool variants for a crop layout share the same visual pass."""
        start = perf_counter()
        needs = defaultdict(set)
        for original_specs, detail_specs in variants:
            for rid, orig, detail in zip(ids, original_specs, detail_specs, strict=True):
                needs[original_key(orig)].add(rid)
                if detail is not None:
                    needs[detail_key(detail)].add(rid)
        pending = {key: requested.difference(self.table(key)) for key, requested in needs.items()}
        if any(pending.values()):
            self.decoder.clear_projection_cache()
        self._originals({key: regions for key, regions in pending.items() if key.startswith('original:') and regions})
        crops = defaultdict(dict)
        for key, regions in pending.items():
            if key.startswith('detail:') and regions:
                scale, size, preserve, pool = json.loads(key.split(':', 1)[1])
                crops[(scale, size, preserve)][key] = (regions, pool)
        for (scale, size, preserve), group in sorted(crops.items()):
            wanted = set().union(*(entry[0] for entry in group.values()))
            ordered = sorted(wanted, key=lambda rid: (self.image_paths[rid], rid))
            step = self.visual.batch_size
            for offset in range(0, len(ordered), step):
                selected = ordered[offset:offset + step]
                prepared = []
                for rid in selected:
                    image, box = self._view(rid)
                    tensor, target = self.visual.prepare_refined_detail(image, box, scale, size,
                                                                      'letterbox' if preserve else 'square')
                    prepared.append((tensor, target, self._area(image, box)))
                patches, shape = self.visual.encode([row[0] for row in prepared])
                self.encode_calls += len(prepared)
                for i, (rid, (_, target, area)) in enumerate(zip(selected, prepared, strict=True)):
                    for key, (regions, pool) in group.items():
                        if rid in regions:
                            vector = self.visual.pool_variant(patches[i:i + 1], target, shape, pool, area_fraction=area)
                            self._put(key, rid, vector)
                if offset and offset % (step * 12) == 0:
                    self.checkpoint(group)
            self.checkpoint(group)
            print(f'Cached {len(ordered)} crops at {scale:g}x/{size}, aspect={preserve}, {len(group)} pool variants', flush=True)
        self.feature_seconds += perf_counter() - start

    def signals(self, ids, include_confidence):
        output = tensor_signals([self.table('signals')[rid] for rid in ids])
        if include_confidence:
            confidence = [{k: v for k, v in self.table('confidence')[rid].items() if k != 'baseline_caption'} for rid in ids]
            output.update(tensor_signals(confidence))
        return output

    def vectors(self, ids, original_specs, detail_specs):
        orig = torch.stack([self.table(original_key(spec))[rid] for rid, spec in zip(ids, original_specs, strict=True)])
        detail = torch.stack([self.table(detail_key(spec))[rid] if spec is not None else orig[i]
                              for i, (rid, spec) in enumerate(zip(ids, detail_specs, strict=True))])
        return orig, detail
