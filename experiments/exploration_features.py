"""Versioned visual feature cache and single-region feature extraction."""

from collections import defaultdict
from functools import lru_cache
import math
from pathlib import Path
from time import perf_counter

from PIL import Image
import torch
from torchvision import transforms as T

from multiscale import clip_box, detail_geometry


@lru_cache(maxsize=4)
def alternate_transform(size):
    return T.Compose([
        T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def image_rgb(path):
    with Image.open(path) as file:
        return file.convert("RGB")


def save_atomic(value, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


class VisualFeatures:
    def __init__(self, captioner, batch_size):
        from exploration_methods import aggregate_patches, count_and_coherence

        self.captioner = captioner
        self.batch_size = batch_size
        self.device = captioner.device
        self.patch_size = captioner.patch_size
        self.dimension = captioner.model.embed_dim
        self.pool = aggregate_patches
        self.signals = count_and_coherence

    def transform(self, image, size):
        if size == self.captioner.model.resize_dim:
            return self.captioner.model.image_transforms_no_crop(image)
        if size <= 0 or size % self.patch_size:
            raise ValueError(f"Resolution must be a positive multiple of {self.patch_size}: {size}")
        return alternate_transform(size)(image)

    @torch.inference_mode()
    def encode(self, batch):
        tensor = torch.stack(batch).to(self.device)
        shape = tuple(dim // self.patch_size for dim in tensor.shape[-2:])
        try:
            patches = self.captioner.model.dino(tensor, is_training=True)["x_norm_patchtokens"]
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
        else:
            if patches.shape != (len(batch), math.prod(shape), self.dimension):
                raise RuntimeError(f"Unexpected patch shape {patches.shape} for grid {shape}")
            return patches, shape
        del tensor
        torch.cuda.empty_cache()
        midpoint = len(batch) // 2
        left, shape = self.encode(batch[:midpoint])
        right, _ = self.encode(batch[midpoint:])
        return torch.cat((left, right)), shape

    def pool_original(self, patches, shape, image, box, method):
        size = self.captioner.model.resize_dim
        if method == "official":
            dim_marker = torch.empty((0, 0, size, size))
            boxes = self.captioner.upstream_boxes(image, box, dim_marker)
            return self.captioner._original_feature(patches, boxes)
        resized = box.scaled(size / image.width, size / image.height)
        return self.pool(patches, resized, shape, self.patch_size, method)

    def pool_context(self, patches, shape, image, box, scale, method):
        size = self.captioner.model.resize_dim
        sx, sy = size / image.width, size / image.height
        outer = clip_box(box.expanded(scale).xywh, image.size).scaled(sx, sy)
        value = self.pool(patches, box.scaled(sx, sy), shape, self.patch_size, method, outer_box=outer)
        return value if value is not None else patches.new_zeros((1, self.dimension))

    def prepare_detail(self, image, box, scale, size):
        bounds, target = detail_geometry(box, image.size, scale)
        crop = image.crop(bounds)
        resized = target.scaled(size / crop.width, size / crop.height)
        return self.transform(crop, size), resized

    def get_signals(self, patches, shape, image, box):
        size = self.captioner.model.resize_dim
        resized = box.scaled(size / image.width, size / image.height)
        stats = self.signals(patches, resized, shape, self.patch_size)
        stats['area_fraction'] = box.w * box.h / (image.width * image.height)
        return stats

    @torch.inference_mode()
    def online(self, image, bbox, config):
        from exploration_methods import compose, detail_keys, required_features

        box = clip_box(bbox, image.size)
        keys = set(required_features(config))
        features = {}
        signals = None
        if config['id'] == 'detail_only':
            # The canonical all-detail control has no dependency on the full-image encoder.
            signals = {key: torch.zeros(1) for key in ('area_fraction', 'patch_count', 'short_side', 'coherence')}
            signals['area_fraction'][0] = box.w * box.h / (image.width * image.height)
            features['original:official'] = torch.zeros((1, self.dimension), device=self.device)
        else:
            patch_map, shape = self.encode([self.transform(image, self.captioner.model.resize_dim)])
            signals = tensor_signals([self.get_signals(patch_map, shape, image, box)])
            for key in keys:
                parts = key.split(':')
                if parts[0] == 'original':
                    features[key] = self.pool_original(patch_map, shape, image, box, parts[1])
                if parts[0] == 'context':
                    features[key] = self.pool_context(patch_map, shape, image, box, float(parts[1]), parts[2])
        selected = set(detail_keys(config, signals))
        for key in keys:
            kind, *parts = key.split(':')
            if kind != 'detail':
                continue
            if key not in selected:
                features[key] = torch.zeros((1, self.dimension), device=self.device)
                continue
            scale, size, method = parts
            batch, target = self.prepare_detail(image, box, float(scale), int(size))
            patches, shape = self.encode([batch])
            features[key] = self.pool(patches, target, shape, self.patch_size, method)
        fused, weights = compose(config, features, signals)
        return fused, weights, bool(selected)


def tensor_signals(stats):
    return {name: torch.tensor([row[name] for row in stats], dtype=torch.float32) for name in stats[0]}


class FeatureCache:
    def __init__(self, directory, visual, regions, images):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.visual = visual
        self.regions = regions
        self.images = images
        self.data = {}
        self.seconds = 0.0
        self.encodes = 0

    def cache_path(self, key):
        return self.directory / (key.replace(':', '_') + '.pt')

    def items(self, key):
        if key not in self.data:
            location = self.cache_path(key)
            self.data[key] = torch.load(location, map_location='cpu', weights_only=True) if location.exists() else {}
        return self.data[key]

    def put(self, key, region_id, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().reshape(self.visual.dimension)
            if not torch.isfinite(value).all():
                raise ValueError(f"Nonfinite feature {key} for {region_id}")
        self.items(key)[region_id] = value

    def checkpoint(self, keys):
        for key in keys:
            save_atomic(self.items(key), self.cache_path(key))

    def needed(self, key, region_ids):
        return set(region_ids).difference(self.items(key))

    @torch.inference_mode()
    def fill(self, keys, region_ids):
        keys = set(keys) | {'signals'}
        pending = {key: self.needed(key, region_ids) for key in keys}
        started = perf_counter()
        original_keys = sorted(key for key in keys if key == 'signals' or key.startswith(('original:', 'context:')))
        originals = set().union(*(pending[key] for key in original_keys))
        grouped = defaultdict(list)
        for region_id in region_ids:
            if region_id in originals:
                grouped[self.images[region_id]].append(region_id)
        size = self.visual.captioner.model.resize_dim
        for index, (path, ids) in enumerate(grouped.items(), start=1):
            image = image_rgb(path)
            patches, shape = self.visual.encode([self.visual.transform(image, size)])
            self.encodes += 1
            for region_id in ids:
                box = clip_box(self.regions[region_id]['bbox_xywh'], image.size)
                for key in original_keys:
                    if region_id not in pending[key]:
                        continue
                    parts = key.split(':')
                    if key == 'signals':
                        value = self.visual.get_signals(patches, shape, image, box)
                    elif parts[0] == 'original':
                        value = self.visual.pool_original(patches, shape, image, box, parts[1])
                    else:
                        value = self.visual.pool_context(patches, shape, image, box, float(parts[1]), parts[2])
                    self.put(key, region_id, value)
            if index % 48 == 0:
                self.checkpoint(original_keys)
        if originals:
            self.checkpoint(original_keys)
            print(f"Cached original/context features for {len(originals)} regions from {len(grouped)} images", flush=True)

        details = defaultdict(list)
        for key in sorted(keys):
            if key.startswith('detail:'):
                _, scale, side, _ = key.split(':')
                details[(float(scale), int(side))].append(key)
        # The CPU image cache avoids reopening the same JPEG for every crop of an image.
        load_image = lru_cache(maxsize=16)(image_rgb)
        for (scale, size), group_keys in details.items():
            needed = set().union(*(pending[key] for key in group_keys))
            ids = sorted(needed, key=lambda rid: (self.images[rid], rid))
            if not ids:
                continue
            for start in range(0, len(ids), self.visual.batch_size):
                chosen = ids[start:start + self.visual.batch_size]
                items = []
                for region_id in chosen:
                    image = load_image(self.images[region_id])
                    box = clip_box(self.regions[region_id]['bbox_xywh'], image.size)
                    items.append(self.visual.prepare_detail(image, box, scale, size))
                patches, shape = self.visual.encode([tensor for tensor, _ in items])
                self.encodes += len(items)
                for index, (region_id, (_, target)) in enumerate(zip(chosen, items)):
                    for key in group_keys:
                        if region_id in pending[key]:
                            value = self.visual.pool(patches[index:index + 1], target, shape, self.visual.patch_size, key.split(':')[-1])
                            if value is None:
                                raise RuntimeError(f"Empty crop feature {key} for {region_id}")
                            self.put(key, region_id, value)
                if start and start % (self.visual.batch_size * 12) == 0:
                    self.checkpoint(group_keys)
            self.checkpoint(group_keys)
            print(f"Cached {len(ids)} regions at crop={scale:g}, resolution={size} ({len(group_keys)} pools)", flush=True)
        self.seconds += perf_counter() - started

    def load(self, keys, region_ids):
        features = {key: torch.stack([self.items(key)[rid] for rid in region_ids]) for key in keys}
        signals = tensor_signals([self.items('signals')[rid] for rid in region_ids])
        return features, signals
