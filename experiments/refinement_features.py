"""Frozen visual features for the Patch-ioner refinement experiments."""

import math

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from exploration_features import VisualFeatures
from exploration_methods import _center_distance, aggregate_patches, count_and_coherence, geometry_signals
from multiscale import Box, detail_geometry, patch_overlap


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def letterbox_dimensions(width, height, size):
    """Use the exact integer content dimensions in the eventual box transform."""
    ratio = size / max(width, height)
    new_width, new_height = max(1, round(width * ratio)), max(1, round(height * ratio))
    left, top = (size - new_width) // 2, (size - new_height) // 2
    return new_width, new_height, left, top


def _specification(spec):
    if isinstance(spec, str):
        return spec, {}
    return spec.get('method', 'overlap_semantic'), {key: val for key, val in spec.items() if key != 'method'}


def _dynamic_alpha(area_fraction, rule):
    if not isinstance(area_fraction, (float, int)) or not math.isfinite(area_fraction) or not 0 <= area_fraction <= 1:
        raise ValueError('Dynamic weighting requires an original image area fraction in [0, 1]')
    if rule == 'large_fade':
        return min(1.0, max(0.0, (0.20 - area_fraction) / 0.13))
    if rule not in (None, 'size_adaptive'):
        raise ValueError(f'Unknown dynamic boundary rule: {rule}')
    relative_size = min(1.0, max(0.0, (area_fraction - 0.02) / 0.08))
    return 1.0 - 0.65 * relative_size


def refined_pool(patches, target, shape, patch_size, spec, *, area_fraction=None):
    """Pool one rectangular support in the existing DINO feature space."""
    method, opts = _specification(spec)
    if patches.ndim != 3 or patches.shape[:2] != (1, math.prod(shape)):
        raise ValueError(f'Expected a single grid of shape {shape}, got {patches.shape}')
    if method == 'dynamic' and set(opts) <= {'dynamic_rule'}:
        alpha = _dynamic_alpha(area_fraction, opts.get('dynamic_rule'))
        if alpha == 1:
            return refined_pool(patches, target, shape, patch_size, 'overlap_semantic')
        plain = refined_pool(patches, target, shape, patch_size, 'overlap')
        return plain if alpha == 0 else plain.lerp(
            refined_pool(patches, target, shape, patch_size, 'overlap_semantic'), alpha
        )
    standard = ('overlap', 'center', 'semantic', 'overlap_semantic')
    if not opts and method in standard:
        result = aggregate_patches(patches, target, shape, patch_size, method)
        if result is None:
            raise ValueError('The target box covers no patches after resizing')
        return result
    if method not in (*standard, 'boundary', 'suppress', 'dynamic', 'residual', 'mean'):
        raise ValueError(f'Unknown parameterized patch pool: {method}')

    areas = patch_overlap(target, shape, patch_size).flatten()
    support = areas > 0
    if not support.any().item():
        raise ValueError('The target box covers no patches after resizing')
    if method == 'mean':
        return patches[:, support.to(patches.device)].mean(dim=1)

    area_power = float(opts.get('area_power', 0.0 if method == 'semantic' else 1.0))
    semantic_power = float(opts.get('semantic_power', 0.0 if method == 'overlap' else 1.0))
    temperature = float(opts.get('temperature', 0.25))
    if (not all(map(math.isfinite, (area_power, semantic_power, temperature)))
            or min(area_power, semantic_power) < 0 or temperature <= 0):
        raise ValueError('Pooling powers must be nonnegative and temperature must be positive')

    fraction = (areas / patch_size**2).to(patches).clamp(0, 1)
    support_device = support.to(patches.device)
    distance = _center_distance(target, shape, patch_size).flatten()
    anchor_index = distance.masked_fill(~support, math.inf).argmin().item()
    cosine = F.cosine_similarity(patches[0], patches[0, anchor_index], dim=-1, eps=1e-6).clamp(-1, 1)
    semantic = torch.exp(((cosine - 1) * semantic_power / temperature).clamp_min(-70))
    geometry = fraction.pow(area_power)
    effective = geometry * semantic
    if method == 'boundary' or opts.get('boundary_only'):
        # Full interior patches retain a multiplier of one before normalization.
        effective = torch.where(fraction >= 1 - 1e-6, 1.0, effective)
    if method == 'suppress' or opts.get('suppress_boundary'):
        # Suppress partial-overlap patches, with weights in [0, 1].
        effective = torch.where(fraction >= 1 - 1e-6, 1.0, effective * fraction)
    effective = effective.masked_fill(~support_device, 0)
    total = effective.sum()
    # Tiny valid boxes can underflow at high area powers. Preserve their overlap pool.
    denominator = total.clamp_min(torch.finfo(patches.dtype).tiny)
    weighted = (patches * (effective / denominator)[None, :, None]).sum(dim=1)
    missing = total <= torch.finfo(patches.dtype).tiny
    if method in ('dynamic', 'residual') or 'residual' in opts or opts.get('size_adaptive'):
        if method == 'dynamic' or opts.get('size_adaptive'):
            alpha = _dynamic_alpha(area_fraction, opts.get('dynamic_rule'))
        else:
            alpha = float(opts.get('residual', 0.5))
            if not math.isfinite(alpha) or not 0 <= alpha <= 1:
                raise ValueError('Residual mixture coefficient must be in [0, 1]')
        plain = aggregate_patches(patches, target, shape, patch_size, 'overlap')
        return torch.where(missing, plain, plain.lerp(weighted, alpha))
    if missing.item():
        return aggregate_patches(patches, target, shape, patch_size, 'overlap')
    return weighted


class RefinementFeatures(VisualFeatures):
    def prepare_refined_detail(self, image, box, scale, size, layout='square'):
        if not math.isfinite(scale) or scale < 1:
            raise ValueError('Crop scale must be at least one')
        if not isinstance(size, int) or size <= 0 or size % self.patch_size:
            raise ValueError(f'Resolution must be a positive multiple of {self.patch_size}: {size}')
        if layout == 'square':
            return self.prepare_detail(image, box, scale, size)
        if layout != 'letterbox':
            raise ValueError(f'Unknown crop layout: {layout}')

        bounds, target = detail_geometry(box, image.size, scale)
        crop = image.crop(bounds)
        width, height, left, top = letterbox_dimensions(crop.width, crop.height, size)
        resized = TF.resize(crop, [height, width], InterpolationMode.BICUBIC)
        content = TF.normalize(TF.to_tensor(resized), IMAGENET_MEAN, IMAGENET_STD)
        padded = content.new_zeros((3, size, size))
        padded[:, top:top + height, left:left + width] = content
        target = target.scaled(width / crop.width, height / crop.height)
        canvas_target = Box(target.x + left, target.y + top, target.w, target.h)
        return padded, canvas_target

    def pool_variant(self, patches, target, shape, spec, *, area_fraction=None):
        return refined_pool(patches, target, shape, self.patch_size, spec, area_fraction=area_fraction)

    def pool_original_variant(self, patches, shape, image, box, spec):
        method, _ = _specification(spec)
        if method == 'official':
            # Preserve the repository's quantized support for the exact baseline.
            return self.pool_original(patches, shape, image, box, 'official')
        size = self.captioner.model.resize_dim
        resized = box.scaled(size / image.width, size / image.height)
        fraction = box.w * box.h / (image.width * image.height)
        return self.pool_variant(patches, resized, shape, spec, area_fraction=fraction)

    def get_geometry_signals(self, image, box):
        """Geometry-only gates can avoid any GPU-derived uncertainty computation."""
        size = self.captioner.model.resize_dim
        resized = box.scaled(size / image.width, size / image.height)
        grid = (size // self.patch_size, size // self.patch_size)
        values = geometry_signals(resized, grid, self.patch_size)
        values['area_fraction'] = box.w * box.h / (image.width * image.height)
        values['aspect_ratio'] = max(box.w, box.h) / min(box.w, box.h)
        return values

    def get_signals(self, patches, shape, image, box):
        size = self.captioner.model.resize_dim
        resized = box.scaled(size / image.width, size / image.height)
        stats = count_and_coherence(patches, resized, shape, self.patch_size)
        areas = patch_overlap(resized, shape, self.patch_size).flatten()
        support = (areas > 0).to(patches.device)
        selected = patches[0, support]
        if len(selected) == 0:
            raise ValueError('The original target box covers no patches')

        center = selected.mean(dim=0)
        cosine = F.cosine_similarity(selected, center, dim=-1, eps=1e-6).clamp(-1, 1)
        dispersion = ((selected - center).square().mean() / selected.square().mean().clamp_min(1e-12)).sqrt()
        normalized_areas = (areas / areas.sum()).to(patches)
        overlap_center = (patches[0] * normalized_areas[:, None]).sum(dim=0)
        change = 1 - F.cosine_similarity(center, overlap_center, dim=0, eps=1e-6).clamp(-1, 1)
        measurements = torch.stack((cosine.mean(), cosine.var(unbiased=False), dispersion, change)).tolist()
        names = ('mean_cosine', 'variance_cosine', 'dispersion', 'aggregation_change')
        stats.update(zip(names, measurements, strict=True))
        stats['aspect_ratio'] = max(box.w, box.h) / min(box.w, box.h)
        stats['area_fraction'] = box.w * box.h / (image.width * image.height)
        return stats
