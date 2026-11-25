"""Frozen visual features for the Patch-ioner refinement experiments."""

import math

import torch
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from visual_features import VisualFeatures
from boundary_semantic import _specification, count_and_coherence, geometry_signals, refined_pool
from multiscale import Box, detail_geometry, patch_overlap


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def letterbox_dimensions(width, height, size):
    """Use the exact integer content dimensions in the eventual box transform."""
    ratio = size / max(width, height)
    new_width, new_height = max(1, round(width * ratio)), max(1, round(height * ratio))
    left, top = (size - new_width) // 2, (size - new_height) // 2
    return new_width, new_height, left, top


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
