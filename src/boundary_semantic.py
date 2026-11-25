"""Patch aggregation with overlap and center-anchored semantic weighting."""

import math

import torch
import torch.nn.functional as F

from multiscale import context_patch_weights, patch_overlap, weighted_pool


def _centers(grid_shape, patch_size):
    rows, columns = grid_shape
    return torch.meshgrid(
        (torch.arange(rows, dtype=torch.float64) + 0.5) * patch_size,
        (torch.arange(columns, dtype=torch.float64) + 0.5) * patch_size,
        indexing="ij",
    )


def _center_distance(box, grid_shape, patch_size):
    y, x = _centers(grid_shape, patch_size)
    dx = (x - (box.x + box.w / 2)) / max(box.w / 2, patch_size / 2)
    dy = (y - (box.y + box.h / 2)) / max(box.h / 2, patch_size / 2)
    return dx.square() + dy.square()


def _cosine(a, b):
    return F.cosine_similarity(a, b, dim=-1, eps=1e-6).clamp(-1.0, 1.0)


def aggregate_patches(patches, resized_box, grid_shape, patch_size, method, outer_box=None):
    """Pool one patch grid [1, P, D] with exact rectangle support or a context ring."""
    if patches.ndim != 3 or patches.shape[:2] != (1, math.prod(grid_shape)):
        raise ValueError(f"Expected a single grid {grid_shape}, got patches {patches.shape}")
    target_area = patch_overlap(resized_box, grid_shape, patch_size)
    if outer_box is not None:
        ring = context_patch_weights(resized_box, outer_box, grid_shape, patch_size)
        if not (ring > 0).any().item() or method == "uniform":
            return weighted_pool(patches, ring)
        if method == "distance":
            y, x = _centers(grid_shape, patch_size)
            dx = ((x - (resized_box.x + resized_box.w / 2)).abs() - resized_box.w / 2).clamp_min(0)
            dy = ((y - (resized_box.y + resized_box.h / 2)).abs() - resized_box.h / 2).clamp_min(0)
            return weighted_pool(patches, ring * torch.exp(-torch.hypot(dx, dy) / (2 * patch_size)))
        if method == "similarity":
            reference = weighted_pool(patches, target_area)
            if reference is None:
                raise ValueError("Context selection needs at least one target patch")
            ring_mask = (ring > 0).flatten().to(patches.device)
            selected_patch = patches[0, ring_mask]
            similarity = _cosine(selected_patch, reference)
            # Keep the upper half of ring cells by similarity, then smooth their weights.
            score = torch.exp((similarity - 1) / 0.25) * (similarity >= similarity.median())
            effective = ring.flatten().to(device=patches.device, dtype=patches.dtype)[ring_mask] * score
            return (selected_patch * (effective / effective.sum())[:, None]).sum(dim=0, keepdim=True)
        raise ValueError(f"Unknown context aggregation: {method}")

    if method == "overlap":
        return weighted_pool(patches, target_area)
    distance = _center_distance(resized_box, grid_shape, patch_size)
    target_mask = target_area > 0
    if not target_mask.any().item():
        return None
    if method == "center":
        # Use the true box center and an inverse-distance kernel on exact support.
        # The official Gaussian is centered on a quantized patch rectangle instead.
        return weighted_pool(patches, target_mask / (1 + distance))
    if method in ("semantic", "overlap_semantic"):
        closest = distance.masked_fill(~target_mask, math.inf).argmin().item()
        anchor = patches[:, closest]
        similarity = _cosine(patches[0], anchor)
        weights = torch.exp((similarity - 1) / 0.25)
        geometry = target_area if method == "overlap_semantic" else target_mask
        effective = geometry.flatten().to(device=patches.device, dtype=patches.dtype) * weights
        return (patches * (effective / effective.sum())[None, :, None]).sum(dim=1)
    raise ValueError(f"Unknown target aggregation: {method}")


def geometry_signals(resized_box, grid_shape, patch_size):
    support = patch_overlap(resized_box, grid_shape, patch_size) > 0
    return {
        "patch_count": support.sum().item(),
        "short_side": min(support.any(dim=0).sum().item(), support.any(dim=1).sum().item()),
        "area_fraction": resized_box.w * resized_box.h / (math.prod(grid_shape) * patch_size**2),
    }


def count_and_coherence(patches, resized_box, grid_shape, patch_size):
    """Mean pairwise cosine without materializing a patch-pair matrix."""
    output = geometry_signals(resized_box, grid_shape, patch_size)
    support = (patch_overlap(resized_box, grid_shape, patch_size) > 0).flatten()
    vectors = F.normalize(patches[0, support.to(patches.device)], dim=-1, eps=1e-6)
    count = output["patch_count"]
    if count < 2:
        # No pairwise disagreement is observable for a single patch.
        output["coherence"] = 1.0
    else:
        pairwise_total = vectors.sum(dim=0).square().sum() - vectors.square().sum()
        output["coherence"] = (pairwise_total / (count * (count - 1))).clamp(-1.0, 1.0).item()
    return output


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
