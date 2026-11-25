"""Train-free aggregation, fusion, and early gating for Patch-ioner."""

import math

import torch
import torch.nn.functional as F

from multiscale import context_patch_weights, patch_overlap, weighted_pool


WEIGHTS_A = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1)
CROP_SCALES = (1.0, 1.2, 1.5, 2.0, 3.0)
RESOLUTIONS = (224, 336, 518, 672)
SIZE_BINS = {
    "patch_count": (35, 150),
    "short_side": (4, 10),
    "area_fraction": (0.02, 0.10),
    "mixed_support": (0.24, 0.58),
}


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


def _number(value):
    return format(value, "g")


def detail_key(scale, resolution, pool):
    return f"detail:{_number(scale)}:{resolution}:{pool}"


def _signal(signals, name):
    if name == "mixed_support":
        # Geometric mean of independently normalized support and narrow-side coverage.
        area_support = (signals["patch_count"] / 150).clamp(0, 1)
        short_support = (signals["short_side"] / 10).clamp(0, 1)
        return (area_support * short_support).sqrt()
    return signals[name]


def size_bins(signals, name, limits=None):
    value = _signal(signals, name)
    lower, upper = SIZE_BINS[name] if limits is None else limits
    return (value >= lower).long() + (value >= upper).long()


def gate_mask(config, signals):
    """Decide whether detail is needed using only signals available before a crop."""
    template = signals["area_fraction"]
    gate = config.get("gate")
    if gate is None:
        enabled = config.get("weight_rule", "fixed") != "fixed" or config.get("weight", 0.3) != 0
        return torch.full_like(template, enabled, dtype=torch.bool)
    cutoff = config.get("gate_threshold")
    if gate == "combined":
        return (signals["patch_count"] <= 24) | ((signals["short_side"] <= 5) & (signals["coherence"] <= 0.70))
    return _signal(signals, gate) <= cutoff


def _scales(config):
    return tuple(config.get("detail_scales", (config.get("detail_scale", 1.2),)))


def detail_keys(config, signals=None):
    # With one live region this lets an online runner skip both gated-off crops and unselected scales.
    scales = _scales(config)
    if signals is not None:
        active = gate_mask(config, signals)
        if not active.any().item():
            return ()
        selector = config.get("scale_selector")
        if selector is not None:
            bins = size_bins(signals, selector)[active].unique().tolist()
            scales = tuple(scales[index] for index in bins)
    elif config.get("weight_rule", "fixed") == "fixed" and config.get("weight", 0.3) == 0:
        return ()
    return tuple(detail_key(scale, config.get("detail_resolution", 518), config.get("detail_pool", "overlap")) for scale in scales)


def required_features(config, signals=None):
    keys = [f"original:{config.get('original_pool', 'official')}", *detail_keys(config, signals)]
    if config.get("context_weight", 0) != 0 and (
        signals is None or _context_enabled(config, signals).any().item()
    ):
        keys.append(f"context:{_number(config.get('context_scale', 2.0))}:{config.get('context_pool', 'uniform')}")
    return tuple(keys)


def _size_weight(config, signals, rule):
    mode = config.get("size_signal", "patch_count")
    x = _signal(signals, mode)
    if rule == "piecewise":
        bins = size_bins(signals, mode, config.get("limits"))
        return x.new_tensor(config["weights"])[bins]
    x = x / config["tau"]
    factor = torch.exp(-x) if rule == "exponential" else (1 + x).reciprocal()
    return config["w_max"] * factor


def _detail_vector(config, features, original, signals):
    scales = _scales(config)
    keys = [detail_key(s, config.get("detail_resolution", 518), config.get("detail_pool", "overlap")) for s in scales]
    selector = config.get("scale_selector")
    if selector is not None:
        bins = size_bins(signals, selector)
        selected = bins[gate_mask(config, signals)].unique().tolist()
        missing = [keys[index] for index in selected if keys[index] not in features]
        if missing:
            raise KeyError(f"Missing selected crop features: {missing}")
        values = torch.stack([features.get(key, original) for key in keys], dim=1)
        return values[torch.arange(len(bins), device=bins.device), bins]
    if len(keys) == 1:
        return features[keys[0]]
    values = torch.stack([features[key] for key in keys], dim=1)
    if config.get("scale_rule") == "agreement":
        unit = F.normalize(values, dim=-1, eps=1e-6)
        pair_sum = (unit @ unit.transpose(1, 2)).sum(dim=-1) - unit.square().sum(dim=-1)
        agreement = pair_sum / (len(keys) - 1)
        weights = (agreement / 0.15).softmax(dim=1)
        return (values * weights[..., None]).sum(dim=1)
    return values.mean(dim=1)


def _context_enabled(config, signals):
    if config.get("context_medium_only"):
        return size_bins(signals, "area_fraction") == 1
    return torch.ones_like(signals["area_fraction"], dtype=torch.bool)


def compose(config, features, signals):
    """Compose batch features in the original DINO space and return effective crop weights."""
    original = features[f"original:{config.get('original_pool', 'official')}"]
    signals = {name: value.to(device=original.device, dtype=original.dtype) for name, value in signals.items()}
    active = gate_mask(config, signals)
    weight = torch.zeros_like(signals["area_fraction"])
    fused = original
    if active.any().item():
        detail = _detail_vector(config, features, original, signals)
        rule = config.get("weight_rule", "fixed")
        if rule == "fixed":
            weight.fill_(config.get("weight", 0.3))
        elif rule in ("piecewise", "exponential", "reciprocal"):
            weight = _size_weight(config, signals, rule)
        elif rule == "heterogeneous":
            weight = config["w_max"] * (1 - signals["coherence"].clamp(0.0, 1.0))
        elif rule in ("dissimilar", "aligned", "size_semantic"):
            similarity = _cosine(original, detail).clamp(0.0, 1.0)
            if rule == "size_semantic":
                size_weight = _size_weight(config, signals, config.get("size_rule", "reciprocal"))
                weight = size_weight * (0.5 + 0.5 * (1 - similarity))
            else:
                score = similarity if rule == "aligned" else 1 - similarity
                weight = config["w_max"] * score
        else:
            raise ValueError(f"Unknown crop-weight rule: {rule}")
        weight = weight.clamp(0.0, 1.0).masked_fill(~active, 0)
        if (weight == 1).all().item():
            fused = detail
        else:
            fused = original * (1 - weight[:, None]) + detail * weight[:, None]

    context_weight = config.get("context_weight", 0)
    eligible = _context_enabled(config, signals)
    if context_weight and eligible.any().item():
        key = f"context:{_number(config.get('context_scale', 2.0))}:{config.get('context_pool', 'uniform')}"
        context = features[key]
        valid = eligible & (context.norm(dim=1) > 0)
        strength = valid[:, None].to(original.dtype) * context_weight
        if context_weight > 0:
            fused = fused * (1 - strength) + context * strength
        else:
            # Use the same unnormalized DINO space for literal weak background subtraction.
            fused = fused + strength * context
    return fused, weight


def piecewise_candidate(signal, weights, limits=None):
    if signal not in SIZE_BINS or len(weights) != 3:
        raise ValueError("Piecewise fusion needs a known signal and three weights")
    return {
        "id": f"C_piecewise_{signal}", "family": "C", "weight_rule": "piecewise",
        "size_signal": signal, "limits": list(SIZE_BINS[signal] if limits is None else limits),
        "weights": list(weights),
    }


def candidate_grid():
    """One-factor screening grid. A-derived piecewise weights are added after measuring A."""
    configs = []

    def add(identifier, family, **options):
        configs.append({"id": identifier, "family": family, **options})

    def token(value):
        return _number(value).replace(".", "p")

    for w in WEIGHTS_A:
        name = "baseline" if w == 0 else "detail_only" if w == 1 else f"A_w{token(w)}"
        add(name, "A", weight=w)
    add("prior_context_fusion", "control", weight=0, context_weight=0.2)
    add("prior_multiscale_fusion", "control", weight=1 / 3, context_weight=0.1)

    for scale in CROP_SCALES:
        if scale != 1.2:  # A_w0p3 is the repeated factor level and control.
            tags = ["B", "F5"] if scale >= 2 else ["B"]
            add(f"B_crop{token(scale)}", "B", detail_scale=scale, weight=0.3, categories=tags)
    for resolution in RESOLUTIONS:
        if resolution != 518:
            add(f"B_res{resolution}", "B", detail_resolution=resolution, weight=0.3)

    for rule in ("exponential", "reciprocal"):
        for maximum in (0.3, 0.5, 0.7):
            for tau in (4, 8, 16, 32):
                add(f"C_{rule}_w{token(maximum)}_tau{tau}", "C", weight_rule=rule, w_max=maximum, tau=tau)
    for signal, tau in (("short_side", 4), ("area_fraction", 0.02), ("mixed_support", 0.3)):
        add(f"C_reciprocal_{signal}", "C", weight_rule="reciprocal", size_signal=signal, w_max=0.5, tau=tau)

    for rule in ("dissimilar", "aligned", "heterogeneous"):
        for maximum in (0.3, 0.7):
            add(f"D_{rule}_w{token(maximum)}", "D", weight_rule=rule, w_max=maximum)
    add("D_size_semantic", "D", weight_rule="size_semantic", w_max=0.5, tau=16)

    scales = [1.0, 1.5, 2.0]
    add("E_mean_only", "E", detail_scales=scales, weight=1)
    add("E_mean_fusion", "E", detail_scales=scales, weight=0.3)
    add("E_agreement", "E", detail_scales=scales, scale_rule="agreement", weight=0.3)
    for signal in ("patch_count", "area_fraction"):
        add(f"E_choose_{signal}", "E", detail_scales=scales, scale_selector=signal, weight=0.3)

    for strength in (0.025, 0.05, 0.1):
        add(f"F_weak{token(strength)}", "F", weight=0, context_weight=strength)
    add("F_medium_only", "F", weight=0, context_weight=0.1, context_medium_only=True)
    for pool in ("similarity", "distance"):
        add(f"F_{pool}", "F", weight=0, context_weight=0.1, context_pool=pool)
    for strength in (0.05, 0.1, 0.2):
        add(f"F_suppress{token(strength)}", "F", weight=0, context_weight=-strength)

    for pool in ("overlap", "center", "semantic", "overlap_semantic"):
        add(f"G_original_{pool}", "G", weight=0, original_pool=pool)
    for pool in ("center", "semantic", "overlap_semantic"):
        add(f"G_detail_{pool}", "G", weight=0.3, detail_pool=pool)
    add("G_both_overlap_semantic", "G", weight=0.3, original_pool="overlap_semantic", detail_pool="overlap_semantic")

    for gate, threshold, label in (("area_fraction", 0.02, "small"), ("patch_count", 35, "patch_count"),
                                   ("short_side", 4, "short_side"), ("coherence", 0.65, "coherence"),
                                   ("combined", None, "combined")):
        add(f"I_{label}", "I", weight=0.3, gate=gate, gate_threshold=threshold)
    return configs
