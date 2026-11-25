"""Frozen size-aware fusion and decisions made before crop encoding."""

import torch


SIZE_BINS = {
    "patch_count": (35, 150),
    "short_side": (4, 10),
    "area_fraction": (0.02, 0.10),
    "mixed_support": (0.24, 0.58),
}


CORE_SETTINGS = {
    'detail_scale': 1.2,
    'detail_resolution': 518,
    'detail_pool': {'method': 'overlap'},
    'original_pool': {'method': 'official'},
}


def signal(signals, name):
    if name == 'mixed_support':
        count = (signals['patch_count'] / 150).clamp(0, 1)
        narrow = (signals['short_side'] / 10).clamp(0, 1)
        return (count * narrow).sqrt()
    return signals[name]


def _risk(gate, signals):
    if 'fixed_signal' in gate:
        return signal(signals, gate['fixed_signal'])
    components = []
    for (name, direction), (low, high) in zip(gate['components'], gate['ranges'], strict=True):
        data = signal(signals, name)
        scaled = ((data - low) / max(high - low, 1e-8)).clamp(0, 1)
        components.append(scaled if direction == 'high' else 1 - scaled)
    return torch.stack(components).mean(dim=0)


def size_bins(signals, name, limits=None):
    value = signal(signals, name)
    lower, upper = SIZE_BINS[name] if limits is None else limits
    return (value >= lower).long() + (value >= upper).long()


def detail_weight(config, signals):
    shape = signals['area_fraction']
    if config.get('weight_rule', 'fixed') == 'piecewise':
        bins = size_bins(signals, config.get('size_signal', 'mixed_support'), config.get('limits'))
        weights = shape.new_tensor(config['weights'])[bins]
    else:
        weights = torch.full_like(shape, config.get('weight', 0.3))
    if 'large_weight' in config:
        weights = torch.where(shape >= 0.10, config['large_weight'], weights)
    if 'fade' in config:
        low, high = config['fade']
        weights *= ((high - shape) / (high - low)).clamp(0, 1)
    return weights


def gate_mask(config, signals):
    """Return rows eligible for a crop before calling the visual encoder."""
    active = detail_weight(config, signals) > 0
    gate = config.get('gate')
    if gate and active.any():
        values = _risk(gate, signals)
        if 'fixed_signal' in gate:
            return active & (values < gate['threshold'])
        return active & (values >= gate['threshold'])
    return active


def _dynamic_values(config, signals, item, default):
    source = config.get(f'{item}_bins')
    if not source:
        return [config.get(item, default)] * len(signals['area_fraction'])
    bins = size_bins(signals, source['signal'], source.get('limits')).tolist()
    return [source['values'][b] for b in bins]


def original_specs(config, signals):
    upper = config.get('large_original_pool')
    original = config.get('original_pool', CORE_SETTINGS['original_pool'])
    return [upper if area >= 0.10 and upper else original for area in signals['area_fraction'].tolist()]


def detail_specs(config, signals):
    """Return crop geometry and pool settings for selected rows, and None otherwise."""
    active = gate_mask(config, signals).tolist()
    scales = _dynamic_values(config, signals, 'detail_scale', CORE_SETTINGS['detail_scale'])
    resolutions = _dynamic_values(config, signals, 'detail_resolution', CORE_SETTINGS['detail_resolution'])
    regular = config.get('detail_pool', CORE_SETTINGS['detail_pool'])
    large = config.get('large_detail_pool', regular)
    minimum_aspect = config.get('preserve_aspect_above', float('inf'))
    areas = signals['area_fraction'].tolist()
    ratios = signals.get('aspect_ratio', torch.ones_like(signals['area_fraction'])).tolist()
    return [(scale, resolution, ratio >= minimum_aspect, large if area >= 0.10 else regular) if enabled else None
            for enabled, scale, resolution, ratio, area in zip(active, scales, resolutions, ratios, areas, strict=True)]


def compose(config, original, detail, signals):
    """Apply the calibrated decision and interpolate the frozen region features."""
    signals = {key: value.to(device=original.device, dtype=original.dtype) for key, value in signals.items()}
    weights = detail_weight(config, signals).masked_fill(~gate_mask(config, signals), 0)
    if not weights.any():
        return original, weights
    if detail is None or detail.shape != original.shape:
        raise ValueError(f"Selected rows need detail features: {config['id']}")
    fused = original * (1 - weights[:, None]) + detail * weights[:, None]
    return torch.where((weights == 0)[:, None], original, fused), weights
