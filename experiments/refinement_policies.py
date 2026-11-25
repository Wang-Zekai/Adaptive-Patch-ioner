"""Frozen feature fusion and pre-crop decisions for Patch-ioner refinement."""

from copy import deepcopy
import json

import torch

from exploration_methods import size_bins


CONFIDENCE_SIGNALS = {
    'mean_token_logprob', 'min_token_logprob', 'token_entropy',
    'memory_max_cosine', 'memory_margin', 'memory_entropy',
}

CORE_SETTINGS = {
    'detail_scale': 1.2,
    'detail_resolution': 518,
    'detail_pool': {'method': 'overlap'},
    'original_pool': {'method': 'official'},
}


def pool(method, **options):
    return {'method': method, **options}


def baseline(identifier='baseline', family='B'):
    return {'id': identifier, 'family': family, 'weight': 0, 'history_id': 'baseline'}


def needs_risk(config):
    return bool(config.get('gate'))


def uses_confidence(config):
    return any(name in CONFIDENCE_SIGNALS for name, _ in config.get('gate', {}).get('components', []))


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


def calibrate(config, signals):
    """Freeze a fixed or budget gate using screen geometry and feature signals only."""
    gate = config.get('gate')
    if not gate or 'threshold' in gate:
        return config
    config = deepcopy(config)
    gate = config['gate']
    ranges = []
    for name, _ in gate['components']:
        values = signal(signals, name).float()
        if not torch.isfinite(values).all():
            raise ValueError(f'Nonfinite calibration signal {name}')
        if len(gate['components']) == 1:
            low, high = values.min().item(), values.max().item()
        else:
            low, high = (torch.quantile(values, q).item() for q in (0.10, 0.90))
        ranges.append([low, high])
    gate['ranges'] = ranges
    risk = _risk(gate, signals).float()
    gate['threshold'] = torch.quantile(risk, 1 - gate['budget']).item()
    return config


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


def _budget(components, proportion):
    return {'components': [list(x) for x in components], 'budget': proportion}


def _below(name, threshold):
    return {'fixed_signal': name, 'threshold': threshold}


def candidate_grid(best=None):
    """One-factor screen and complete ablation, with named historical controls."""
    weighted = {'weight_rule': 'piecewise', 'size_signal': 'mixed_support',
                'limits': [0.24, 0.58], 'weights': [0.3, 0.8, 0.5]}
    if best is not None:
        if best.get('detail_pool') != 'overlap_semantic' or best.get('weight_rule') != 'piecewise':
            raise ValueError('The locked control must be the earlier piecewise overlap-semantic policy')
        weighted = {key: best[key] for key in weighted}
    weighted = {**weighted, 'detail_scale': best.get('detail_scale', 1.2) if best else 1.2,
                'detail_resolution': best.get('detail_resolution', 518) if best else 518}
    boundary = pool('overlap_semantic')
    combo = {**weighted, 'detail_pool': boundary}
    candidates = []

    def add(identifier, family, base=None, **options):
        candidates.append({'id': identifier, 'family': family, **(base or {}), **options})

    add('baseline', 'B', weight=0, history_id='baseline')
    add('fixed_detail', 'B', weight=0.3, history_id='A_w0p3')
    add('size_only', 'B', weighted, history_id='C_piecewise_mixed_support')
    add('boundary_only', 'B', weight=0.3, detail_pool=boundary, history_id='G_detail_overlap_semantic')
    add('current_best', 'B', combo, history_id='J_dynamic_boundary')
    common_gate = _budget([('patch_count', 'low')], 0.50)
    add('B_size_select', 'B', weighted, gate=common_gate)
    add('B_boundary_select', 'B', weight=0.3, detail_pool=boundary, gate=common_gate)
    add('B_size_boundary_select', 'B', combo, gate=common_gate)
    add('B_best_large_fallback', 'B', combo, gate=_below('area_fraction', 0.10))
    add('B_semantic_only', 'B', weighted, detail_pool=pool('semantic'))

    for name in ('area_fraction', 'patch_count', 'short_side', 'aspect_ratio', 'mixed_support'):
        for proportion in (0.50, 0.70, 0.85):
            if name == 'patch_count' and proportion == 0.50:
                continue  # Already measured by B_size_boundary_select with the identical gate.
            direction = 'high' if name == 'aspect_ratio' else 'low'
            add(f'A_gate_{name}_{int(proportion * 100)}', 'A', combo, gate=_budget([(name, direction)], proportion))
    add('A_soft_fade', 'A', combo, fade=[0.07, 0.20], detail_pool=pool('dynamic', dynamic_rule='large_fade'))
    add('A_soft_weight_only', 'A', combo, fade=[0.07, 0.20])
    add('A_large_boundary_original', 'A', combo, large_weight=0, large_original_pool=boundary)
    add('A_large_detail_unweighted', 'A', combo, large_weight=0.12, large_detail_pool=pool('overlap'))

    for name in ('mean_cosine', 'variance_cosine', 'dispersion', 'aggregation_change'):
        direction = 'low' if name == 'mean_cosine' else 'high'
        add(f'C_visual_{name}', 'C', combo, gate=_budget([(name, direction)], 0.40))
    for name in ('mean_token_logprob', 'min_token_logprob', 'token_entropy', 'memory_max_cosine', 'memory_margin', 'memory_entropy'):
        direction = 'high' if 'entropy' in name else 'low'
        add(f'C_conf_{name}', 'C', combo, gate=_budget([(name, direction)], 0.40))
    joint = [('mixed_support', 'low'), ('mean_cosine', 'low'), ('variance_cosine', 'high')]
    add('C_joint_with_confidence', 'C', combo, gate=_budget([*joint, ('token_entropy', 'high'), ('memory_margin', 'low')], 0.40))
    for proportion in (0.20, 0.30, 0.40, 0.50, 0.70):
        add(f'C_budget_{int(proportion * 100)}', 'C', combo, gate=_budget(joint, proportion))

    for key, values in (('area_power', (0.5, 2.0)), ('semantic_power', (0.65, 1.6)), ('temperature', (0.10, 0.40))):
        for value in values:
            suffix = str(value).replace('.', 'p')
            add(f'D_{key}_{suffix}', 'D', combo, detail_pool=pool('overlap_semantic', **{key: value}))
    for method in ('boundary', 'suppress', 'dynamic'):
        add(f'D_{method}', 'D', combo, detail_pool=pool(method))
    add('D_original_only', 'D', weighted, original_pool=boundary)
    add('D_both', 'D', combo, original_pool=boundary)
    add('D_asymmetric', 'D', combo, original_pool=pool('residual', residual=0.25))
    add('D_detail_residual', 'D', combo, detail_pool=pool('residual', residual=0.50))

    for scale in (1.0, 1.05, 1.1, 1.3, 1.4):
        add(f'E_crop_{str(scale).replace(".", "p")}', 'E', combo, detail_scale=scale)
    add('E_area_scales', 'E', combo, detail_scale_bins={'signal': 'area_fraction', 'values': [1.05, 1.2, 1.4]})
    add('E_narrow_scales', 'E', combo, detail_scale_bins={'signal': 'short_side', 'values': [1.05, 1.2, 1.4]})
    add('E_letterbox_narrow', 'E', combo, preserve_aspect_above=2.0)
    add('E_letterbox_all', 'E', combo, preserve_aspect_above=1.0)
    add('E_area_resolution', 'E', combo, detail_resolution_bins={'signal': 'area_fraction', 'values': [672, 336, 224]})
    add('E_support_resolution', 'E', combo, detail_resolution_bins={'signal': 'mixed_support', 'values': [518, 336, 224]})
    add('E_narrow_resolution', 'E', combo, detail_resolution_bins={'signal': 'short_side', 'values': [672, 518, 336]})
    add('E_res336', 'E', combo, detail_resolution=336)
    return candidates


def semantic_signature(config):
    """Configuration identity for exact-result reuse across renamed controls."""
    excluded = {'id', 'family', 'history_id', 'notes'}
    semantics = {key: value for key, value in config.items() if key not in excluded}
    return json.dumps({**CORE_SETTINGS, 'weight_rule': 'fixed', 'weight': 0.3, **semantics},
                      sort_keys=True, separators=(',', ':'))


def combine_candidates(sources):
    """Generate compact Stage 3 combinations from validation winners, before test lock."""
    base = sources['current_best']
    fallback = sources['B_best_large_fallback']
    selective = sources.get('selective') or sources['B_size_boundary_select']
    boundary = sources.get('boundary') or base
    geometry = sources.get('geometry') or base
    entries = []

    def build(identifier, *parts):
        cfg = {**base}
        for source, keys in parts:
            cfg.update({key: source[key] for key in keys if key in source})
        cfg.update(id=identifier, family='G')
        cfg.pop('history_id', None)
        entries.append(cfg)

    pool_keys = ('detail_pool', 'original_pool', 'large_original_pool', 'large_detail_pool')
    crop_keys = ('detail_scale', 'detail_scale_bins', 'detail_resolution', 'detail_resolution_bins', 'preserve_aspect_above')
    build('G_fallback_boundary', (fallback, ('gate',)), (boundary, pool_keys))
    build('G_selective_crop', (selective, ('gate',)), (geometry, crop_keys))
    build('G_selective_boundary', (selective, ('gate',)), (boundary, pool_keys))
    build('G_geometry_boundary', (geometry, crop_keys), (boundary, pool_keys))
    build('G_fallback_geometry', (fallback, ('gate',)), (geometry, crop_keys))
    return entries
