"""Predeclared validation-only selection and final roles."""

from collections import defaultdict

from refinement_policies import combine_candidates as combine, semantic_signature


CONTROLS = ('baseline', 'fixed_detail', 'size_only', 'boundary_only', 'current_best',
            'B_size_select', 'B_boundary_select', 'B_size_boundary_select', 'B_semantic_only', 'B_best_large_fallback')


def overall(data):
    return sorted((r for r in data['summary'] if r['area_group'] == 'all'), key=lambda r: (-r['CIDEr'], r['mode']))


def choose_selective(data):
    return next(row['mode'] for row in overall(data)
                if data['metadata']['configs'][row['mode']]['family'] in ('A', 'B', 'C')
                and 0 < row['detail_activation_rate'] < 0.8)


def validation_candidates(screen, limit):
    all_configs = screen['metadata']['configs']
    picked = [all_configs[name] for name in CONTROLS]
    picked.extend(config for name, config in sorted(all_configs.items()) if name.startswith('C_budget_'))
    families = defaultdict(list)
    for row in overall(screen):
        config = all_configs[row['mode']]
        if config['family'] in 'ACDEF':
            families[config['family']].append(config)
    for family in 'ACDEF':
        picked.extend(families[family][:limit])
    return list({c['id']: c for c in picked}.values())


def combine_candidates(validation, screening):
    configs = validation['metadata']['configs']
    ranked = [configs[row['mode']] for row in overall(validation)]
    sources = {key: configs[key] for key in ('current_best', 'B_best_large_fallback', 'B_size_boundary_select')}
    sources.update(selective=configs[choose_selective(validation)],
                   boundary=next(c for c in ranked if c['family'] == 'D'),
                   geometry=next(c for c in ranked if c['family'] == 'E'))
    combinations = combine(sources)
    rank_candidates = [c for c in ranked if c['family'] == 'F']
    best_quality = next(c for c in ranked if c['family'] != 'F')
    if rank_candidates and best_quality['id'] not in ('baseline', 'fixed_detail', 'current_best'):
        combinations.append({**rank_candidates[0], 'id': 'G_quality_rerank', 'family': 'G',
                             'sources': [configs['baseline'], configs['fixed_detail'], configs['current_best'], best_quality]})
    seen = {semantic_signature(c) for c in configs.values()}
    return [config for config in combinations if semantic_signature(config) not in seen]


def online_shortlist(data):
    configs = data['metadata']['configs']
    ranked = overall(data)
    names = ['baseline', 'fixed_detail', 'current_best', 'B_best_large_fallback']
    names.extend(r['mode'] for r in ranked[:8])
    names.extend(r['mode'] for r in ranked if r['mode'].startswith('C_budget_'))
    names.extend(r['mode'] for r in [r for r in ranked if 0 < r['detail_activation_rate'] <= 0.5][:4])
    names.extend(r['mode'] for r in ranked if 'rank_policy' in configs[r['mode']])
    return [configs[name] for name in dict.fromkeys(names)]


def final_roles(validation, online, extra_target=0.2):
    configs = validation['metadata']['configs']
    score = {r['mode']: r['CIDEr'] for r in overall(validation)}
    large = {r['mode']: r['CIDEr'] for r in validation['summary'] if r['area_group'] == 'large'}
    latency = {r['mode']: r['mean_inference_ms'] for r in online['summary'] if r['area_group'] == 'all'}
    safe = [name for name in latency if name not in ('baseline', 'fixed_detail', 'current_best') and score[name] > score['baseline']]
    if len(safe) < 3:
        raise ValueError('No three distinct positive-gain validation candidates with online measurements')
    quality = max(safe, key=lambda name: score[name])
    remaining = [name for name in safe if name != quality]
    eligible = [name for name in remaining if latency[name] <= latency['baseline'] * (1 + extra_target)]
    guarded = [name for name in eligible if large[name] >= large['baseline']]
    if guarded or eligible:
        balanced = max(guarded or eligible, key=lambda name: score[name])
        rule = 'highest CIDEr within latency target, preferring no large-object loss'
    else:
        balanced = min(remaining, key=lambda name: ((latency[name] / latency['baseline'] - 1) /
                                                    (score[name] - score['baseline']), -score[name]))
        rule = 'lowest relative latency increase per CIDEr gain, because no candidate met the latency target'
    efficient = min((name for name in remaining if name != balanced), key=lambda name: (latency[name], -score[name]))
    roles = {'Quality': quality, 'Balanced': balanced, 'Efficient': efficient}
    names = ['baseline', 'fixed_detail', 'current_best', *roles.values(), 'size_only', 'boundary_only', 'B_best_large_fallback']
    details = {'balanced_rule': rule, 'validation_target_max_extra_latency': extra_target,
               'validation_measurements': {name: {'CIDEr': score[name], 'large_CIDEr': large[name], 'online_ms': latency[name]}
                                           for name in ('baseline', *roles.values())}}
    return roles, [configs[name] for name in dict.fromkeys(names)], details
