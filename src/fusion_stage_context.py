"""Research-only stage and race-context pooling of frozen joint forecasts.

Each stage is a normalized conditional distribution. Inference enumerates
every possible prefix; it never receives the actual finishing order.
"""
from functools import lru_cache
from itertools import permutations

import numpy as np

import fusion_equations as fusion

VERSION = 'stage_context_pool_v1'
CONFIG = {'minimum_group_races': 100, 'minimum_group_days': 14,
          'group_shrinkage_races': 300, 'blend_grid': [0., .25, .5, .75, 1.]}
VARIANTS = ('context_joint', 'stage_global', 'stage_context')


@lru_cache(maxsize=32)
def prefix_index(keys):
    triples = [tuple(map(int, k.split('-'))) for k in keys]
    cars = sorted({c for t in triples for c in t})
    if len(triples) != len(set(triples)) or set(triples) != set(permutations(cars, 3)):
        raise ValueError('complete distinct-car ordered triples required')
    firsts = {c: i for i, c in enumerate(cars)}
    pairs = {p: i for i, p in enumerate(permutations(cars, 2))}
    return (np.array([firsts[t[0]] for t in triples]),
            np.array([pairs[t[:2]] for t in triples]), len(cars), len(pairs))


def decompose(keys, matrix):
    """Return expert × stage × triple probabilities P(a), P(b|a), P(c|ab)."""
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[1] != len(keys) or
            not np.isfinite(matrix).all() or (matrix <= 0).any() or
            not np.allclose(matrix.sum(axis=1), 1., rtol=0, atol=1e-8)):
        raise ValueError('strictly positive full normalized expert distributions required')
    a, ab, na, nab = prefix_index(tuple(keys))
    first = np.array([np.bincount(a, weights=p, minlength=na)[a] for p in matrix])
    pair = np.array([np.bincount(ab, weights=p, minlength=nab)[ab] for p in matrix])
    return np.stack((first, pair / first, matrix / pair), axis=1)


def fit_weights(probabilities, prior=None):
    """Same optimizer and penalties as the existing complete-data global pool."""
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or len(p) == 0 or not np.isfinite(p).all() or (p <= 0).any() or (p > 1).any():
        raise ValueError('invalid training probabilities')
    prior = np.full(p.shape[1], 1 / p.shape[1]) if prior is None else np.asarray(prior, dtype=float)
    if prior.shape != (p.shape[1],) or not np.isfinite(prior).all() or (prior <= 0).any():
        raise ValueError('invalid prior weights')
    prior = prior / prior.sum()
    errors = -np.log(p)
    errors -= errors.mean(axis=0)
    errors /= np.maximum(np.linalg.norm(errors, axis=0), 1e-12)
    correlation = errors.T @ errors
    logits = np.log(prior)
    for _ in range(fusion.CONFIG['pool_iterations']):
        w = np.exp(logits - logits.max()); w /= w.sum()
        grad = (1 - p / (p @ w)[:, None]).mean(axis=0)
        grad += fusion.CONFIG['pool_redundancy'] * (correlation @ w)
        grad += fusion.CONFIG['pool_prior_penalty'] * (np.log(w / prior) + 1)
        logits -= .2 * w * (grad - float(w @ grad))
    w = np.exp(logits - logits.max())
    return w / w.sum()


def context_key(field, race_class):
    # Only race metadata known before the race; no result-dependent routing.
    return f'{int(field)}|{str(race_class)}'


def fit(true_joint, true_stages, contexts, dates, pool, cutoff):
    dates = np.asarray(dates, dtype=str)
    contexts = np.asarray(contexts, dtype=str)
    joint = np.asarray(true_joint, dtype=float)
    stages = np.asarray(true_stages, dtype=float)
    names = sorted(pool['weights'])
    if len(dates) == 0 or (dates >= cutoff).any():
        raise ValueError('fitting must use only dates before the held-out year')
    if (joint.shape != (len(dates), len(names)) or stages.shape != (len(dates), len(names), 3)
            or len(contexts) != len(dates)):
        raise ValueError('training shapes do not match the frozen expert schema')
    global_joint = np.array([pool['weights'][n] for n in names])
    # Verify that this is precisely the old stacking period, not a different fit.
    if not np.allclose(fit_weights(joint), global_joint, rtol=1e-10, atol=1e-12):
        raise ValueError('global mixture cannot be reproduced from this training period')
    global_stage = np.array([fit_weights(stages[:, :, s]) for s in range(3)])
    groups = {}
    for key in sorted(set(contexts)):
        ix = contexts == key
        count, days = int(ix.sum()), int(len(set(dates[ix])))
        if count < CONFIG['minimum_group_races'] or days < CONFIG['minimum_group_days']:
            continue
        local_joint = fit_weights(joint[ix], global_joint)
        local_stage = np.array([fit_weights(stages[ix, :, s], global_stage[s]) for s in range(3)])
        strength = count / (count + CONFIG['group_shrinkage_races'])
        groups[key] = {'races': count, 'days': days, 'local_fraction': strength,
                      'joint': ((1 - strength) * global_joint + strength * local_joint).tolist(),
                      'stages': ((1 - strength) * global_stage + strength * local_stage).tolist()}
    return {'version': VERSION, 'config': dict(CONFIG), 'names': names, 'aliases': pool['aliases'],
            'global_joint': global_joint.tolist(), 'global_stages': global_stage.tolist(),
            'groups': groups, 'fit_first': min(dates), 'fit_last': max(dates),
            'fit_races': len(dates), 'cutoff_exclusive': cutoff}


def expert_matrix(pool, available):
    groups = fusion.cluster_values(pool, available)
    if set(groups) != set(pool['weights']):
        raise ValueError('missing frozen expert; refusing to invent its distribution')
    names = sorted(pool['weights'])
    keys = sorted(groups[names[0]], key=lambda t: tuple(map(int, t.split('-'))))
    if any(set(p) != set(keys) for p in groups.values()):
        raise ValueError('expert fields differ')
    matrix = np.array([[groups[n][k] for k in keys] for n in names])
    decompose(keys, matrix)
    return keys, matrix


def raw_predictions(model, keys, matrix, context):
    parts = decompose(keys, matrix)
    if len(matrix) != len(model['names']):
        raise ValueError('expert schema mismatch')
    group = model['groups'].get(context)
    joint_weights = group['joint'] if group else model['global_joint']
    stage_weights = group['stages'] if group else model['global_stages']
    result = {'context_joint': np.asarray(joint_weights) @ matrix}
    for name, weights in (('stage_global', model['global_stages']), ('stage_context', stage_weights)):
        result[name] = np.prod(np.einsum('se,est->st', np.asarray(weights), parts), axis=0)
    for p in result.values():
        if not np.isfinite(p).all() or (p <= 0).any() or abs(p.sum() - 1) > 1e-8:
            raise ValueError('conditional chain did not normalize')
    return result


def temper(p, value):
    z = np.log(p) / value
    v = np.exp(z - z.max())
    return v / v.sum()


def calibrate(records, model):
    """Select blend and temperature in the separate, older calibration phase."""
    grid = [(a, t) for a in CONFIG['blend_grid'] for t in fusion.CONFIG['temperatures']]
    losses = {name: np.zeros(len(grid)) for name in VARIANTS}
    baseline_loss = 0.; dates = []; seen = set()
    for record in records:
        date = record['date']
        if not model['fit_last'] < date < model['cutoff_exclusive']:
            raise ValueError('calibration must be disjoint, later than fitting, and before holdout')
        if record['race_id'] in seen:
            raise ValueError('duplicate calibration race')
        seen.add(record['race_id']); dates.append(date)
        actual, baseline = record['actual_index'], np.asarray(record['baseline'])
        baseline_loss -= np.log(baseline[actual])
        if set(record['raw']) != set(VARIANTS):
            raise ValueError('calibration variants missing')
        for name, p in record['raw'].items():
            if name not in losses:
                raise ValueError('unexpected variant')
            for i, (alpha, tau) in enumerate(grid):
                mixed = (1 - alpha) * baseline + alpha * temper(p, tau)
                losses[name][i] -= np.log(mixed[actual])
    if not dates:
        raise ValueError('no calibration races')
    result = {}
    for name, values in losses.items():
        best = min(range(len(grid)), key=lambda i: (values[i], grid[i][0], abs(grid[i][1] - 1)))
        alpha, tau = grid[best]
        result[name] = {'blend': alpha, 'temperature': tau, 'nll': float(values[best] / len(dates)),
                        'grid_losses': [{'blend': a, 'temperature': t, 'nll': float(v / len(dates))}
                                        for (a, t), v in zip(grid, values)]}
    return {'variants': result, 'races': len(dates), 'first': min(dates), 'last': max(dates),
            'baseline_nll': float(baseline_loss / len(dates)), 'selection_metric': 'nll_on_older_calibration'}


def predict(model, keys, matrix, context, baseline):
    raw = raw_predictions(model, keys, matrix, context)
    result = {}
    for name, p in raw.items():
        cfg = model['calibration']['variants'][name]
        result[name] = (1 - cfg['blend']) * np.asarray(baseline) + cfg['blend'] * temper(p, cfg['temperature'])
    return result
