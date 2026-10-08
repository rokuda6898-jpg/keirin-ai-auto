"""Three distinct experimental probability models. No production authority.

Training accepts only samples already screened by equation_lab. No target-day
labels, fictitious trajectory observations, or hand-selected winning weights.
"""
import math
from collections import Counter, defaultdict
from itertools import permutations

METHODS = ('market_residual', 'pairwise_order', 'state_paths')
LABELS = {'market_residual': '市場の見落としを学ぶ式',
          'pairwise_order': '先着関係から組む式', 'state_paths': '途中隊列の遷移式'}
CONFIG = {'minimum_training_races': 50, 'minimum_training_days': 7,
          'training_lookback_days': 180, 'maximum_training_races': 2000,
          'iterations': 160, 'learning_rate': .10, 'ridge': .03,
          'transition_prior_mass': 5.0, 'feature_version': 'bounded_preclose_v1'}


def number(value, default=0.0):
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (ValueError, TypeError):
        return default


def softmax(values):
    if not values or any(not math.isfinite(v) for v in values.values()):
        raise ValueError('invalid probability scores')
    high = max(values.values())
    weights = {k: math.exp(max(-700, v - high)) for k, v in values.items()}
    mass = sum(weights.values())
    return {k: v / mass for k, v in weights.items()}


def rider_features(rows):
    """Only explicit pre-race fields; missing measurements remain flagged."""
    result = {}
    scores = [number(r.get('score')) for r in rows]
    center = sum(scores) / len(scores)
    for r in rows:
        p = min(1, max(1e-8, number(r.get('p_win'), 1 / len(rows))))
        result[int(r['car_no'])] = [
            math.log(p) / 10,
            max(-3, min(3, (number(r.get('score'), center) - center) / 10)),
            min(1, max(0, number(r.get('back_count')) / 20)),
            min(1, max(0, number(r.get('place2_rate')))),
            min(1, max(0, number(r.get('place3_rate')))),
            min(1, max(0, number(r.get('recent_avg_finish'), 5) / 9)),
            float(r.get('score') is None), float(r.get('recent_avg_finish') is None)]
    return result


def joint_features(rows, triple, features=None):
    features = features or rider_features(rows)
    by_car = {int(r['car_no']): r for r in rows}
    a, b, c = triple
    def same(i, j):
        left, right = by_car[i], by_car[j]
        verified = all(r.get('line_verification_status') == 'verified'
                       and r.get('line_id') is not None for r in (left, right))
        return float(verified and left['line_id'] == right['line_id'])
    # Fixed low-capacity specification: individual, pair, and triple effects.
    ab, bc, ac = same(a, b), same(b, c), same(a, c)
    pressure = min(1, sum(f[2] for f in features.values()) / 2)
    return [*features[a][:6], *features[b][:6], *features[c][:6],
            ab, bc, ac, ab * bc, ab * pressure, bc * pressure]


def eligible_training(samples):
    return (len(samples) >= CONFIG['minimum_training_races'] and
            len({s['date'] for s in samples}) >= CONFIG['minimum_training_days'])


def fit(samples, paths=()):
    status = {'training_races': len(samples), 'training_days': len({s['date'] for s in samples}),
              'config': dict(CONFIG), 'models': {m: None for m in METHODS},
              'reasons': {m: 'insufficient_prior_training' for m in METHODS}}
    if not eligible_training(samples):
        return status
    import numpy as np
    # Each race has equal mass, independently of field size or ticket count.
    designs, offsets, targets = [], [], []
    pair_x, pair_y, pair_weights = [], [], []
    for sample in samples:
        rows, actual = sample['riders'], tuple(map(int, sample['actual'].split('-')))
        fs = rider_features(rows)
        cars = sorted(fs)
        combos = list(permutations(cars, 3))
        designs.append(np.asarray([joint_features(rows, t, fs) for t in combos]))
        offsets.append(np.asarray([-math.log(sample['quotes']['-'.join(map(str, t))]) for t in combos]))
        targets.append(combos.index(actual))
        ranks = {car: pos for pos, car in enumerate(actual)}
        local_x, local_y = [], []
        for i, a in enumerate(cars):
            for b in cars[i + 1:]:
                if a not in ranks and b not in ranks:
                    continue  # Do not invent the order of finishers below third.
                local_x.append([u - v for u, v in zip(fs[a], fs[b])])
                local_y.append(float(ranks.get(a, 3) < ranks.get(b, 3)))
        pair_x.extend(local_x); pair_y.extend(local_y)
        pair_weights.extend([1 / len(local_x)] * len(local_x))
    beta = np.zeros(designs[0].shape[1])
    for _ in range(CONFIG['iterations']):
        grad = np.zeros_like(beta)
        for X, offset, target in zip(designs, offsets, targets):
            logits = X @ beta + offset
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()
            grad += X.T @ probs - X[target]
        beta -= CONFIG['learning_rate'] * (grad / len(designs) + CONFIG['ridge'] * beta)
    X, y, weight = np.asarray(pair_x), np.asarray(pair_y), np.asarray(pair_weights)
    weight /= weight.sum()
    pair_beta = np.zeros(X.shape[1])
    for _ in range(CONFIG['iterations']):
        q = 1 / (1 + np.exp(-np.clip(X @ pair_beta, -30, 30)))
        grad = X.T @ ((q - y) * weight) + CONFIG['ridge'] * pair_beta
        pair_beta -= CONFIG['learning_rate'] * grad
    status['models']['market_residual'] = {'beta': beta.tolist()}
    status['models']['pairwise_order'] = {'beta': pair_beta.tolist()}
    status['reasons'].update(market_residual='trained_shadow', pairwise_order='trained_shadow')
    status['models']['state_paths'] = fit_paths(samples, paths)
    status['reasons']['state_paths'] = ('trained_shadow' if status['models']['state_paths']
                                       else 'observed_intermediate_orders_required')
    return status


def rank_order(rows):
    return sorted((int(r['car_no']) for r in rows),
                  key=lambda c: (-next(number(r.get('p_win')) for r in rows if int(r['car_no']) == c), c))


def fit_paths(samples, paths):
    """Coarse top-three queue transitions, not a microscopic energy simulator.

States are pre-race strength ranks, never persistent car-number identities.
Every training trajectory needs four actual observed stages and official order.
"""
    sample_map = {s['race_id']: s for s in samples}
    valid = defaultdict(list)
    seen = set()
    for path in paths:
        sample = sample_map.get(path.get('race_id'))
        if not sample or path['race_id'] in seen:
            continue
        seen.add(path['race_id'])
        orders = path.get('orders', {})
        ranks = {c: i + 1 for i, c in enumerate(rank_order(sample['riders']))}
        stages = [orders.get(k) for k in ('start', 'bell', 'back', 'finish')]
        if any(not isinstance(s, list) or len(s) != 3 or len(set(s)) != 3
               or not set(s) <= set(ranks) for s in stages):
            continue
        if '-'.join(map(str, stages[-1])) != sample['actual']:
            continue
        states = ['-'.join(str(ranks[c]) for c in order) for order in stages]
        valid[len(ranks)].append((sample['date'], states))
    groups = {}
    for size, rows in valid.items():
        if len(rows) < CONFIG['minimum_training_races'] or len({d for d, _ in rows}) < CONFIG['minimum_training_days']:
            continue
        initials = Counter(s[0] for _, s in rows)
        transitions, marginals = [], []
        for stage in range(3):
            matrix = defaultdict(Counter)
            for _, s in rows:
                matrix[s[stage]][s[stage + 1]] += 1
            transitions.append({a: dict(v) for a, v in matrix.items()})
            marginals.append(dict(Counter(s[stage + 1] for _, s in rows)))
        groups[str(size)] = {'initial': dict(initials), 'transitions': transitions,
                            'marginals': marginals, 'observed_races': len(rows)}
    return {'groups': groups} if groups else None


def predict(method, model, rows, quotes):
    if model is None:
        return None
    fs = rider_features(rows)
    cars = sorted(fs)
    combos = list(permutations(cars, 3))
    expected = {'-'.join(map(str, t)) for t in combos}
    if set(quotes) != expected or any(not math.isfinite(p) or not 1 <= p < 9999.9 for p in quotes.values()):
        raise ValueError('complete uncapped single-snapshot prices required')
    if method == 'market_residual':
        return softmax({'-'.join(map(str, t)): -math.log(quotes['-'.join(map(str, t))]) +
            sum(a * b for a, b in zip(model['beta'], joint_features(rows, t, fs))) for t in combos})
    if method == 'pairwise_order':
        log_q = {}
        for a, b in permutations(cars, 2):
            logit = sum(w * (u - v) for w, u, v in zip(model['beta'], fs[a], fs[b]))
            log_q[a, b] = -max(0, -logit) - math.log1p(math.exp(-abs(logit)))
        scores = {}
        for a, b, c in combos:
            value = log_q[a, b] + log_q[a, c] + log_q[b, c]
            value += sum(log_q[w, other] for other in cars if other not in (a, b, c) for w in (a, b, c))
            scores[f'{a}-{b}-{c}'] = value
        return softmax(scores)
    if method != 'state_paths':
        raise ValueError('unknown equation')
    group = model['groups'].get(str(len(cars)))
    if group is None:
        return None
    states = ['-'.join(map(str, t)) for t in permutations(range(1, len(cars) + 1), 3)]
    def prior(counts):
        total = sum(counts.values()) + 1
        return {s: (counts.get(s, 0) + 1 / len(states)) / total for s in states}
    distribution = prior(group['initial'])
    alpha = CONFIG['transition_prior_mass']
    for matrix, marginal in zip(group['transitions'], group['marginals']):
        base = prior(marginal)
        next_dist = {s: 0.0 for s in states}
        for state, mass in distribution.items():
            counts = matrix.get(state, {})
            total = sum(counts.values()) + alpha
            for target in states:
                next_dist[target] += mass * (counts.get(target, 0) + alpha * base[target]) / total
        distribution = next_dist
    ranking = rank_order(rows)
    return {'-'.join(str(ranking[int(k) - 1]) for k in state.split('-')): p for state, p in distribution.items()}
