"""Untrained, explicitly provisional standalone tickets for equation research.

These fixed heuristics are NOT the fitted equation models, are not calibrated
probabilities, and never enter the CEO, risk portfolio or purchase executor.
Only canonical pre-close captures may call this code.
"""
import math
from itertools import permutations

from equation_models import METHODS, joint_features, predict, rider_features, softmax


def provisional_distribution(method, riders, quotes):
    if method not in METHODS:
        raise ValueError('unknown standalone method')
    features = rider_features(riders)
    cars = sorted(features)
    combinations = list(permutations(cars, 3))
    expected = {'-'.join(map(str, t)) for t in combinations}
    if set(quotes) != expected or any(
            not isinstance(v, (int, float)) or not math.isfinite(v) or
            not 1 <= v < 9999.9 for v in quotes.values()):
        raise ValueError('a complete uncapped pre-close market is required')

    if method == 'pairwise_order':
        # Fixed, untrained illustration of pairwise ordering. Never labelled a fit.
        baseline = {'beta': [2.2, .6, .5, .4, .25, -.8, 0, 0]}
        return predict(method, baseline, riders, quotes)

    scores = {}
    by_car = {int(r['car_no']): r for r in riders}
    for triple in combinations:
        a, b, c = triple
        key = '-'.join(map(str, triple))
        f = joint_features(riders, triple, features)
        if method == 'market_residual':
            # Market-implied reference with a predeclared, untrained score tilt.
            scores[key] = -math.log(quotes[key]) + .20 * (
                f[1] + .65 * f[7] + .4 * f[13] + .2 * f[18])
        else:
            # Pre-race line/pace PROXY. No observed start/bell/back order exists.
            verified = (by_car[a].get('line_verification_status') == 'verified'
                        and by_car[b].get('line_verification_status') == 'verified'
                        and by_car[a].get('line_id') is not None
                        and by_car[a].get('line_id') == by_car[b].get('line_id'))
            scores[key] = (1.7 * features[a][2] + .9 * features[a][1]
                           + 1.1 * features[b][1] + .7 * features[c][3]
                           - .65 * features[a][5] + (.85 if verified else 0))
    return softmax(scores)


def standalone_tickets(distribution, quotes, per_group=12):
    """Independent top picks per formula and per odds band; no EV gate or risk quota."""
    if set(distribution) != set(quotes) or abs(sum(distribution.values()) - 1) > 1e-8:
        raise ValueError('invalid standalone distribution')
    if not 1 <= per_group <= 12:
        raise ValueError('invalid ticket count')
    result = {'本線': [], '穴': []}
    for buy, prob in sorted(distribution.items(), key=lambda item: (-item[1], item[0])):
        if not math.isfinite(prob) or prob <= 0:
            raise ValueError('invalid standalone probability')
        odds = quotes[buy]
        group = '穴' if odds >= 100 else '本線'
        if len(result[group]) < per_group:
            result[group].append({
                'buy': buy, 'prob': prob, 'odds': odds,
                'group': group, 'stake_yen': 100,
            })
    return result
