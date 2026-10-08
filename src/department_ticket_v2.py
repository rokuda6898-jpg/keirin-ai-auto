"""Explicit challenger engine; legacy risk selection is never routed here.

All factorial arms share scale-invariant gates and one policy. Market association
is used once; probabilities remain uncalibrated hypotheses, not certified EV.
"""
import math
from itertools import permutations


def normalize(weights):
    values = {int(k): max(1e-12, float(v)) for k, v in weights.items()}
    if not all(math.isfinite(v) for v in values.values()) or not values:
        raise ValueError('invalid position weights')
    mass = sum(values.values())
    return {k: v / mass for k, v in values.items()}


def pool(weights, size):
    """Relative 2% boundary, invariant to multiplying the score scale."""
    ordered = sorted(weights, key=lambda c: (-weights[c], c))
    cutoff = weights[ordered[min(size, len(ordered)) - 1]]
    tolerance = max(weights.values()) * .02
    return {c for c in ordered if weights[c] >= cutoff - tolerance}


def distributions(scores, market_joint=None, context=None):
    first = normalize({r['car_no']: r['first'] for r in scores})
    second_base = {r['car_no']: r['second'] for r in scores}
    third_base = {r['car_no']: r['third'] for r in scores}
    cars = sorted(first)
    second, third, joint = {}, {}, {}
    for a in cars:
        weights = {b: second_base[b] for b in cars if b != a}
        if market_joint:
            # One geometric blend with the market's second-place support for
            # this winner. Its normalization supplies P(second=b | first=a).
            weights = {b: math.sqrt(w * sum(p for (x, y, _), p in market_joint.items()
                                            if (x, y) == (a, b))) for b, w in weights.items()}
        if context:
            weights = {b: w * context(a, None, b, 2) for b, w in weights.items()}
        second[a] = normalize(weights)
        for b in second[a]:
            weights = {c: third_base[c] for c in cars if c not in (a, b)}
            if market_joint:
                # Likewise P(third=c | first=a, second=b); first-place
                # popularity is never used to filter second/third candidates.
                weights = {c: math.sqrt(w * market_joint[a, b, c]) for c, w in weights.items()}
            if context:
                weights = {c: w * context(a, b, c, 3) for c, w in weights.items()}
            third[a, b] = normalize(weights)
            for c, prob in third[a, b].items():
                joint[a, b, c] = first[a] * second[a][b] * prob
    return first, second, third, joint


def market_distribution(cars, quotes):
    expected = set(permutations(cars, 3))
    values = {tuple(map(int, buy.split('-'))): 1 / price for buy, price in quotes.items()}
    if set(values) != expected:
        return None
    total = sum(values.values())
    return {k: v / total for k, v in values.items()}


def candidate_portfolios(scores, quotes, policy, *, conditional=False, market_only=False, context=None):
    cars = sorted(r['car_no'] for r in scores)
    market = market_distribution(cars, quotes)
    if (conditional or market_only) and market is None:
        return {'available': False, 'reason': 'complete_market_required', 'pools': {'本線': [], '穴': []}}
    first, second, third, joint = distributions(scores, market if conditional else None, context)
    if market_only:
        joint = market
    risk = float(policy['risk_score'])
    heads = {policy['fixed_car']} if policy.get('first_fixed') else pool(first, 3 if risk >= 50 else 2)
    seconds = {a: pool(weights, min(len(weights), 5 if risk >= 50 else 4)) for a, weights in second.items()}
    thirds = {ab: pool(weights, min(len(weights), 6 if risk >= 50 else 5)) for ab, weights in third.items()}
    if policy.get('first_fixed') or risk >= 70:
        thirds = {ab: set(weights) for ab, weights in third.items()}
    groups = {'本線': [], '穴': []}
    for (a, b, c), probability in joint.items():
        buy = f'{a}-{b}-{c}'
        if buy not in quotes:
            continue
        price = quotes[buy]
        group = '穴' if price >= 100 else '本線'
        ev = probability * price
        # The market-only arm is an explicit probability-ranking benchmark,
        # not a claim that market-implied EV clears the AI eligibility gate.
        if not market_only:
            if group == '本線' and (a not in heads or b not in seconds[a] or c not in thirds[a, b] or ev < 1.10):
                continue
            if group == '穴' and (probability < .001 or ev < 1.25):
                continue
        groups[group].append({'buy': buy, 'group': group, 'stake_yen': 100,
                              'odds': price, 'prob': probability, 'ev': ev})
    for group, values in groups.items():
        values.sort(key=lambda t: (-round(t['prob'], 14), -round(t['ev'], 12), t['buy']) if group == '本線' or market_only
                    else (-round(t['ev'], 12), -round(t['prob'], 14), t['buy']))
        groups[group] = values[:12]
    return {'available': True, 'reason': 'eligible', 'pools': groups,
            'distribution': {f'{a}-{b}-{c}': p for (a, b, c), p in joint.items()},
            'probability_method': 'market_only' if market_only else 'conditional_once' if conditional else 'position_scores',
            'calibrated': False}


def take(portfolio, group, count):
    if not portfolio['available'] or count <= 0 or len(portfolio['pools'][group]) < count:
        return None
    return portfolio['pools'][group][:count]


def public_preserved(riders, market, legacy_frame, policy):
    """Live specialist repair, with no increase to either legacy ticket count."""
    from betting_logic import clean_odds, score_riders, race_plan
    # Hold the unmodified evaluator's risk policy fixed across challengers.
    # Preserved score units never enter the legacy absolute-gap thresholds.
    policy = race_plan(score_riders(riders, market))
    prices = {str(r.buy): float(r.odds_used) for r in clean_odds(market).itertuples()
              if math.isfinite(float(r.odds_used)) and 1 <= float(r.odds_used) < 9999.9}
    scores = [{'car_no': int(r.car_no), 'first': float(r.score_first),
               'second': float(r.score_second), 'third': float(r.score_third)} for r in riders.itertuples()]
    portfolio = candidate_portfolios(scores, prices, policy)
    tickets = []
    for group in ('本線', '穴'):
        cap = int((legacy_frame.is_selected & legacy_frame.ticket_group.eq(group)).sum())
        tickets.extend(portfolio['pools'][group][:cap])
    return tickets
