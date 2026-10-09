"""Discrete, independent virtual 6,000-yen plans. Never executes purchases.

Within each formula's probability-ranked top N, each ticket gets at least
100 yen. The remaining 100-yen units maximize expected log(6000 + payout).
The 6000-yen utility cushion penalizes concentration; it is not extra spending.
For mutually exclusive trifectas this separable concave objective has an exact
discrete greedy solution. Expected ROI is a model estimate, not realized ROI.
"""
import math
from official_outcomes import ticket

BUDGET = 6000
UNIT = 100
POINTS = (3, 6, 12)
POLICY = 'top_probability_then_discrete_expected_log_payout_v1'


def allocate(ranked, prices, points, budget=BUDGET):
    if points not in POINTS or budget != BUDGET:
        raise ValueError('only independent 3/6/12-point 6000-yen plans are supported')
    picks = ranked[:points]
    if len(picks) != points:
        return {'status': 'candidate_shortage', 'budget_yen': budget, 'spent_yen': 0, 'tickets': []}
    if len({t['buy'] for t in picks}) != points or any(not ticket(t['buy']) for t in picks):
        raise ValueError('invalid or duplicate tickets')
    ps = [float(t['probability']) for t in picks]
    if any(not math.isfinite(p) or not 0 < p <= 1 for p in ps) or sum(ps) > 1+1e-8:
        raise ValueError('invalid probabilities')
    if any(t['buy'] not in prices or not math.isfinite(float(prices[t['buy']]))
           or not 1 <= float(prices[t['buy']]) < 9999.9 for t in picks):
        return {'status': 'verified_odds_required', 'budget_yen': budget, 'spent_yen': 0, 'tickets': []}
    odds = [float(prices[t['buy']]) for t in picks]
    stakes = [UNIT]*points
    for _ in range((budget-UNIT*points)//UNIT):
        gains = [p*math.log1p(o*UNIT/(budget+o*s)) for p,o,s in zip(ps, odds, stakes)]
        best = max(range(points), key=lambda i: (gains[i], -i))
        stakes[best] += UNIT
    expected = sum(p*o*s for p,o,s in zip(ps, odds, stakes))
    return {'status': 'allocated', 'policy': POLICY, 'budget_yen': budget, 'spent_yen': sum(stakes),
            'estimated_hit_probability': sum(ps), 'estimated_return_yen': expected,
            'estimated_roi': expected/budget, 'expected_profit_yen': expected-budget,
            'tickets': [{'buy': t['buy'], 'probability': p, 'odds_at_capture': o,
                         'stake_yen': s, 'return_if_hit_yen': o*s} for t,p,o,s in zip(picks,ps,odds,stakes)]}
