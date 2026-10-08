"""Opt-in position market hypotheses; prices are support, not calibrated truth."""
from itertools import permutations
import math


def position_market(riders, market):
    cars = sorted(int(c) for c in riders.car_no)
    expected = set(permutations(cars, 3))
    quotes = {}
    for row in market.itertuples():
        key = tuple(int(c) for c in str(row.buy).split('-'))
        price = float(row.odds_used)
        if key not in expected or key in quotes or not math.isfinite(price) or not 1 <= price < 9999.9:
            raise ValueError('position market requires unique valid complete quotes')
        quotes[key] = 1 / price
    if set(quotes) != expected:
        raise ValueError('position market requires all permutations')
    total = sum(quotes.values())
    joint = {key: value / total for key, value in quotes.items()}
    marginal = [{car: sum(p for key, p in joint.items() if key[pos] == car)
                 for car in cars} for pos in range(3)]
    ranks = [{car: i + 1 for i, car in enumerate(sorted(cars, key=lambda c: (-m[c], c)))}
             for m in marginal]
    # Association ratios remove the generic second/third popularity. Only the
    # market's change in support for this chosen prefix adjusts the model.
    by = riders.set_index('car_no')
    second, third = {}, {}
    for a in cars:
        weights = {b: float(by.at[b, 'score_second']) *
                   math.sqrt(sum(p for (x, y, _), p in joint.items() if x == a and y == b)
                             / marginal[1][b]) for b in cars if b != a}
        mass = sum(weights.values())
        second[a] = {b: w / mass for b, w in weights.items()}
        for b in cars:
            if b == a:
                continue
            weights = {c: float(by.at[c, 'score_third']) * math.sqrt(joint[a, b, c] / marginal[2][c])
                       for c in cars if c not in (a, b)}
            mass = sum(weights.values())
            third[a, b] = {c: w / mass for c, w in weights.items()}
    probabilities = {(a, b, c): float(by.at[a, 'p_win']) * second[a][b] * third[a, b][c]
                     for a, b, c in expected}
    return {'ranks': ranks, 'marginals': marginal, 'third_given_first_second': third,
            'probabilities': probabilities,
            'method': 'position_market_association_v1_uncalibrated'}
