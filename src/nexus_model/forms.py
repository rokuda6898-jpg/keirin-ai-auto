"""Frozen inference extracted from audited research; do not train here."""
import numpy as np
import itertools, math, functools
from collections import Counter, defaultdict, deque
from datetime import date,datetime,timezone,timedelta
from . import shape as legacy

def expand(blocks):
    tickets = []
    for block in blocks:
        part = [t for t in itertools.product(block['first'], block['second'], block['third']) if len(set(t)) == 3]
        assert len(part) == block['count']
        tickets.extend(part)
    assert len(tickets) == len(set(tickets)), 'Overlapping formations would double the stake'
    return sorted(tickets)


def encode(raw_blocks, cars=None):
    def car(x):
        return int(cars[x]) if cars is not None else int(x)
    blocks = []
    for first, seconds, thirds in raw_blocks:
        f, b, c = [car(first)], sorted(map(car, seconds)), sorted(map(car, thirds))
        count = sum(len(set(t)) == 3 for t in itertools.product(f, b, c))
        assert count > 0
        blocks.append(dict(first=f, second=b, third=c, count=count))
    assert 1 <= len(blocks) <= 4
    return blocks


def select(cars, scores, count, policy):
    if policy not in ['axis1', 'axis2', 'four', 'axis2_wide', 'four_wide']:
        raise ValueError('The chosen policy must use full formations on every race')
    v = legacy.cube(cars, scores)
    anchors = sorted(range(len(cars)), key=lambda f: (-float(v[f].sum()), cars[f]))
    base_policy = policy.removesuffix('_wide')
    vv = legacy.flatten_third(v) if policy.endswith('_wide') else v
    raw = legacy.formation_set(vv, count, anchors, [base_policy])[base_policy]
    blocks = encode(raw, cars)
    tickets = expand(blocks)
    assert len(tickets) == count and all(set(t) <= set(cars) for t in tickets)
    return {'blocks': blocks, 'tickets': [list(t) for t in tickets]}

