"""Let the saved model choose a fixed or flexible first; open second if needed.

This is a ticket-selection policy, not a new trained model.  A plan with two
second-place candidates is preserved, including a single first-place anchor.
Otherwise try one-ticket exchanges in score order, with a
bounded exact-formation packing search.  The fallback scores a small declared
family of complete formations; it does not claim a global constrained optimum.
"""
import itertools
import math

from . import exact

POLICY_ID = 'flexible-first-v1'
MAX_ONE_SWAP_CHECKS = 128


def _expand(blocks):
    tickets = []
    for block in blocks:
        part = [t for t in itertools.product(block['first'], block['second'], block['third'])
                if len(set(t)) == 3]
        if len(part) != block['count']:
            raise ValueError('formation count does not match its full expansion')
        tickets.extend(part)
    if len(tickets) != len(set(tickets)):
        raise ValueError('overlapping formations')
    return set(tickets)


def _has_second_alternatives(tickets):
    return len({t[1] for t in tickets}) >= 2


def _changed_plan(blocks, tickets, original, search):
    # Exact pack results are cached, so give the caller its own block lists.
    blocks = [dict(first=list(b['first']), second=list(b['second']),
                   third=list(b['third']), count=b['count']) for b in blocks]
    tickets = set(tickets)
    if not _has_second_alternatives(tickets) or len(blocks) > 4 or _expand(blocks) != tickets:
        raise ValueError('open-formation contract failed')
    return dict(blocks=blocks, tickets=[list(t) for t in sorted(tickets)],
                policy_id=POLICY_ID, plan_changed=True, search=search,
                replacement_count=len(original - tickets))


def _one_swap(original, scores):
    """Find a score-ranked one-ticket repair when exact packing permits it."""
    additions = [t for t in scores if t not in original]
    options = []
    for removed in sorted(original):
        remaining = original - {removed}
        seconds = {t[1] for t in remaining}
        for added in additions:
            if len(seconds | {added[1]}) < 2:
                continue
            options.append((scores[added] - scores[removed], removed, added))
    options.sort(key=lambda x: (-x[0], x[1], x[2]))
    for _, removed, added in options[:MAX_ONE_SWAP_CHECKS]:
        tickets = tuple(sorted((original - {removed}) | {added}))
        blocks = exact.pack(tickets, maxblocks=4)
        if blocks is not None:
            return _changed_plan(blocks, tickets, original, 'open_one_ticket_exchange')
    return None


def _nine_block(first, second, cars, scores):
    """Best nine-ticket rectangle for this first rider and three seconds."""
    outside = [c for c in cars if c != first and c not in second]
    column = {c: sum(scores[first, b, c] for b in second if b != c)
              for c in cars if c != first}
    outside.sort(key=lambda c: (-column[c], c))
    # 3 * len(third) - len(second intersect third) == 9 has these two forms.
    pools = [list(second) + outside[:1]]
    if len(outside) >= 3:
        pools.append(outside[:3])
    third = min(pools, key=lambda cs: (-sum(column[c] for c in cs), tuple(sorted(cs))))
    block = dict(first=[first], second=list(second), third=sorted(third), count=9)
    return sum(column[c] for c in third), block


def _compact_fallback(cars, scores, cap, original):
    """Score a three-rider box plus zero, one or two disjoint nine-ticket blocks.

    The family always opens both first and second and uses at most three
    complete formations.  Its 6 / 15 / 24 tickets include no duplicated stake.
    The original full triple scores select every rider and every third pool.
    """
    extra_count = (cap - 6) // 9
    answer = None
    best = -math.inf
    for core in itertools.combinations(sorted(cars), 3):
        box = dict(first=list(core), second=list(core), third=list(core), count=6)
        value = sum(scores[t] for t in itertools.permutations(core, 3))
        additions = []
        for first in sorted(set(cars) - set(core)):
            mass, block = _nine_block(first, core, cars, scores)
            additions.append((mass, first, block))
        additions.sort(key=lambda x: (-x[0], x[1]))
        blocks = [box] + [x[2] for x in additions[:extra_count]]
        value += sum(x[0] for x in additions[:extra_count])
        if value > best:
            best, answer = value, blocks
    tickets = _expand(answer)
    if len(tickets) != cap:
        raise ValueError('compact open formation has the wrong ticket count')
    return _changed_plan(answer, tickets, original, 'open_compact_family')


def select(cars, q, cap, plan):
    """Keep the model's first-place choice; never mutate its full-order scores."""
    cars = tuple(cars)
    if len(cars) < 5 or len(set(cars)) != len(cars) or cap not in (6, 15, 24):
        raise ValueError('open formations require distinct riders and 6, 15 or 24 tickets')
    triples = tuple(itertools.permutations(cars, 3))
    values = tuple(float(x) for x in q)
    if len(values) != len(triples) or any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError('invalid complete-order scores')
    listed = [tuple(t) for t in plan['tickets']]
    original = set(listed)
    if (len(listed) != cap or len(original) != cap or not 1 <= len(plan['blocks']) <= 4
            or any(len(set(t)) != 3 or not set(t) <= set(cars) for t in original)
            or _expand(plan['blocks']) != original):
        raise ValueError('invalid original complete formation')
    if _has_second_alternatives(original):
        # Preserve the existing chosen tickets, their ordering, and block shapes.
        return dict(plan, policy_id=POLICY_ID, plan_changed=False,
                    search=plan.get('search', 'saved_formation'), replacement_count=0)
    scores = dict(zip(triples, values))
    changed = _one_swap(original, scores)
    return changed if changed is not None else _compact_fallback(cars, scores, cap, original)
