"""Immutable conditional-trifecta stake simulations; never places bets.

Allocation uses only a quote captured while publishing a NEW prediction.
Settlement uses only official ticket-level payouts, never the saved quote.
Legacy v1 plans keep their JPY 10,000 stakes. New v2 plans use the smallest
feasible budget between JPY 5,000 and 10,000, in JPY 100 increments.
"""
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from hashlib import sha256
import json
import re

BUDGET = 10_000
MIN_BUDGET = 5_000
UNIT = 100
VERSION = 'conditional-budget-10000-v1'
RANGE_VERSION = 'conditional-budget-5000-10000-v2'


def valid_tickets(tickets):
    return (isinstance(tickets, list) and 0 < len(tickets) <= 12
            and len(set(tickets)) == len(tickets)
            and all(isinstance(t, str) and re.fullmatch(r'[1-9]-[1-9]-[1-9]', t)
                    and len(set(t.split('-'))) == 3 for t in tickets))


def number(value):
    try:
        value = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        raise ValueError('invalid numeric value') from None
    if not value.is_finite() or value <= 0:
        raise ValueError('nonpositive or nonfinite numeric value')
    return value


def fingerprint(tickets):
    return sha256(json.dumps(tickets, separators=(',', ':')).encode()).hexdigest()


def allocate(tickets, odds, budget=BUDGET, unit=UNIT):
    """Exact discrete maximin; preserve every ticket and use the whole budget."""
    if not valid_tickets(tickets):
        raise ValueError('invalid or duplicate trifecta tickets')
    if (type(budget) is not int or type(unit) is not int or unit <= 0
            or budget <= 0 or budget % unit):
        raise ValueError('invalid budget or unit')
    prices = [number(odds[t]) for t in tickets]
    fractions = [Fraction(p) * unit for p in prices]
    units = budget // unit
    base = {'version': VERSION, 'budget_yen': budget, 'unit_yen': unit,
            'mode': 'simulation', 'ticket_sha256': fingerprint(tickets),
            'snapshot_odds': dict(zip(tickets, map(str, prices))),
            'total_stake_yen': 0, 'allocations': []}

    def required(target):
        ratios = [Fraction(target) / p for p in fractions]
        return [max(1, (v.numerator + v.denominator - 1) // v.denominator)
                for v in ratios]

    minimum = required(budget)
    if sum(minimum) > units:
        return {**base, 'status': 'infeasible',
                'reason': f'全買い目を残すと{budget:,}円でトリガミ回避が成立しません。見送り。',
                'required_stake_yen': sum(minimum) * unit}
    # The optimum must be a payout reachable by one ticket's integer stake.
    candidates = sorted({p * k for p in fractions
                         for k in range(1, units - len(tickets) + 2)})
    lo, hi = 0, len(candidates)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(required(candidates[mid])) <= units:
            lo = mid + 1
        else:
            hi = mid
    target = candidates[lo - 1]
    counts = required(target)
    for _ in range(units - sum(counts)):
        i = min(range(len(tickets)), key=lambda j: (fractions[j] * counts[j], j))
        counts[i] += 1
    amounts = [c * unit for c in counts]
    payouts = [p * c for p, c in zip(fractions, counts)]
    if sum(amounts) != budget or min(payouts) < budget:
        raise ValueError('allocation postcondition failed')
    return {**base, 'status': 'allocated', 'reason': '予想時オッズで配分成立',
            'total_stake_yen': budget, 'minimum_payout_yen': float(min(payouts)),
            'minimum_profit_yen': float(min(payouts) - budget),
            'allocations': [
                {'buy': t, 'odds': str(p), 'stake_yen': s,
                 'estimated_payout_yen': float(pay),
                 'break_even_odds': float(Fraction(budget, s))}
                for t, p, s, pay in zip(tickets, prices, amounts, payouts)]}


def allocate_range(tickets, odds, min_budget=MIN_BUDGET, max_budget=BUDGET):
    """Keep every ticket and choose the smallest no-loss budget, exactly.

    For each candidate budget, the sum of ceil(budget / return-per-unit)
    is a necessary and sufficient feasibility check. Only the first feasible
    budget needs the existing exact maximin allocator. Quote values never
    change during the search, and an impossible plan never drops selections.
    """
    if not valid_tickets(tickets):
        raise ValueError('invalid or duplicate trifecta tickets')
    if (type(min_budget) is not int or type(max_budget) is not int
            or not MIN_BUDGET <= min_budget <= max_budget <= BUDGET
            or min_budget % UNIT or max_budget % UNIT):
        raise ValueError('budget range must be 5,000–10,000 yen in 100-yen steps')
    prices = {t: number(odds[t]) for t in tickets}
    returns = [Fraction(prices[t]) * UNIT for t in tickets]
    chosen = max_budget
    for budget in range(min_budget, max_budget + UNIT, UNIT):
        ratios = [Fraction(budget) / payout for payout in returns]
        required = sum(max(1, (r.numerator + r.denominator - 1) // r.denominator)
                       for r in ratios)
        if required <= budget // UNIT:
            chosen = budget
            break
    plan = allocate(tickets, prices, budget=chosen)
    plan.update(version=RANGE_VERSION, minimum_budget_yen=min_budget,
                maximum_budget_yen=max_budget,
                budget_selection='smallest_feasible_no_loss')
    if plan['status'] == 'infeasible':
        plan['reason'] = (f'全買い目を残すと{min_budget:,}〜{max_budget:,}円の範囲で'
                          'トリガミ回避が成立しません。配分版は見送り。')
    return plan


def unavailable(reason, version=VERSION):
    plan = {'version': version, 'status': 'unavailable', 'reason': reason,
            'mode': 'simulation', 'budget_yen': BUDGET, 'unit_yen': UNIT,
            'total_stake_yen': 0, 'allocations': []}
    if version == RANGE_VERSION:
        plan.update(minimum_budget_yen=MIN_BUDGET, maximum_budget_yen=BUDGET,
                    budget_selection='smallest_feasible_no_loss')
    return plan


def freeze(row, quote_loader, clock):
    """Called once, only for new forecasts. No retry/backfill on later runs."""
    if 'budget' in row:
        return row['budget']
    attempted = clock()
    try:
        tickets = row['top12']
        if not valid_tickets(tickets):
            raise ValueError('invalid tickets')
        quote = quote_loader(row, clock)
        from datetime import datetime
        captured = datetime.fromisoformat(quote['captured_at_jst'])
        saved = clock()
        if (captured.tzinfo is None or saved.tzinfo is None
                or not 0 <= (saved - captured).total_seconds() <= 180
                or saved.timestamp() >= float(row['close_at'])
                or captured.date().isoformat() != row['date']):
            raise ValueError('stale, post-close, or cross-date quote')
        for key in ('race_id', 'date', 'venue'):
            if str(quote.get(key, '')) != str(row[key]):
                raise ValueError('quote identity mismatch: ' + key)
        if (int(quote.get('race_no', -1)) != int(float(row['race_no']))
                or quote.get('bet_type') != 'trifecta'
                or not quote.get('source_url') or not quote.get('provider')):
            raise ValueError('quote race/bet/source mismatch')
        plan = allocate_range(tickets, quote['odds'])
        plan.update(odds_captured_at_jst=captured.isoformat(),
                    allocation_frozen_at_jst=saved.isoformat(),
                    allocation_attempted_at_jst=attempted.isoformat(),
                    source_url=quote['source_url'], provider=quote['provider'])
        return plan
    except Exception as error:
        # Quotes must never prevent saving the unchanged model prediction.
        return unavailable('予想時オッズを利用できません: ' + str(error)[:220],
                           version=RANGE_VERSION) | {
            'allocation_attempted_at_jst': attempted.isoformat()}


def check_saved_plan(plan, tickets):
    if not isinstance(plan, dict):
        return False
    lines = plan.get('allocations', [])
    budget = plan.get('budget_yen')
    version = plan.get('version')
    budget_valid = (type(budget) is int and (
        (version == VERSION and budget == BUDGET)
        or (version == RANGE_VERSION and MIN_BUDGET <= budget <= BUDGET
            and budget % UNIT == 0)))
    return (budget_valid and isinstance(lines, list)
            and all(isinstance(x, dict) for x in lines)
            and plan.get('unit_yen') == UNIT and plan.get('total_stake_yen') == budget
            and valid_tickets(tickets) and plan.get('ticket_sha256') == fingerprint(tickets)
            and [x.get('buy') for x in lines] == tickets
            and all(type(x.get('stake_yen')) is int and x['stake_yen'] >= UNIT
                    and x['stake_yen'] % UNIT == 0 for x in lines)
            and sum(x['stake_yen'] for x in lines) == budget)


def settle(plan, tickets, outcome, cancelled_cars=()):
    """Official payouts are yen per 100 yen; no forecast odds are read here.

    The existing result adapter does not certify refunds/voids. Hold these
    cases visibly for official reconciliation rather than fabricate returns.
    """
    empty = {'mode': 'simulation', 'payout_yen': None, 'profit_yen': None,
             'roi': None, 'hit': None, 'invested_yen': 0}
    if not plan or plan.get('status') != 'allocated':
        return {**empty, 'status': 'not_purchased'}
    budget = plan.get('budget_yen')
    if not check_saved_plan(plan, tickets):
        return {**empty, 'status': 'invalid_snapshot',
                'invested_yen': budget if type(budget) is int and budget > 0 else 0}
    empty['invested_yen'] = budget
    if any(set(t.split('-')) & set(map(str, cancelled_cars)) for t in tickets):
        return {**empty, 'status': 'refund_pending'}
    if not outcome:
        return {**empty, 'status': 'pending'}
    winners = set(outcome['winning_buys'])
    hit = bool(winners & set(tickets))
    if not outcome.get('payout_complete'):
        return {**empty, 'status': 'payout_pending', 'hit': hit}
    try:
        # Validate all winning payouts, including both winning orders in ties.
        pays = {t: number(outcome['payouts'][t]) for t in winners}
        if any(v != v.to_integral_value() for v in pays.values()):
            raise ValueError('nonintegral official payout')
        paid = sum((int(pays[t['buy']]) * (t['stake_yen'] // UNIT)
                    for t in plan['allocations'] if t['buy'] in winners), 0)
    except (ValueError, KeyError, TypeError):
        return {**empty, 'status': 'payout_pending', 'hit': hit}
    return {**empty, 'status': 'settled', 'hit': hit, 'payout_yen': paid,
            'profit_yen': paid - budget, 'roi': paid / budget,
            'result_source': outcome.get('result_source', ''),
            'result_source_url': outcome.get('source_url', ''),
            'result_observed_at_jst': outcome.get('source_observed_at', '')}
