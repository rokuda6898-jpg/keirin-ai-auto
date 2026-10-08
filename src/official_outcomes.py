"""Official ticket-level settlement, including ties; standard-library only."""
import json
import math
from itertools import permutations


def ticket(value):
    cars = str(value).split('-')
    return len(cars) == 3 and len(set(cars)) == 3 and all(c.isdigit() and 1 <= int(c) <= 9 for c in cars)


def winning_orders(ordered):
    """Expand tied official positions, respecting occupied finishing slots."""
    groups = {}
    for order, car, *_ in ordered:
        if order in (1, 2, 3):
            groups.setdefault(order, []).append(int(car))
    prefixes = [()]
    occupied = 0
    for order, cars in sorted(groups.items()):
        if order != occupied + 1 or len(set(cars)) != len(cars):
            return []
        take = min(3 - occupied, len(cars))
        prefixes = [p + suffix for p in prefixes for suffix in permutations(sorted(cars), take)]
        occupied += len(cars)
        if occupied >= 3:
            return sorted('-'.join(map(str, p)) for p in prefixes)
    return []


def normalize_outcome(result):
    if str(result.get('official_result_available')).lower() not in ('true', '1'):
        return None
    raw = result.get('payouts_trifecta_json', {})
    try:
        raw = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        raw = {}
    payouts = {}
    for buy, price in (raw.items() if isinstance(raw, dict) else []):
        try:
            value = float(price)
        except (ValueError, TypeError):
            continue
        if ticket(buy) and math.isfinite(value) and value > 0:
            payouts[buy] = value
    winners = result.get('actual_trifecta_buys', [])
    if isinstance(winners, str):
        try:
            winners = json.loads(winners)
        except ValueError:
            winners = []
    winners = {b for b in winners if ticket(b)} if isinstance(winners, list) else set()
    winners.update(b for b in str(result.get('actual_trifecta', '')).split('|') if ticket(b))
    winners.update(payouts)
    if not winners:
        return None
    # A single legacy payoff cannot stand in for every tied winning ticket.
    if len(winners) == 1 and not payouts:
        try:
            value = float(result.get('actual_trifecta_odds')) * 100
            if math.isfinite(value) and value > 0:
                payouts[next(iter(winners))] = value
        except (ValueError, TypeError):
            pass
    return {'winning_buys': sorted(winners), 'payouts': payouts,
            'payout_complete': set(payouts) == winners,
            'source_url': str(result.get('source_url', '')),
            'result_source': str(result.get('result_source', '')),
            'source_observed_at': str(result.get('result_observed_at_jst', '')),
            'decided_at': result.get('decided_at')}


def ticket_return(tickets, outcome):
    winners = set(outcome['winning_buys'])
    hit = any(t['buy'] in winners for t in tickets)
    returned = sum(outcome['payouts'].get(t['buy'], 0) * t['stake_yen'] / 100 for t in tickets)
    return hit, returned if outcome['payout_complete'] else None


def winning_ticket_values(row, bet_type='trifecta'):
    if bet_type == 'trifecta':
        outcome = normalize_outcome({**dict(row), 'official_result_available': True})
        if outcome:
            return outcome['winning_buys']
    try:
        raw = row.get(f'payouts_{bet_type}_json', '{}')
        payouts = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(payouts, dict) and payouts:
            return sorted(k for k, v in payouts.items() if float(v) > 0)
    except (ValueError, TypeError):
        pass
    return str(row.get(f'actual_{bet_type}', '')).split('|')
