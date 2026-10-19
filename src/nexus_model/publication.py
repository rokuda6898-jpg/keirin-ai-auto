"""Preclose allocation snapshots and outcome settlement; no source requests."""
from datetime import datetime
from . import budget

ALLOCATION_POLICY = 'break-even-v1'


def is_open_prediction(prediction):
    tickets = [str(t).split('-') for t in prediction.get('tickets', [])]
    return bool(tickets) and len({t[1] for t in tickets}) >= 2


def allocation_snapshot(prediction, quotes, odds_at, saved_at, close_at):
    """Never turn a postclose quote or newly calculated amount into a prior bet."""
    if any(datetime.fromisoformat(t) >= datetime.fromisoformat(close_at)
           for t in (odds_at, saved_at)):
        raise ValueError('allocation and odds must be saved before close')
    tickets = prediction['tickets']
    if any(t not in quotes for t in tickets):
        raise ValueError('complete preclose quotes required')
    allocations = {str(b): budget.allocate(tickets, quotes, b)
                   for b in range(5000, 10001, 100)}
    auto_budget = next((int(b) for b, a in allocations.items() if a['available']), None)
    # Publish the six manual presets plus the chosen auto budget. All 51
    # budgets were evaluated, but duplicating every stake matrix bloats feeds.
    feasible = [int(b) for b, a in allocations.items() if a['available']]
    keep = set(range(5000, 10001, 1000)) | ({auto_budget} if auto_budget else set())
    compact = {b: a for b, a in allocations.items() if int(b) in keep}
    return dict(allocations=compact, auto_budget=auto_budget,
                evaluated_budgets=51, feasible_budgets=feasible,
                allocation_policy=ALLOCATION_POLICY, allocation_saved_at=saved_at,
                auto_saved_at=saved_at, odds_at=odds_at,
                odds={t: quotes[t] for t in tickets})


def settle_prediction(prediction, payouts, refunded=(), refund_all=False):
    results = {'flat': budget.settle({t: 100 for t in prediction['tickets']},
                                     payouts, refunded, refund_all)}
    for key, allocation in prediction.get('allocations', {}).items():
        # Old, genuinely preclose snapshots remain historical records even if
        # they could not avoid a loss. New snapshots contain no such stakes.
        if allocation.get('available') and allocation.get('stakes'):
            results[str(key)] = budget.settle(allocation['stakes'], payouts, refunded, refund_all)
    chosen = prediction.get('auto_budget')
    if prediction.get('auto_saved_at') and chosen and str(chosen) in results:
        results['auto'] = dict(results[str(chosen)], budget=chosen)
    return results
