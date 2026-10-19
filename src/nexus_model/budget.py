"""Exact 100-yen allocations from preclose quotes and integer-yen settlement."""

from fractions import Fraction
from heapq import heapify, heappop, heappush
from numbers import Integral


def allocate(tickets, odds, budget):
    """Keep every ticket and maximize the lowest quoted return.

    An unavailable allocation deliberately contains no stakes. Callers must
    preserve the preclose quote and allocation snapshot when publishing it.
    """
    if (
        isinstance(budget, bool)
        or not isinstance(budget, Integral)
        or budget < 5000
        or budget > 10000
        or budget % 100
    ):
        raise ValueError("budget must be 5000..10000 in 100 yen steps")
    budget = int(budget)
    tickets = list(tickets)
    if not tickets or len(set(tickets)) != len(tickets) or len(tickets) * 100 > budget:
        raise ValueError("invalid ticket set")

    prices, missing, invalid = [], [], []
    for ticket in tickets:
        try:
            quote = odds[ticket]
        except (KeyError, TypeError):
            missing.append(ticket)
            continue
        if quote is None or (isinstance(quote, str) and quote.strip() in ("", "-", "—")):
            missing.append(ticket)
            continue
        try:
            price = Fraction(str(quote)) * 100
            if price <= 0:
                raise ValueError("nonpositive odds")
        except (ValueError, ZeroDivisionError, TypeError, OverflowError):
            invalid.append(ticket)
            continue
        prices.append(price)

    if missing:
        return dict(
            available=False,
            budget=budget,
            reason_code="missing_odds",
            reason="購入前オッズ待ち",
            missing_tickets=missing,
            invalid_tickets=invalid,
        )
    if invalid:
        return dict(
            available=False,
            budget=budget,
            reason_code="invalid_odds",
            reason="購入前オッズを確認できません",
            invalid_tickets=invalid,
        )

    units = budget // 100
    amounts = [1] * len(tickets)
    heap = [(price, price, ticket, i) for i, (ticket, price) in enumerate(zip(tickets, prices))]
    heapify(heap)
    # To lift the current minimum, every ticket at that minimum must receive
    # another unit. Filling a minimum first is therefore exact discrete maximin;
    # ties favor lower odds, then the ticket key, for reproducible allocations.
    for _ in range(units - len(tickets)):
        _, price, ticket, i = heappop(heap)
        amounts[i] += 1
        heappush(heap, (price * amounts[i], price, ticket, i))

    minimum = heap[0][0]
    # Fractions avoid optimistic floating-point rounding. Quotes yielding a
    # fractional yen are conservatively reported in whole yen.
    minimum_yen = minimum.numerator // minimum.denominator
    if minimum < budget:
        return dict(
            available=False,
            budget=budget,
            reason_code="no_break_even",
            reason="この予算・オッズでは全買い目のトリガミ回避配分はできません",
            best_possible_return=minimum_yen,
            quote_break_even=False,
        )
    return dict(
        available=True,
        budget=budget,
        stakes=dict(zip(tickets, [amount * 100 for amount in amounts])),
        minimum_return=minimum_yen,
        best_possible_return=minimum_yen,
        quote_break_even=True,
    )


def _integer_yen(value, field):
    """Accept exact integer amounts without passing through float arithmetic."""
    try:
        amount = Fraction(str(value))
    except (ValueError, ZeroDivisionError, TypeError, OverflowError) as exc:
        raise ValueError(f"{field} must be a nonnegative integer yen amount") from exc
    if amount.denominator != 1 or amount < 0:
        raise ValueError(f"{field} must be a nonnegative integer yen amount")
    return amount.numerator


def settle(stakes, payouts, refunded=(), refund_all=False):
    """Settle frozen stakes using official payouts per 100 yen.

    ``stake`` and ``returned`` retain the original gross-cash definitions.
    ``net_stake`` excludes refunds; ``winnings`` excludes returned stakes, so
    winnings / net_stake can be aggregated without treating a refund as a hit.
    """
    amounts = {ticket: _integer_yen(value, "stake") for ticket, value in stakes.items()}
    if any(amount % 100 for amount in amounts.values()):
        raise ValueError("stakes must be in 100 yen units")
    stake = sum(amounts.values())
    refunded = set(amounts) if refund_all else set(refunded)
    refund = sum(amount for ticket, amount in amounts.items() if ticket in refunded)
    net_stake = stake - refund
    winnings = sum(
        _integer_yen(payouts[ticket], "payout") * (amount // 100)
        for ticket, amount in amounts.items()
        if amount and ticket not in refunded and ticket in payouts
    )
    return dict(
        stake=stake,
        returned=winnings + refund,
        net_stake=net_stake,
        winnings=winnings,
        refund=refund,
        hit=winnings > 0,
        void=net_stake == 0,
    )
