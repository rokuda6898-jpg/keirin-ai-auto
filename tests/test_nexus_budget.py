"""Allocation feasibility, independent maximin checks, and cash settlement."""

import itertools
import sys
import unittest
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nexus_model.budget import allocate, settle


def compositions(total, count):
    """Every positive integer allocation, independently of the allocator."""
    for cuts in itertools.combinations(range(1, total), count - 1):
        boundaries = (0, *cuts, total)
        yield tuple(boundaries[i + 1] - boundaries[i] for i in range(count))


class Allocation(unittest.TestCase):
    def test_rounding_changes_feasibility_at_5100(self):
        tickets = ["1-2-3", "2-3-1", "3-1-2"]
        odds = dict.fromkeys(tickets, "3.0")
        unavailable = allocate(tickets, odds, 5000)
        self.assertFalse(unavailable["available"])
        self.assertEqual(unavailable["reason_code"], "no_break_even")
        self.assertEqual(unavailable["best_possible_return"], 4800)
        self.assertNotIn("stakes", unavailable)
        available = allocate(tickets, odds, 5100)
        self.assertTrue(available["available"])
        self.assertEqual(available["stakes"], dict.fromkeys(tickets, 1700))
        self.assertEqual(available["minimum_return"], 5100)
        self.assertFalse(allocate(tickets, odds, 5200)["available"])

    def test_all_51_budgets_and_invalid_budget_units(self):
        for budget in range(5000, 10001, 100):
            with self.subTest(budget=budget):
                result = allocate(["1-2-3"], {"1-2-3": "2.0"}, budget)
                self.assertEqual(result["stakes"], {"1-2-3": budget})
                self.assertEqual(result["minimum_return"], budget * 2)
                self.assertIs(type(result["minimum_return"]), int)
        for budget in (4900, 10100, 5050, 5000.0, "5000", True):
            with self.subTest(invalid_budget=budget), self.assertRaises(ValueError):
                allocate(["1-2-3"], {"1-2-3": 2}, budget)

    def test_all_tickets_preserved_with_integer_stakes(self):
        tickets = ["1-2-3", "2-1-3", "3-2-1"]
        odds = dict(zip(tickets, ("2.0", "20.0", "300.0")))
        original_tickets, original_odds = tickets[:], odds.copy()
        for budget in (5000, 5700, 10000):
            result = allocate(tickets, odds, budget)
            self.assertTrue(result["available"])
            self.assertEqual(set(result["stakes"]), set(tickets))
            self.assertEqual(sum(result["stakes"].values()), budget)
            for ticket, stake in result["stakes"].items():
                self.assertIs(type(stake), int)
                self.assertGreaterEqual(stake, 100)
                self.assertEqual(stake % 100, 0)
                self.assertGreaterEqual(Fraction(odds[ticket]) * stake, budget)
        self.assertEqual(tickets, original_tickets)
        self.assertEqual(odds, original_odds)

    def test_missing_or_invalid_odds_never_produce_stakes(self):
        tickets = ["1-2-3", "2-1-3"]
        for quote in (None, "", "—"):
            with self.subTest(missing_quote=quote):
                result = allocate(tickets, {tickets[0]: 20, tickets[1]: quote}, 5000)
                self.assertEqual(result["reason_code"], "missing_odds")
                self.assertFalse(result["available"])
                self.assertEqual(result["missing_tickets"], [tickets[1]])
                self.assertNotIn("stakes", result)
        result = allocate(tickets, {tickets[0]: 20}, 5000)
        self.assertEqual(result["reason_code"], "missing_odds")
        for quote in (0, -1, float("nan"), float("inf"), "invalid"):
            with self.subTest(invalid_quote=quote):
                result = allocate(tickets, {tickets[0]: 20, tickets[1]: quote}, 5000)
                self.assertEqual(result["reason_code"], "invalid_odds")
                self.assertFalse(result["available"])
                self.assertNotIn("stakes", result)

    def test_invalid_ticket_sets_are_rejected(self):
        for tickets in ([], ["1-2-3", "1-2-3"], list(map(str, range(51)))):
            with self.subTest(tickets=tickets), self.assertRaises(ValueError):
                allocate(tickets, {}, 5000)

    def test_maximin_matches_exhaustive_allocations(self):
        # Small ticket sets allow every allocation to be checked. Cases cover
        # unequal odds, ties, successful allocations and impossible budgets.
        for quotes, budget in (
            (("2.0", "3.0", "9.0"), 5000),
            (("3.0", "3.0", "3.0"), 5000),
            (("3.0", "3.0", "3.0"), 5100),
            (("2.1", "3.7", "9.6"), 5100),
            (("2.0", "2.1", "3.0"), 5000),
            (("1.1", "17.3"), 5300),
        ):
            with self.subTest(quotes=quotes, budget=budget):
                tickets = list(map(str, range(len(quotes))))
                prices = [Fraction(quote) * 100 for quote in quotes]
                optimum = max(
                    min(price * units for price, units in zip(prices, allocation))
                    for allocation in compositions(budget // 100, len(tickets))
                )
                result = allocate(tickets, dict(zip(tickets, quotes)), budget)
                self.assertEqual(result["best_possible_return"], optimum)
                self.assertEqual(result["available"], optimum >= budget)
                if result["available"]:
                    actual = min(
                        Fraction(quote) * result["stakes"][ticket]
                        for ticket, quote in zip(tickets, quotes)
                    )
                    self.assertEqual(actual, optimum)
                    self.assertEqual(sum(result["stakes"].values()), budget)

    def test_fractional_quote_is_never_rounded_up_to_break_even(self):
        result = allocate(["1-2-3"], {"1-2-3": "0.9999999999999999"}, 5000)
        self.assertFalse(result["available"])
        self.assertEqual(result["reason_code"], "no_break_even")
        self.assertEqual(result["best_possible_return"], 4999)


class Settlement(unittest.TestCase):
    def test_multiple_wins_and_partial_refund_in_integer_yen(self):
        result = settle(
            {"1-2-3": 200, "2-1-3": 100, "4-5-6": 100},
            {"1-2-3": 400, "2-1-3": 500, "4-5-6": 9999},
            ["4-5-6"],
        )
        self.assertEqual(
            result,
            dict(stake=400, returned=1400, net_stake=300, winnings=1300, refund=100, hit=True, void=False),
        )
        for field in ("stake", "returned", "net_stake", "winnings", "refund"):
            self.assertIs(type(result[field]), int)

    def test_all_refunded_by_flag_or_by_ticket_list(self):
        stakes = {"1-2-3": 200, "2-1-3": 100}
        for kwargs in ({"refund_all": True}, {"refunded": list(stakes)}):
            with self.subTest(kwargs=kwargs):
                result = settle(stakes, {"1-2-3": 1000}, **kwargs)
                self.assertEqual(
                    result,
                    dict(stake=300, returned=300, net_stake=0, winnings=0, refund=300, hit=False, void=True),
                )

    def test_loss_and_zero_stake_are_not_hits(self):
        result = settle({"1-2-3": 200, "2-1-3": 100}, {"3-1-2": 500})
        self.assertEqual(result["net_stake"], 300)
        self.assertEqual(result["winnings"], 0)
        self.assertEqual(result["returned"], 0)
        self.assertFalse(result["hit"])
        self.assertFalse(result["void"])
        self.assertFalse(settle({"1-2-3": 0, "2-1-3": 100}, {"1-2-3": 1000})["hit"])

    def test_no_float_precision_loss_in_payout(self):
        payout = 9007199254740993
        result = settle({"1-2-3": 300}, {"1-2-3": payout})
        self.assertEqual(result["winnings"], payout * 3)
        self.assertIs(type(result["returned"]), int)
        self.assertEqual(settle({"1-2-3": "200"}, {"1-2-3": "1230"})["winnings"], 2460)

    def test_invalid_cash_amounts_are_rejected(self):
        for stakes, payouts in (
            ({"1-2-3": 150}, {"1-2-3": 500}),
            ({"1-2-3": -100}, {}),
            ({"1-2-3": 100}, {"1-2-3": 500.5}),
            ({"1-2-3": 100}, {"1-2-3": float("nan")}),
        ):
            with self.subTest(stakes=stakes, payouts=payouts), self.assertRaises(ValueError):
                settle(stakes, payouts)


if __name__ == "__main__":
    unittest.main()
