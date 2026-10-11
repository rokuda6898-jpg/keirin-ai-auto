import copy
from datetime import datetime, timedelta
from fractions import Fraction
from itertools import permutations, combinations
import unittest
from conditional_budget import allocate, freeze, settle, unavailable

TICKETS = ['-'.join(p) for p in permutations('1234567', 3)][:12]
NOW = datetime.fromisoformat('2026-10-11T09:00:00+09:00')


def row():
    return {'race_id': '012820261011', 'date': '2026-10-11', 'venue': '立川',
            'race_no': 1, 'close_at': (NOW + timedelta(hours=1)).timestamp(),
            'top12': TICKETS.copy(), 'snapshot_at_jst': NOW.isoformat()}


def quote(r, clock):
    return {k: r[k] for k in ('race_id', 'date', 'venue', 'race_no')} | {
        'provider': 'fixture', 'source_url': 'https://example.test/odds',
        'bet_type': 'trifecta', 'captured_at_jst': clock().isoformat(),
        'odds': dict.fromkeys(TICKETS, '20.0')}


def result(winners=None, pays=None, complete=True):
    return {'winning_buys': winners or [TICKETS[0]],
            'payouts': pays if pays is not None else {TICKETS[0]: 2000},
            'payout_complete': complete}


class ConditionalBudgetTests(unittest.TestCase):
    def test_twenty_times_feasible_and_uses_exact_budget(self):
        p = allocate(TICKETS, dict.fromkeys(TICKETS, 20))
        self.assertEqual(p['status'], 'allocated')
        self.assertEqual(p['minimum_payout_yen'], 16000)
        self.assertEqual(sum(x['stake_yen'] for x in p['allocations']), 10000)
        self.assertEqual([x['buy'] for x in p['allocations']], TICKETS)
        self.assertTrue(all(x['stake_yen'] >= 100 and x['stake_yen'] % 100 == 0 for x in p['allocations']))
        self.assertNotIn('roi', p)

    def test_ten_times_infeasible(self):
        p = allocate(TICKETS, dict.fromkeys(TICKETS, 10))
        self.assertEqual(p['status'], 'infeasible')
        self.assertEqual(p['required_stake_yen'], 12000)
        self.assertEqual(p['allocations'], [])

    def test_rounding_makes_twelve_point_one_infeasible(self):
        p = allocate(TICKETS, dict.fromkeys(TICKETS, '12.1'))
        self.assertEqual(p['status'], 'infeasible')
        self.assertEqual(p['required_stake_yen'], 10800)

    def test_break_even_is_not_a_profit(self):
        p = allocate(TICKETS[:2], dict.fromkeys(TICKETS[:2], 2))
        self.assertEqual(p['minimum_profit_yen'], 0)

    def test_missing_and_invalid_quotes_are_not_substituted(self):
        for value in ('NaN', 'Infinity', 0, -1, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                allocate(TICKETS, dict.fromkeys(TICKETS, value))
        with self.assertRaises(KeyError):
            allocate(TICKETS, {})
        with self.assertRaises(ValueError):
            allocate([TICKETS[0]] * 2, dict.fromkeys(TICKETS, 20))

    def test_discrete_optimum_matches_exhaustive_compositions(self):
        tickets = TICKETS[:3]
        for odds in ((3, 5, 12), ('4.1', '4.7', '8.3'), (2, 9, 20)):
            p = allocate(tickets, dict(zip(tickets, odds)), budget=1200)
            optimum = max(min(Fraction(str(o)) * 100 * n for o, n in zip(odds, counts))
                          for i, j in combinations(range(1, 12), 2)
                          for counts in [(i, j-i, 12-j)])
            if optimum < 1200:
                self.assertEqual(p['status'], 'infeasible')
            else:
                self.assertEqual(p['minimum_payout_yen'], float(optimum))

    def test_freeze_retains_initial_quote_and_never_refetches(self):
        r = row(); p = freeze(r, quote, lambda: NOW); r['budget'] = p
        def forbidden(*args):
            self.fail('frozen quote was refetched')
        self.assertIs(freeze(r, forbidden, lambda: NOW), p)
        self.assertEqual(p['snapshot_odds'][TICKETS[0]], '20.0')

    def test_identity_timestamp_and_bet_type_guards(self):
        changes = [{'race_id': 'other'}, {'date': '2026-10-10'}, {'venue': '別場'},
                   {'race_no': 2}, {'bet_type': 'trio'}, {'source_url': ''},
                   {'captured_at_jst': (NOW-timedelta(minutes=4)).isoformat()},
                   {'captured_at_jst': (NOW+timedelta(seconds=1)).isoformat()},
                   {'captured_at_jst': NOW.replace(tzinfo=None).isoformat()},
                   {'odds': {}}]
        for change in changes:
            with self.subTest(change=change):
                p = freeze(row(), lambda r, c: quote(r, c) | change, lambda: NOW)
                self.assertEqual(p['status'], 'unavailable')

    def test_close_during_capture_is_not_allocated(self):
        self.assertEqual(freeze(row(), lambda r,c: quote(r, lambda: NOW),
                                lambda: NOW+timedelta(hours=2))['status'], 'unavailable')

    def test_roi_is_absent_until_official_results(self):
        p = allocate(TICKETS, dict.fromkeys(TICKETS, 20))
        s = settle(p, TICKETS, None)
        self.assertEqual(s['status'], 'pending')
        self.assertIsNone(s['roi'])
        self.assertIsNone(s['payout_yen'])

    def test_actual_payout_not_prediction_odds(self):
        p = allocate(TICKETS, dict.fromkeys(TICKETS, 20))
        before = copy.deepcopy(p)
        s = settle(p, TICKETS, result(pays={TICKETS[0]: 1000}))
        actual = p['allocations'][0]['stake_yen'] // 100 * 1000
        self.assertEqual(s['payout_yen'], actual)
        self.assertEqual(s['roi'], actual/10000)
        self.assertLess(s['profit_yen'], 0)
        self.assertEqual(before, p)

    def test_loss_costs_ten_thousand(self):
        p = allocate(TICKETS, dict.fromkeys(TICKETS, 20))
        s = settle(p, TICKETS, result(['7-6-5'], {'7-6-5': 3000}))
        self.assertEqual((s['hit'], s['payout_yen'], s['profit_yen'], s['roi']),
                         (False, 0, -10000, 0))

    def test_tied_winners_each_use_their_own_stake_and_official_price(self):
        p = allocate(TICKETS, dict.fromkeys(TICKETS, 20))
        pays = dict(zip(TICKETS[:2], (1100, 1500)))
        s = settle(p, TICKETS, result(TICKETS[:2], pays))
        expected = sum(x['stake_yen']//100*pays[x['buy']] for x in p['allocations'][:2])
        self.assertEqual(s['payout_yen'], expected)
        self.assertIsNone(settle(p, TICKETS, result(TICKETS[:2], pays, False))['roi'])

    def test_missing_payout_refund_and_corrupt_snapshot_remain_visible(self):
        p = allocate(TICKETS, dict.fromkeys(TICKETS, 20))
        self.assertEqual(settle(p, TICKETS, result(complete=False))['status'], 'payout_pending')
        self.assertEqual(settle(p, TICKETS, result(), ['1'])['status'], 'refund_pending')
        bad = copy.deepcopy(p); bad['allocations'][0]['stake_yen'] += 100
        self.assertEqual(settle(bad, TICKETS, result())['status'], 'invalid_snapshot')
        self.assertEqual(settle(p, list(reversed(TICKETS)), result())['status'], 'invalid_snapshot')
        self.assertEqual(settle(unavailable('none'), TICKETS, result())['status'], 'not_purchased')


if __name__ == '__main__':
    unittest.main()
