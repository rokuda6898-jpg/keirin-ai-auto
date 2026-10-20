"""Focused invariants for the bounded research selectors and settlement."""
from collections import defaultdict
from itertools import permutations
import math
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

from recent_candidate_compare_v1 import (
    BASE, FULL_POLICIES, selected_full, selected_saved, settle,
    third_tempered_scores, ticket_tuple,
)


class CandidateTests(unittest.TestCase):
    def setUp(self):
        self.keys = ['-'.join(map(str,t)) for t in permutations(range(1, 6), 3)]
        raw = {ticket: 1/(i+1) for i,ticket in enumerate(self.keys)}
        denominator = math.fsum(raw.values())
        self.q = {ticket:value/denominator for ticket,value in raw.items()}

    def test_fixed_count_and_no_duplicate_for_all_full_q_policies(self):
        for policy in FULL_POLICIES:
            with self.subTest(policy=policy):
                selected = selected_full(self.q, policy)
                self.assertEqual(len(selected), 12)
                self.assertEqual(len(set(selected)), 12)
                self.assertTrue(set(selected).issubset(self.q))

    def test_pair_cap_limits_concentration(self):
        counts = defaultdict(int)
        for ticket in selected_full(self.q, 'pair_cap3_top12'):
            counts[ticket_tuple(ticket)[:2]] += 1
        self.assertLessEqual(max(counts.values()), 3)
        self.assertGreaterEqual(len(counts), 4)

    def test_tempering_preserves_every_first_second_pair_mass(self):
        scores = third_tempered_scores(self.q)
        pair_before, pair_after = defaultdict(float), defaultdict(float)
        for ticket in self.q:
            pair = ticket_tuple(ticket)[:2]
            pair_before[pair] += self.q[ticket]
            pair_after[pair] += scores[ticket]
        for pair in pair_before:
            self.assertAlmostEqual(pair_before[pair], pair_after[pair], places=14)
        self.assertAlmostEqual(sum(scores.values()), 1.0, places=14)
        a, b = self.keys[:2]
        self.assertEqual(ticket_tuple(a)[:2], ticket_tuple(b)[:2])
        self.assertAlmostEqual(scores[a]/scores[b], math.sqrt(self.q[a]/self.q[b]), places=13)

    def test_tempering_is_identity_for_uniform_third_conditionals(self):
        q = {ticket:1/len(self.keys) for ticket in self.keys}
        for ticket, score in third_tempered_scores(q).items():
            self.assertAlmostEqual(score, q[ticket], places=15)

    def test_two_boxes_are_complete_permutations_not_fixed_orders(self):
        selected = selected_full(self.q, 'two_triple_boxes12')
        groups = defaultdict(set)
        for ticket in selected:
            triple = ticket_tuple(ticket)
            groups[tuple(sorted(triple))].add(triple)
        self.assertEqual(len(groups), 2)
        for group, values in groups.items():
            self.assertEqual(values, set(permutations(group)))

    def test_prefix_retention_uses_original_mass_and_minimum(self):
        tickets = self.keys[:12]
        probabilities = [.07]*12
        self.assertEqual(selected_saved(tickets, probabilities, 'retain90_min6'), tickets[:11])
        concentrated = [.70] + [.01]*11
        self.assertEqual(selected_saved(tickets, concentrated, 'retain90_min6'), tickets[:6])
        self.assertEqual(selected_saved(tickets, probabilities, 'top10'), tickets[:10])
        self.assertEqual(selected_saved(tickets, probabilities, BASE), tickets)

    def test_official_multiple_winners_sum_exact_yen_but_hit_is_one_race(self):
        result = settle(['1-2-3','1-3-2','2-1-3'],
                        {'winning_buys':['1-2-3','1-3-2'], 'payouts':{'1-2-3':'1270','1-3-2':1600.0}})
        self.assertEqual(result['invested_yen'], 300)
        self.assertEqual(result['payout_yen'], 2870)
        self.assertIs(result['hit'], True)
        self.assertIsInstance(result['payout_yen'], int)
        self.assertFalse(result['torigami'])

    def test_incomplete_or_fractional_official_payout_is_not_scored(self):
        with self.assertRaises(KeyError):
            settle(['1-2-3'], {'winning_buys':['1-2-3','1-3-2'], 'payouts':{'1-2-3':1000}})
        with self.assertRaises(AssertionError):
            settle(['1-2-3'], {'winning_buys':['1-2-3'], 'payouts':{'1-2-3':'1000.1'}})


if __name__ == '__main__':
    unittest.main()
