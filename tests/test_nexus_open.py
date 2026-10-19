import copy
import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nexus_model import open_formation


def expanded(blocks):
    return [t for b in blocks for t in itertools.product(b['first'], b['second'], b['third'])
            if len(set(t)) == 3]


def plan(blocks):
    for block in blocks:
        block['count'] = len(expanded([block]))
    return dict(blocks=blocks, tickets=[list(t) for t in sorted(expanded(blocks))], search='original_fixture')


def fixed_first(cars, cap):
    a, b, c, d, e, f, g = cars[:7]
    if cap == 6:
        return plan([dict(first=[a], second=[b, c], third=[b, c, d, e])])
    if cap == 15:
        return plan([dict(first=[a], second=[b, c, d], third=[b, c, d, e, f, g])])
    return plan([dict(first=[a], second=[b, c, d], third=[b, c, d, e, f, g]),
                 dict(first=[a], second=[e, f, g], third=[b, e, f, g])])


def already_open(cars, cap):
    a, b, c, d, e, f, g = cars[:7]
    if cap == 6:
        return plan([dict(first=[a, b], second=[a, b], third=[c, d, e])])
    if cap == 15:
        return plan([dict(first=[a], second=[b, c], third=[b, c, d, e]),
                     dict(first=[b], second=[a, c], third=[a, c, d, e]),
                     dict(first=[c], second=[a], third=[b, d, e])])
    return plan([dict(first=[a], second=[b, c, d], third=[b, c, d, e, f, g]),
                 dict(first=[b], second=[a, c, d], third=[a, c, d, e])])


class OpenFormationContract(unittest.TestCase):
    def check_contract(self, cars, cap, result):
        tickets = expanded(result['blocks'])
        self.assertEqual(len(tickets), cap)
        self.assertEqual(len(set(tickets)), cap)
        self.assertEqual(set(tickets), set(map(tuple, result['tickets'])))
        self.assertLessEqual(len(result['blocks']), 4)
        self.assertTrue(all(set(t) <= set(cars) for t in tickets))
        self.assertGreaterEqual(len({t[0] for t in tickets}), 1)
        self.assertGreaterEqual(len({t[1] for t in tickets}), 2)
        self.assertEqual(result['policy_id'], 'flexible-first-v1')
        self.assertTrue(all(len(expanded([b])) == b['count'] for b in result['blocks']))

    def concentrated_scores(self, cars, position):
        triples = list(itertools.permutations(cars, 3))
        q = np.array([1000.0 if t[position] == cars[0] else 1.0 + i / 10000
                      for i, t in enumerate(triples)])
        q /= q.sum()
        q.setflags(write=False)
        return q

    def test_fixed_or_flexible_first_is_preserved_for_seven_and_nine_riders(self):
        for n in (7, 9):
            cars = list(range(1, n + 1))
            for cap in (6, 15, 24):
                for first_fixed in (False, True):
                    with self.subTest(riders=n, cap=cap, first_fixed=first_fixed):
                        original = fixed_first(cars, cap) if first_fixed else already_open(cars, cap)
                        snapshot = copy.deepcopy(original)
                        q = self.concentrated_scores(cars, 0)
                        before = q.copy()
                        result = open_formation.select(cars, q, cap, original)
                        self.check_contract(cars, cap, result)
                        self.assertFalse(result['plan_changed'])
                        self.assertIs(result['blocks'], original['blocks'])
                        self.assertIs(result['tickets'], original['tickets'])
                        self.assertEqual(result['search'], original['search'])
                        self.assertEqual(original, snapshot)
                        if first_fixed:
                            self.assertEqual({t[0] for t in result['tickets']}, {cars[0]})
                        np.testing.assert_array_equal(q, before)

    def test_second_concentration_opens_all_caps_without_mutation(self):
        for n in (7, 9):
            cars = list(range(1, n + 1))
            for cap in (6, 15, 24):
                with self.subTest(riders=n, cap=cap):
                    original = fixed_first(cars, cap)
                    original = plan([dict(first=b['second'], second=b['first'], third=b['third'])
                                     for b in original['blocks']])
                    snapshot = copy.deepcopy(original)
                    q = self.concentrated_scores(cars, 1)
                    before = q.copy()
                    result = open_formation.select(cars, q, cap, original)
                    self.check_contract(cars, cap, result)
                    self.assertTrue(result['plan_changed'])
                    self.assertEqual(original, snapshot)
                    np.testing.assert_array_equal(q, before)
                    if cap == 24:
                        # Six different first riders cannot fit exact.pack's
                        # four single-first blocks; exercise the compact fallback.
                        self.assertEqual(result['search'], 'open_compact_family')

    def test_one_exchange_keeps_five_tickets_and_uses_complete_order_scores(self):
        cars = list(range(1, 8))
        original = fixed_first(cars, 6)
        original = plan([dict(first=b['second'], second=b['first'], third=b['third'])
                         for b in original['blocks']])
        existing = set(map(tuple, original['tickets']))
        removed, added = (3, 1, 5), (2, 3, 4)
        scores = {t: 0.001 for t in itertools.permutations(cars, 3)}
        scores.update({t: 10.0 for t in existing})
        scores[removed], scores[added] = 1.0, 8.0
        q = np.array(list(scores.values()))
        q /= q.sum()
        before = q.copy()
        result = open_formation.select(cars, q, 6, original)
        self.check_contract(cars, 6, result)
        self.assertEqual(set(map(tuple, result['tickets'])), existing - {removed} | {added})
        self.assertEqual(result['replacement_count'], 1)
        self.assertEqual(result['search'], 'open_one_ticket_exchange')
        np.testing.assert_array_equal(q, before)

    def test_second_can_open_while_first_remains_fixed(self):
        cars = list(range(1, 10))
        original = plan([dict(first=[1], second=[2], third=[3, 4, 5, 6, 7, 8])])
        q = np.array([1000.0 if t[:2] == (1, 2) else 10.0 if t[0] == 1 else 1.0
                      for t in itertools.permutations(cars, 3)])
        q /= q.sum()
        before = q.copy()
        result = open_formation.select(cars, q, 6, original)
        self.check_contract(cars, 6, result)
        self.assertEqual(result['replacement_count'], 1)
        self.assertEqual({t[0] for t in result['tickets']}, {1})
        np.testing.assert_array_equal(q, before)

    def test_malformed_plan_and_nonfinite_scores_are_rejected(self):
        cars = list(range(1, 8))
        q = np.ones(210) / 210
        original = already_open(cars, 6)
        original['blocks'][0]['count'] += 1
        with self.assertRaises(ValueError):
            open_formation.select(cars, q, 6, original)
        q[0] = np.nan
        with self.assertRaises(ValueError):
            open_formation.select(cars, q, 6, already_open(cars, 6))


if __name__ == '__main__':
    unittest.main()
