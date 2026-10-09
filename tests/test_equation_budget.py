import unittest
from itertools import permutations
import equation_budget as budget


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.picks = [{'buy': '-'.join(map(str,t)), 'probability': .02}
                      for t in list(permutations(range(1,5),3))[:12]]
        self.prices = {t['buy']: 30. for t in self.picks}

    def test_each_plan_is_exactly_6000_and_integer_100_yen(self):
        for n in (3,6,12):
            plan = budget.allocate(self.picks, self.prices, n)
            self.assertEqual(len(plan['tickets']), n)
            self.assertEqual(sum(t['stake_yen'] for t in plan['tickets']), 6000)
            self.assertTrue(all(t['stake_yen'] >= 100 and t['stake_yen'] % 100 == 0 for t in plan['tickets']))
            self.assertAlmostEqual(plan['estimated_roi'], .6)

    def test_uses_both_probability_and_odds(self):
        self.prices[self.picks[0]['buy']] = 100.
        plan = budget.allocate(self.picks, self.prices, 3)
        self.assertGreater(plan['tickets'][0]['stake_yen'], plan['tickets'][1]['stake_yen'])
        self.picks[1]['probability'] = .1
        plan = budget.allocate(self.picks, self.prices, 3)
        self.assertGreater(plan['tickets'][1]['stake_yen'], plan['tickets'][0]['stake_yen'])

    def test_no_fictitious_odds_or_extra_candidates(self):
        self.assertEqual(budget.allocate(self.picks, {}, 3)['spent_yen'], 0)
        self.prices[self.picks[0]['buy']] = 9999.9
        self.assertEqual(budget.allocate(self.picks, self.prices, 3)['spent_yen'], 0)
        self.assertEqual(budget.allocate(self.picks[:2], {}, 3)['status'], 'candidate_shortage')

    def test_discrete_allocation_matches_exhaustive_three_ticket_optimum(self):
        import math
        self.picks[0]['probability'] = .07
        odds = [12., 30., 100.]
        prices = {t['buy']:o for t,o in zip(self.picks, odds)}
        plan = budget.allocate(self.picks, prices, 3)
        objective = lambda s: sum(t['probability']*math.log(6000+o*v) for t,o,v in zip(self.picks,odds,s))
        best = max(objective([100*a,100*b,100*(60-a-b)]) for a in range(1,59) for b in range(1,60-a))
        self.assertAlmostEqual(objective([t['stake_yen'] for t in plan['tickets']]), best)


if __name__ == '__main__':
    unittest.main()
