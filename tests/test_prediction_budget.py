import unittest

import pandas as pd

from predict import allocate_daily_budget


class PredictionBudgetTests(unittest.TestCase):
    def test_allocates_by_edge_and_confidence_within_daily_cap(self):
        bets = pd.DataFrame(
            {
                "prob": [0.8, 0.4, 0.2],
                "expected_profit_100yen": [1000, 500, 300],
            }
        )

        allocated = allocate_daily_budget(bets, budget_yen=1500, max_per_bet_yen=1000)

        stakes = allocated["stake_yen"]
        self.assertLessEqual(int(stakes.sum()), 1500)
        self.assertTrue(stakes.ge(100).all())
        self.assertTrue(stakes.mod(100).eq(0).all())
        self.assertLessEqual(int(stakes.max()), 1000)
        self.assertGreater(int(stakes.iloc[0]), int(stakes.iloc[1]))
        self.assertGreater(int(stakes.iloc[1]), int(stakes.iloc[2]))

    def test_zero_budget_does_not_create_subminimum_stakes(self):
        bets = pd.DataFrame({"prob": [0.8], "expected_profit_100yen": [200]})
        allocated = allocate_daily_budget(bets, budget_yen=0)
        self.assertEqual(int(allocated["stake_yen"].sum()), 0)


if __name__ == "__main__":
    unittest.main()
