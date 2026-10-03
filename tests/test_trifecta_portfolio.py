import unittest

import pandas as pd

from predict import filter_trifecta_candidates_by_confidence


def trifecta_candidates(cars):
    rows = []
    for a in cars:
        for b in cars:
            for c in cars:
                if len({a, b, c}) == 3:
                    rows.append({"bet_type": "trifecta", "buy": f"{a}-{b}-{c}", "prob": 0.01})
    rows.append({"bet_type": "exacta", "buy": f"{cars[0]}-{cars[1]}", "prob": 0.1})
    return pd.DataFrame(rows)


class TrifectaPortfolioTests(unittest.TestCase):
    def race(self, probs):
        return pd.DataFrame({
            "car_no": list(range(1, len(probs) + 1)),
            "p_win": probs,
        })

    def test_clear_race_keeps_one_head(self):
        race = self.race([0.60, 0.15, 0.10, 0.07, 0.05, 0.03])
        out = filter_trifecta_candidates_by_confidence(trifecta_candidates(range(1, 7)), race)
        tri = out[out["bet_type"].eq("trifecta")]
        self.assertEqual(set(tri["buy"].str.split("-").str[0]), {"1"})
        self.assertTrue(tri["trifecta_portfolio_mode"].eq("clear_1head").all())

    def test_balanced_race_keeps_two_heads(self):
        race = self.race([0.38, 0.25, 0.14, 0.09, 0.08, 0.06])
        out = filter_trifecta_candidates_by_confidence(trifecta_candidates(range(1, 7)), race)
        tri = out[out["bet_type"].eq("trifecta")]
        self.assertEqual(set(tri["buy"].str.split("-").str[0]), {"1", "2"})
        self.assertTrue(tri["trifecta_portfolio_mode"].eq("balanced_2head").all())

    def test_near_tie_keeps_three_heads(self):
        race = self.race([0.25, 0.23, 0.18, 0.13, 0.11, 0.10])
        out = filter_trifecta_candidates_by_confidence(trifecta_candidates(range(1, 7)), race)
        tri = out[out["bet_type"].eq("trifecta")]
        self.assertEqual(set(tri["buy"].str.split("-").str[0]), {"1", "2", "3"})
        self.assertTrue(tri["trifecta_portfolio_mode"].eq("near_tie_3head").all())

    def test_second_and_third_breadth_are_limited(self):
        race = self.race([0.38, 0.25, 0.14, 0.09, 0.08, 0.04, 0.02])
        out = filter_trifecta_candidates_by_confidence(trifecta_candidates(range(1, 8)), race)
        tri = out[out["bet_type"].eq("trifecta")]
        seconds = set(tri["buy"].str.split("-").str[1])
        thirds = set(tri["buy"].str.split("-").str[2])
        self.assertTrue(seconds.issubset({"1", "2", "3", "4"}))
        self.assertTrue(thirds.issubset({"1", "2", "3", "4", "5", "6"}))

    def test_non_trifecta_candidates_are_preserved(self):
        race = self.race([0.60, 0.15, 0.10, 0.07, 0.05, 0.03])
        out = filter_trifecta_candidates_by_confidence(trifecta_candidates(range(1, 7)), race)
        self.assertEqual(int(out["bet_type"].eq("exacta").sum()), 1)


if __name__ == "__main__":
    unittest.main()
