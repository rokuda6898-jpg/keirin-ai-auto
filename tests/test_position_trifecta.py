import unittest

import pandas as pd

from multi_bet_backtest import pl3
from predict import filter_trifecta_candidates_by_confidence


class PositionTrifectaTests(unittest.TestCase):
    def test_pl3_uses_position_specific_probabilities(self):
        a = {"p_win": 0.50, "p_second": 0.10, "p_third": 0.10}
        b = {"p_win": 0.25, "p_second": 0.60, "p_third": 0.20}
        c = {"p_win": 0.15, "p_second": 0.20, "p_third": 0.60}
        d = {"p_win": 0.10, "p_second": 0.10, "p_third": 0.10}
        self.assertGreater(pl3(a, b, c), pl3(a, c, b))
        self.assertGreater(pl3(a, b, c), pl3(b, a, c))

    def test_portfolio_uses_second_and_third_specialist_ranks(self):
        race = pd.DataFrame({
            "car_no": [1, 2, 3, 4, 5, 6, 7],
            "p_win": [0.45, 0.20, 0.12, 0.09, 0.06, 0.05, 0.03],
            "p_second": [0.05, 0.10, 0.15, 0.20, 0.25, 0.20, 0.05],
            "p_third": [0.04, 0.06, 0.10, 0.15, 0.20, 0.25, 0.20],
        })
        rows = []
        for a in range(1, 8):
            for b in range(1, 8):
                for c in range(1, 8):
                    if len({a, b, c}) == 3:
                        rows.append({"bet_type": "trifecta", "buy": f"{a}-{b}-{c}", "prob": 0.01})
        out = filter_trifecta_candidates_by_confidence(pd.DataFrame(rows), race)
        tri = out[out["bet_type"].eq("trifecta")]

        seconds = set(tri["buy"].str.split("-").str[1].astype(int))
        thirds = set(tri["buy"].str.split("-").str[2].astype(int))
        self.assertTrue(seconds.issubset({3, 4, 5, 6}))
        self.assertTrue(thirds.issubset({2, 3, 4, 5, 6, 7}))


if __name__ == "__main__":
    unittest.main()
