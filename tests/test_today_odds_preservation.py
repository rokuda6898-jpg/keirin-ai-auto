import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fetch_today_entries import merge_existing_same_day_odds


class TodayOddsPreservationTests(unittest.TestCase):
    def test_empty_refresh_keeps_same_day_odds_only(self):
        entries = pd.DataFrame([{"date": "2026-10-02", "race_id": "r1"}])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "today_odds.csv"
            pd.DataFrame([
                {"date": "2026-10-02", "race_id": "r1", "bet_type": "trifecta", "buy": "1-2-3", "odds_used": 25.0},
                {"date": "2026-10-01", "race_id": "old", "bet_type": "trifecta", "buy": "1-2-3", "odds_used": 99.0},
            ]).to_csv(path, index=False)
            merged = merge_existing_same_day_odds(entries, pd.DataFrame(), path)
        self.assertEqual(merged["race_id"].tolist(), ["r1"])

    def test_partial_refresh_replaces_only_refreshed_race(self):
        entries = pd.DataFrame([
            {"date": "2026-10-02", "race_id": "r1"},
            {"date": "2026-10-02", "race_id": "r2"},
        ])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "today_odds.csv"
            pd.DataFrame([
                {"date": "2026-10-02", "race_id": "r1", "bet_type": "trifecta", "buy": "1-2-3", "odds_used": 20.0},
                {"date": "2026-10-02", "race_id": "r2", "bet_type": "trifecta", "buy": "2-1-3", "odds_used": 30.0},
            ]).to_csv(path, index=False)
            fresh = pd.DataFrame([
                {"date": "2026-10-02", "race_id": "r2", "bet_type": "trifecta", "buy": "2-1-3", "odds_used": 35.0},
            ])
            merged = merge_existing_same_day_odds(entries, fresh, path)
        by_race = dict(zip(merged["race_id"], merged["odds_used"]))
        self.assertEqual(by_race["r1"], 20.0)
        self.assertEqual(by_race["r2"], 35.0)


if __name__ == "__main__":
    unittest.main()
