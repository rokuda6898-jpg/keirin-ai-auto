"""Tests for independent, observed-trifecta market-first axis shadow."""
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from itertools import permutations
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from market_axis_shadow import first_from_quotes, freeze_market_axis, build_market_report


class MarketAxisShadowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.now = datetime(2026, 10, 8, 11, 30, tzinfo=ZoneInfo("Asia/Tokyo"))
        self.close = (self.now + timedelta(minutes=15)).timestamp()
        self.race = pd.DataFrame([
            {"race_id": "r1", "car_no": car, "close_at": self.close,
             "venue": "test", "race_no": 1} for car in (1, 2, 3)
        ])
        stamp = (self.now - timedelta(seconds=80)).isoformat()
        self.quotes = pd.DataFrame([
            {"race_id": "r1", "bet_type": "trifecta",
             "buy": "-".join(map(str, p)),
             "odds_used": 10.0 if p[0] == 3 else 100.0,
             "odds_captured_at_jst": stamp,
             "odds_verification_status": "verified"}
            for p in permutations([1, 2, 3], 3)
        ])
        self.ref = {"r1": {"variants": {"current_hole": 1, "six_department_consensus": 2},
                           "snapshot_at": (self.now-timedelta(minutes=1)).isoformat()}}
    def tearDown(self):
        self.temp.cleanup()

    def test_full_observed_market_selects_independent_axis(self):
        self.assertEqual(first_from_quotes(self.race, self.quotes, self.now, self.close), 3)
        frozen = freeze_market_axis(self.race, self.quotes, self.now, self.ref, self.root)
        self.assertEqual(len(frozen), 1)
        self.assertEqual(frozen[0]["market_first"], 3)
        self.assertFalse(frozen[0]["purchase_authorized"])
        self.assertEqual(frozen[0]["current_hole"], 1)
        self.assertEqual(frozen[0]["consensus"], 2)
        again = freeze_market_axis(self.race, self.quotes, self.now, self.ref, self.root)
        self.assertEqual(again, [])
        self.assertEqual(len((self.root / "company/market_first_shadow_ledger.jsonl").read_text().splitlines()), 1)

    def test_missing_stale_or_postclose_quotes_cannot_create_forecasts(self):
        old = self.quotes.copy()
        old.loc[:, "odds_captured_at_jst"] = (self.now-timedelta(minutes=6)).isoformat()
        self.assertIsNone(first_from_quotes(self.race, old, self.now, self.close))
        missing = self.quotes.iloc[:-1]
        self.assertIsNone(first_from_quotes(self.race, missing, self.now, self.close))
        late = self.quotes.copy()
        late.loc[:, "odds_captured_at_jst"] = (self.now+timedelta(minutes=1)).isoformat()
        self.assertIsNone(first_from_quotes(self.race, late, self.now, self.close))
        self.assertEqual(freeze_market_axis(self.race, self.quotes,
                         self.now+timedelta(minutes=12), self.ref, self.root), [])
        self.assertFalse((self.root / "company/market_first_shadow_ledger.jsonl").exists())

    def test_only_official_results_settle_and_old_result_survives_bad_feed(self):
        freeze_market_axis(self.race, self.quotes, self.now, self.ref, self.root)
        result_path = self.root / "latest_results.json"
        result_path.write_text(json.dumps([{"race_id":"r1",
           "actual_trifecta":"3-1-2","official_result_available":False}]))
        self.assertEqual(build_market_report(self.root)["settled_races"], 0)
        result_path.write_text(json.dumps([{"race_id":"r1",
           "actual_trifecta":"3-1-2","official_result_available":True}]))
        report = build_market_report(self.root)
        self.assertEqual(report["settled_races"], 1)
        self.assertEqual(report["variants"]["market_first"]["winner_hits"], 1)
        self.assertEqual(report["variants"]["current_hole"]["winner_hits"], 0)
        self.assertFalse(report["purchase_authorized"])
        self.assertFalse(report["automatic_promotion"])
        result_path.write_text("{broken")
        self.assertEqual(build_market_report(self.root)["settled_races"], 1)
        self.assertTrue((self.root / "company/market_first_shadow_report.html").exists())


if __name__ == "__main__":
    unittest.main()
