import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from annual_knowledge import build_annual_profiles, forecast_departments, audit_department_predictions, collect_prior_record_events


class AnnualKnowledgeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.history = self.root / "history.csv"
        rows = []
        for rid, date in [("old", "2025-10-05"), ("boundary", "2025-10-06"),
                          ("recent", "2026-10-05"), ("today", "2026-10-06"), ("future", "2026-10-07")]:
            for car in [1, 2, 3]:
                rows.append({"race_id": rid, "player_id": str(car), "date": date,
                             "car_no": car, "finish_pos": car, "entries_number": 3,
                             "line_position": car, "style": "逃", "result_factor": "逃" if car == 1 else None})
        pd.DataFrame(rows).to_csv(self.history, index=False)

    def test_one_year_boundary_no_today_or_future_and_no_archive_deletion(self):
        report = build_annual_profiles("2026-10-06", self.history, self.root)
        self.assertEqual(report["total_archive_races"], 5)
        self.assertEqual(report["annual_races"], 2)
        self.assertEqual(report["profiles"]["1"]["races"], 2)
        self.assertEqual(report["profiles"]["1"]["winning_tactics"], {"逃": 2})
        self.assertEqual(report["profiles"]["2"]["tactics"], {})
        self.assertEqual(len(pd.read_csv(self.history)), 15)
        self.assertEqual(report["fingerprint"], build_annual_profiles("2026-10-06", self.history, self.root)["fingerprint"])

    def test_history_change_invalidates_profile_cache(self):
        before = build_annual_profiles("2026-10-06", self.history, self.root)
        frame = pd.read_csv(self.history); frame.loc[frame.race_id.eq("recent"), "finish_pos"] = [3, 2, 1]
        frame.to_csv(self.history, index=False)
        after = build_annual_profiles("2026-10-06", self.history, self.root)
        self.assertNotEqual(before["fingerprint"], after["fingerprint"])
        self.assertEqual(after["profiles"]["1"]["rates"][0], .5)

    def test_each_department_forecasts_and_official_settlement_is_durable(self):
        report = build_annual_profiles("2026-10-06", self.history, self.root)
        now = datetime(2026, 10, 6, 12, tzinfo=ZoneInfo("Asia/Tokyo"))
        race = pd.DataFrame([{"race_id": "target", "date": "2026-10-06", "venue": "test", "race_no": 1,
                             "player_id": str(car), "car_no": car, "p_win": 1/3, "line_position": car,
                             "close_at": (now + timedelta(minutes=20)).timestamp()} for car in [1, 2, 3]])
        odds = pd.DataFrame([{"race_id": "target", "bet_type": "trifecta", "buy": "1-2-3", "odds_used": 100}])
        forecasts = forecast_departments(race, odds, report, now, self.root)
        self.assertEqual(len(forecasts), 4)
        self.assertTrue(all(not f["purchase_authorized"] for f in forecasts))
        (self.root / "latest_results.json").write_text(json.dumps([{"race_id": "target",
            "official_result_available": True, "actual_trifecta": "1-2-3", "actual_trifecta_odds": 10}]))
        audit = audit_department_predictions(self.root)
        self.assertTrue(all(d["races"] == 1 for d in audit["departments"]))
        (self.root / "latest_results.json").write_text("[]")
        self.assertTrue(all(d["races"] == 1 for d in audit_department_predictions(self.root)["departments"]))
        self.assertEqual(forecast_departments(race, odds, report, now + timedelta(hours=1), self.root), [])

    def test_observed_past_events_exclude_today_and_preserve_missing_fields(self):
        from unittest.mock import patch
        data = {"records": [{"playerId": "1", "latestCupResults": [{"raceResults": [
            {"raceId": "011320261005", "playerId": "1", "order": 1, "factor": "逃", "splitLine": True},
            {"raceId": "011320261006", "playerId": "1", "order": 1, "factor": "差"}]}]}]}
        fixed = datetime(2026, 10, 6, 12, tzinfo=ZoneInfo("Asia/Tokyo"))
        with patch("annual_knowledge.datetime") as clock:
            clock.now.return_value = fixed
            count = collect_prior_record_events(data, "https://www.winticket.jp/test", self.root)
        self.assertEqual(count, 1)
        observed = pd.read_csv(self.root / "company/rider_official_observations.csv")
        self.assertEqual(observed.result_factor.iloc[0], "逃")
        self.assertTrue(observed.result_event_splitLine.iloc[0])
        self.assertTrue(observed.result_event_spurtSucceeded.isna().all())
