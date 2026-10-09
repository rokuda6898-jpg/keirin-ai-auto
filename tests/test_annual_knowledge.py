import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from annual_knowledge import build_annual_profiles, forecast_departments, audit_department_predictions, collect_prior_record_events


class AnnualKnowledgeTests(unittest.TestCase):
    def test_live_context_changes_specialist_order_without_touching_risk(self):
        from unittest.mock import patch
        now=datetime(2026,10,6,12,tzinfo=ZoneInfo('Asia/Tokyo'))
        race=pd.DataFrame([{'race_id':'r','date':'2026-10-06','venue':'test','race_no':1,
            'player_id':str(c),'car_no':c,'p_win':.25,'p_second':.25,'p_third':.25,
            'close_at':(now+timedelta(minutes=20)).timestamp()} for c in range(1,5)])
        report={'window_end_exclusive':'2026-10-06','asof_date':'2026-10-06',
                'profiles':{},'annual_races':0}
        def context(department, frame, profiles):
            preferred={'data_department':4,'pace_department':3,'line_department':2}[department]
            return (lambda a,b,c,p: 1.2 if p==3 and c==preferred else 1), {'status':'available'}
        with patch('department_context.context_for',side_effect=context):
            changed=forecast_departments(race,pd.DataFrame(),report,now,self.root/'new')
        with patch('department_context.context_for',return_value=(None,{'status':'missing'})):
            baseline=forecast_departments(race,pd.DataFrame(),report,now,self.root/'old')
        before={r['department']:r for r in baseline};after={r['department']:r for r in changed}
        self.assertEqual(before['risk_department'],after['risk_department'])
        self.assertNotEqual(before['data_department']['top3_cars'],after['data_department']['top3_cars'])
        self.assertEqual(after['data_department']['top3_cars'][-1],4)

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
        self.assertEqual(report["profiles"]["1"]["winning_tactics"], {"逃": 3})
        self.assertEqual(report["profiles"]["2"]["tactics"], {})
        self.assertEqual(len(pd.read_csv(self.history)), 15)
        self.assertEqual(report["fingerprint"], build_annual_profiles("2026-10-06", self.history, self.root)["fingerprint"])

    def test_adaptive_period_boundaries_weights_and_historical_only_rider(self):
        rows = []
        for player, count in [("30", 30), ("29", 29), ("15", 15), ("14", 14), ("0", 0)]:
            for i in range(count):
                rows.append({"race_id": f"new{i}", "player_id": player, "date": "2026-10-05", "finish_pos": 1})
            rows.extend([
                {"race_id": "year2", "player_id": player, "date": "2024-10-06", "finish_pos": 2},
                {"race_id": "year3", "player_id": player, "date": "2023-10-06", "finish_pos": 3},
                {"race_id": "outside", "player_id": player, "date": "2023-10-05", "finish_pos": 1},
                {"race_id": "today", "player_id": player, "date": "2026-10-06", "finish_pos": 1},
                {"race_id": "future", "player_id": player, "date": "2027-10-05", "finish_pos": 1},
            ])
        pd.DataFrame(rows).to_csv(self.history, index=False)
        profiles = build_annual_profiles("2026-10-06", self.history, self.root)["profiles"]
        for player, years in [("30", 1), ("29", 2), ("15", 2), ("14", 3), ("0", 3)]:
            self.assertEqual(profiles[player]["reference_years"], years)
        score = profiles["14"]["evaluation"]
        self.assertEqual(score["races"], 16)
        self.assertEqual(score["effective_races"], 14.75)
        self.assertAlmostEqual(score["rates"][0], 14 / 14.75)
        self.assertAlmostEqual(score["rates"][1], .5 / 14.75)
        self.assertAlmostEqual(score["rates"][2], .25 / 14.75)
        self.assertEqual(profiles["0"]["races"], 0)
        self.assertEqual(profiles["0"]["evaluation"]["races"], 2)
        self.assertEqual(profiles["30"]["evaluation"]["races"], 30)
        self.assertEqual(profiles["29"]["evaluation"]["races"], 30)

    def test_invalid_finishes_do_not_trigger_shorter_window(self):
        rows = [{"race_id": f"invalid{i}", "player_id": "1", "date": "2026-10-05", "finish_pos": 99} for i in range(40)]
        rows.append({"race_id": "oldvalid", "player_id": "1", "date": "2024-10-06", "finish_pos": 1})
        pd.DataFrame(rows).to_csv(self.history, index=False)
        profile = build_annual_profiles("2026-10-06", self.history, self.root)["profiles"]["1"]
        self.assertEqual(profile["reference_years"], 3)
        self.assertEqual(profile["evaluation"]["effective_races"], .5)
        self.assertEqual(profile["evaluation"]["rates"], [1, 0, 0])

    def test_partial_supplement_does_not_inflate_archive_count(self):
        folder = self.root / "company"
        folder.mkdir()
        pd.DataFrame([{"race_id": "supplement", "player_id": "1", "date": "2026-10-04",
                       "finish_pos": 1}]).to_csv(folder / "rider_official_observations.csv", index=False)
        report = build_annual_profiles("2026-10-06", self.history, self.root)
        self.assertEqual(report["total_archive_races"], 5)
        self.assertEqual(report["supplemental_result_races"], 1)
        self.assertEqual(report["annual_archive_races"], 2)
        self.assertEqual(report["annual_races"], 3)

    def test_behavior_associations_use_only_observed_flags(self):
        frame = pd.read_csv(self.history)
        frame["result_event_back"] = None
        frame.loc[frame.player_id.eq(1) & frame.race_id.eq("recent"), "result_event_back"] = True
        frame.to_csv(self.history, index=False)
        report = build_annual_profiles("2026-10-06", self.history, self.root)
        event = report["profiles"]["1"]["events"]["back"]
        self.assertEqual(event["observed"], 1)
        self.assertEqual(event["true_results"]["races"], 1)
        self.assertEqual(event["true_results"]["rates"], [1, 0, 0])
        self.assertEqual(report["profiles"]["2"]["events"]["back"]["observed"], 0)

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
        self.assertTrue(all(f["rider_reference"]["1"]["years"] == 3 for f in forecasts))
        self.assertTrue(all(f["rider_reference"]["1"]["effective_races"] == 2.5 for f in forecasts))
        self.assertTrue(all(not f["purchase_authorized"] for f in forecasts))
        (self.root / "latest_results.json").write_text(json.dumps([{"race_id": "target",
            "official_result_available": True, "actual_trifecta": "1-2-3", "actual_trifecta_odds": 10}]))
        audit = audit_department_predictions(self.root)
        self.assertTrue(all(d["races"] == 1 for d in audit["departments"]))
        (self.root / "latest_results.json").write_text("[]")
        self.assertTrue(all(d["races"] == 1 for d in audit_department_predictions(self.root)["departments"]))
        # Closed races still receive an explicit per-department status, but
        # never a new hindsight forecast. Previously frozen forecasts remain.
        closed = forecast_departments(race, odds, report, now + timedelta(hours=1), self.root)
        self.assertEqual(len(closed), 4)
        self.assertTrue(all(item.get("display_status") == "preclose_forecast_preserved"
                            for item in closed))
        report['asof_date']='2026-10-07'
        report['window_end_exclusive']='2026-10-07'
        with self.assertRaisesRegex(ValueError,'cutoff'):
            forecast_departments(race,odds,report,now,self.root)

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
