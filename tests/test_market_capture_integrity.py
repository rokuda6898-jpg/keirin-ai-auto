"""Regression coverage for market ingestion timing, diagnostics and withdrawals."""
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from itertools import permutations
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common
import fetch_upcoming_entries as upcoming
import market_axis_shadow as market

JST = ZoneInfo("Asia/Tokyo")


class MarketCaptureIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.now = datetime(2026, 10, 8, 15, 30, tzinfo=JST)
        self.close = (self.now + timedelta(minutes=15)).timestamp()
        self.ref = {"000001": {"snapshot_at": (self.now - timedelta(minutes=1)).isoformat(),
                    "variants": {"current_hole": 1, "six_department_consensus": 2}}}
        self.race, self.odds = self.frames(range(1, 8))

    def tearDown(self):
        self.temp.cleanup()

    def frames(self, cars):
        cars = list(cars)
        race = pd.DataFrame([{"race_id": "000001", "car_no": c,
                             "close_at": self.close, "venue": "test"} for c in cars])
        stamp = (self.now - timedelta(seconds=80)).isoformat()
        odds = pd.DataFrame([{"race_id": "000001", "bet_type": "trifecta",
                             "buy": "-".join(map(str, p)),
                             "odds_used": 10.0 if p[0] == 3 else 100.0,
                             "odds_captured_at_jst": stamp,
                             "odds_verification_status": "verified"}
                            for p in permutations(cars, 3)])
        return race, odds

    def audit(self):
        return json.loads((self.root / "company" / market.CAPTURE_STATUS).read_text())

    def test_complete_seven_nine_and_withdrawn_fields(self):
        for cars, expected in ((range(1, 8), 210), (range(1, 10), 504), ([1, 3, 4, 5, 6, 7], 120)):
            with self.subTest(cars=list(cars)):
                race, odds = self.frames(cars)
                axis, reason, detail = market.inspect_quotes(race, odds, self.now, self.close)
                self.assertEqual((axis, reason), (3, "eligible"))
                self.assertEqual(detail["expected_quotes"], expected)
                self.assertEqual(detail["oldest_quote_age_seconds"], 80)

    def test_distinct_rejection_reasons_are_reported(self):
        cases = []
        cases.append((pd.DataFrame(), "odds_empty"))
        cases.append((self.odds.drop(columns="odds_captured_at_jst"), "odds_schema_missing"))
        cases.append((self.odds.iloc[:-1], "incomplete_quotes"))
        cases.append((pd.concat([self.odds, self.odds.iloc[:1]]), "duplicate_quotes"))
        old = self.odds.copy()
        old["odds_captured_at_jst"] = (self.now - timedelta(seconds=301)).isoformat()
        cases.append((old, "stale_quotes"))
        future = self.odds.copy()
        future["odds_captured_at_jst"] = (self.now + timedelta(seconds=1)).isoformat()
        cases.append((future, "future_capture_time"))
        excluded = self.odds.copy()
        excluded["odds_verification_status"] = "excluded_time_or_source"
        cases.append((excluded, "excluded_quotes"))
        invalid = self.odds.copy()
        invalid.loc[0, "odds_used"] = float("inf")
        cases.append((invalid, "invalid_prices"))
        for quotes, reason in cases:
            with self.subTest(reason=reason):
                self.assertEqual(market.freeze_market_axis(
                    self.race, quotes, self.now, self.ref, self.root), [])
                self.assertEqual(self.audit()["reason_counts"], {reason: 1})
                self.assertFalse((self.root / "company" / market.LEDGER).exists())

    def test_reference_missing_and_invalid_close_are_not_silent(self):
        market.freeze_market_axis(self.race, self.odds, self.now, {}, self.root)
        self.assertEqual(self.audit()["reason_counts"], {"reference_missing": 1})
        bad = self.race.copy()
        bad["close_at"] = float("nan")
        market.freeze_market_axis(bad, self.odds, self.now, self.ref, self.root)
        self.assertEqual(self.audit()["reason_counts"], {"invalid_close_time": 1})
        bad.loc[0, "close_at"] = self.close
        market.freeze_market_axis(bad, self.odds, self.now, self.ref, self.root)
        self.assertEqual(self.audit()["reason_counts"], {"inconsistent_close_time": 1})

    def test_capture_before_slow_model_work_stays_immutable(self):
        with patch("department_coverage._frozen_axis_experiments", return_value=self.ref):
            rows = market.capture_after_fetch(self.race, self.odds, self.root, now=self.now)
        self.assertEqual(len(rows), 1)
        path = self.root / "company" / market.LEDGER
        original = path.read_bytes()
        later = self.now + timedelta(minutes=6)
        self.assertEqual(market.first_from_quotes(self.race, self.odds, later, self.close), None)
        self.assertEqual(market.freeze_market_axis(self.race, self.odds, later, self.ref, self.root), [])
        self.assertEqual(path.read_bytes(), original)
        row = json.loads(original.decode().splitlines()[0])
        self.assertEqual(row["snapshot_at"], self.now.isoformat(timespec="seconds"))
        self.assertEqual(row["capture_phase"], "ingestion")
        self.assertEqual(self.audit()["reason_counts"], {"already_saved": 1})

    def test_diagnostic_run_cannot_create_prediction_evidence(self):
        self.assertEqual(market.freeze_market_axis(self.race, self.odds, self.now,
            self.ref, self.root, phase="diagnostic", dry_run=True), [])
        self.assertEqual(self.audit()["saved_this_run"], 0)
        self.assertEqual(self.audit()["reason_counts"], {"eligible_not_saved": 1})
        self.assertFalse((self.root / "company" / market.LEDGER).exists())

    def test_malformed_journal_is_not_overwritten(self):
        folder = self.root / "company"
        folder.mkdir()
        path = folder / market.LEDGER
        path.write_text("{broken")
        self.assertEqual(market.freeze_market_axis(self.race, self.odds,
            self.now, self.ref, self.root), [])
        self.assertEqual(path.read_text(), "{broken")
        self.assertEqual(self.audit()["reason_counts"], {"invalid_existing_ledger": 1})

    def test_market_report_exposes_capture_audit(self):
        market.freeze_market_axis(self.race, pd.DataFrame(), self.now, self.ref, self.root)
        report = market.build_market_report(self.root)
        self.assertEqual(report["capture_audits"]["prediction"]["reason_counts"], {"odds_empty": 1})
        self.assertEqual(report["frozen_races"], 0)
        self.assertFalse(report["purchase_authorized"])


class UpcomingCaptureIntegrityTests(unittest.TestCase):
    def test_declared_withdrawals_are_complete_not_contiguous(self):
        rows = [{"race_id": "r", "car_no": c, "entries_number": 6,
                 "declared_entries_number": 7, "cancelled_car_numbers": "0,2"}
                for c in (1, 3, 4, 5, 6, 7)]
        self.assertTrue(upcoming._entries_complete(pd.DataFrame(rows), "r"))
        self.assertFalse(upcoming._entries_complete(pd.DataFrame(rows[:-1]), "r"))
        self.assertFalse(upcoming._entries_complete(pd.DataFrame(rows + [rows[0]]), "r"))
        full = pd.DataFrame([{"race_id": "r", "car_no": c, "entries_number": 9} for c in range(1, 10)])
        self.assertTrue(upcoming._entries_complete(full, "r"))
        self.assertFalse(upcoming._entries_complete(full.iloc[:-1], "r"))

    def test_source_retry_is_bounded_without_overwriting_daily_snapshot(self):
        now = datetime.now(JST)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source = root / "today.csv"
            source.write_text("original-data")
            schedule = root / "schedule.csv"
            pd.DataFrame([{"race_id": "r", "date": now.strftime("%Y-%m-%d"),
                "close_at": (now + timedelta(minutes=25)).timestamp(), "source_url": "test"}]).to_csv(schedule, index=False)
            with patch.multiple(upcoming, RACE_SCHEDULE_CSV=schedule,
                                UPCOMING_COUNT_FILE=root / "count.txt"), \
                 patch.object(upcoming, "ensure_dirs"), \
                 patch.object(upcoming, "parse_race_page", side_effect=ValueError("unavailable")) as parse, \
                 patch.object(upcoming.time, "sleep"), \
                 patch.object(upcoming, "save_today_frames") as save:
                with self.assertRaisesRegex(ValueError, "failed to fetch"):
                    upcoming.fetch_upcoming()
            self.assertEqual(parse.call_count, upcoming.MAX_RACE_FETCH_ATTEMPTS)
            save.assert_not_called()
            self.assertEqual(source.read_text(), "original-data")

    def test_market_capture_occurs_before_full_day_save(self):
        now = datetime.now(JST)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            close = (now + timedelta(minutes=25)).timestamp()
            rows = [{"race_id": "r", "car_no": c, "entries_number": 3,
                     "close_at": close, "date": now.strftime("%Y-%m-%d")}
                    for c in (1, 2, 3)]
            source, quotes, schedule = root / "today.csv", root / "odds.csv", root / "schedule.csv"
            pd.DataFrame(rows).to_csv(source, index=False)
            pd.DataFrame([{"race_id": "r", "date": now.strftime("%Y-%m-%d"),
                          "close_at": close, "source_url": "test"}]).to_csv(schedule, index=False)
            events = []
            def capture(*args, **kwargs):
                events.append("capture")
                return []
            def persist(entries, odds):
                events.append("persist")
                return pd.DataFrame(entries), pd.DataFrame(odds), None
            with patch.multiple(upcoming, RACE_SCHEDULE_CSV=schedule,
                                UPCOMING_COUNT_FILE=root / "count.txt",
                                UPCOMING_METADATA_FILE=root / "meta.json"), \
                 patch.multiple(common, TODAY_CSV=source, TODAY_ODDS_CSV=quotes), \
                 patch.object(upcoming, "ensure_dirs"), \
                 patch.object(upcoming, "parse_race_page", return_value=(rows, [])), \
                 patch.object(upcoming, "append_win_odds_history"), \
                 patch.object(upcoming, "capture_after_fetch", side_effect=capture), \
                 patch.object(upcoming, "save_today_frames", side_effect=persist), \
                 patch.object(upcoming.time, "sleep"):
                self.assertEqual(upcoming.fetch_upcoming(), 1)
            self.assertEqual(events, ["capture", "persist", "capture"])


if __name__ == "__main__":
    unittest.main()
