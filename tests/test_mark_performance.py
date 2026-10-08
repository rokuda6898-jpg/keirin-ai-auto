"""Synthetic tests for prospective-only rider mark reporting."""
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mark_performance import (
    build_mark_report, record_preclose_marks, race_prediction_marks,
)

JST = ZoneInfo("Asia/Tokyo")


def predictions(close_time):
    win = [0.70, 0.20, 0.07, 0.02, 0.005, 0.003, 0.002]
    second = [0.10, 0.60, 0.15, 0.08, 0.04, 0.02, 0.01]
    third = [0.08, 0.10, 0.55, 0.12, 0.08, 0.04, 0.03]
    return pd.DataFrame([
        {"race_id": "000001", "date": "2026-10-08", "venue": "試験場",
         "race_no": 1, "car_no": i + 1, "close_at": close_time.timestamp(),
         "p_win": win[i], "p_second": second[i], "p_third": third[i]}
        for i in range(7)
    ])


class MarkAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        self.now = datetime(2026, 10, 8, 9, 0, tzinfo=JST)
        self.pred = predictions(self.now + timedelta(minutes=10))

    def tearDown(self):
        self.temp.cleanup()

    def test_rider_marks_are_six_tiered_and_unique(self):
        marks = race_prediction_marks(self.pred)
        self.assertEqual(marks[1], "◎")
        self.assertEqual(len(marks), 5)
        self.assertEqual(set(marks.values()), {"◎", "○", "▲", "△", "☆"})
        self.assertNotIn(7, marks)

    def test_freeze_once_and_score_only_official_results(self):
        self.assertEqual(record_preclose_marks(self.pred, self.now, self.out), 1)
        self.assertEqual(record_preclose_marks(self.pred, self.now + timedelta(minutes=1), self.out), 0)
        ledger = pd.read_csv(self.out / "mark_prediction_ledger.csv", dtype={"race_id": str})
        self.assertEqual(len(ledger), 7)
        (self.out / "latest_results.json").write_text(json.dumps([{
            "race_id": "000001", "actual_trifecta": "1-2-3",
            "official_result_available": True, "result_source": "official",
        }]), encoding="utf-8")
        report = build_mark_report(self.out)
        self.assertEqual(report["frozen_races"], 1)
        self.assertEqual(report["settled_races"], 1)
        self.assertEqual(report["marks"]["◎"]["first"], 1)
        self.assertEqual(report["marks"]["○"]["top2"], 1)
        self.assertEqual(report["marks"]["▲"]["top3"], 1)
        self.assertTrue((self.out / "company" / "mark_performance.html").exists())
        self.assertEqual(build_mark_report(self.out)["settled_races"], 1)

    def test_rejects_postclose_and_out_of_window_predictions(self):
        self.assertEqual(record_preclose_marks(self.pred, self.now + timedelta(minutes=12), self.out), 0)
        self.assertEqual(record_preclose_marks(
            predictions(self.now + timedelta(minutes=45)), self.now, self.out), 0)
        self.assertEqual(build_mark_report(self.out)["settled_races"], 0)

    def test_unofficial_results_are_not_scored(self):
        record_preclose_marks(self.pred, self.now, self.out)
        (self.out / "latest_results.json").write_text(json.dumps([{
            "race_id": "000001", "actual_trifecta": "1-2-3",
            "official_result_available": False,
        }]), encoding="utf-8")
        self.assertEqual(build_mark_report(self.out)["settled_races"], 0)


if __name__ == "__main__":
    unittest.main()
