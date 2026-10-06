import copy
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
from betting_logic import STRATEGY_VERSION
from prediction_quality import quality_audit, miss_reason
from ticket_return_department import save_snapshots


class PredictionQualityTests(unittest.TestCase):
    def row(self):
        now = datetime(2026, 10, 6, 12, tzinfo=ZoneInfo("Asia/Tokyo"))
        return {"strategy_version": STRATEGY_VERSION, "race_id": "r", "snapshot_at": now.isoformat(),
                "close_at": (now + timedelta(minutes=20)).timestamp(), "actual_trifecta": "1-2-3",
                "tickets": [{"buy": "1-2-4", "prob": .1}],
                "position_probabilities": [{"position": 1, "probabilities": {"1": .3, "2": .3, "3": .4}}]}

    def test_filter_compression_and_fixed_axis_are_distinct(self):
        row = self.row()
        candidate = {"buy": "1-2-3", "ev": 1.2, "main_formation": True, "hole_formation": False}
        row["candidate_evidence"] = {"main_ev": 1.1, "hole_ev": 1.25, "candidates": [candidate]}
        self.assertEqual(miss_reason(row), "ev_compression_miss")
        candidate["ev"] = 1.05
        self.assertEqual(miss_reason(row), "ev_filter_skip")
        candidate["ev"] = None
        self.assertEqual(miss_reason(row), "odds_missing")
        row["candidate_evidence"].update(first_fixed=True, fixed_car=2)
        self.assertEqual(miss_reason(row), "fixed_axis_miss")

    def test_fixed_axis_wins_are_separate_from_ticket_hits(self):
        row = self.row()
        row['candidate_evidence'] = {'first_fixed': True, 'fixed_car': 1}
        report = quality_audit([row])
        axis = report['fixed_axis_audit']
        self.assertEqual(axis['axis_wins'], 1)
        self.assertEqual(axis['actual_win_rate'], 1)
        self.assertAlmostEqual(axis['mean_predicted'], .3)
        self.assertNotIn('hit', report['reason_counts'])
        row['position_probabilities'] = []
        axis = quality_audit([row])['fixed_axis_audit']
        self.assertEqual(axis['missing_probability_races'], 1)
        self.assertIsNone(axis['actual_win_rate'])

    def test_legacy_coverage_does_not_invent_ev_compression(self):
        row = self.row()
        self.assertEqual(miss_reason(row), "third_not_covered")
        row["tickets"][0]["buy"] = "1-3-2"
        self.assertEqual(miss_reason(row), "second_not_covered")
        row["tickets"][0]["buy"] = "2-1-3"
        self.assertEqual(miss_reason(row), "first_not_covered")
        row["tickets"] = []
        self.assertEqual(miss_reason(row), "skip")

    def test_bins_include_all_riders_not_just_top_pick_and_separate_ticket_samples(self):
        report = quality_audit([self.row()])
        first = report["positions"][0]
        band = first["bins"][3]
        self.assertEqual(first["observations"], 3)
        self.assertEqual(band["observations"], 2)
        self.assertEqual(band["races"], 1)
        self.assertAlmostEqual(band["mean_predicted"], .3)
        self.assertEqual(band["actual_rate"], .5)
        self.assertAlmostEqual(first["multiclass_brier"], .74)
        self.assertAlmostEqual(first["uniform_brier"], 2/3)
        self.assertEqual(report["selected_ticket_calibration"]["observations"], 1)
        self.assertFalse(report["automatically_recalibrated"])

    def test_boundary_probabilities_invalid_missing_and_deduplication(self):
        row = self.row()
        row["position_probabilities"][0]["probabilities"] = {"1": 1, "2": 0, "3": 0}
        report = quality_audit([row, copy.deepcopy(row)])
        self.assertEqual(report["races"], 1)
        self.assertEqual(report["positions"][0]["bins"][9]["actual_rate"], 1)
        row["position_probabilities"][0]["probabilities"]["1"] = float("nan")
        self.assertEqual(quality_audit([row])["positions"][0]["races"], 0)
        row["position_probabilities"] = []
        self.assertEqual(quality_audit([row])["positions"][0]["observations"], 0)

    def test_post_close_old_strategy_and_invalid_results_rejected(self):
        row = self.row()
        row["close_at"] = datetime.fromisoformat(row["snapshot_at"]).timestamp()
        self.assertEqual(quality_audit([row])["races"], 0)
        row = self.row(); row["strategy_version"] = "old"
        self.assertEqual(quality_audit([row])["races"], 0)
        row = self.row(); row["actual_trifecta"] = "1-1-2"
        self.assertEqual(quality_audit([row])["races"], 0)

    def test_candidate_evidence_is_frozen_with_missing_odds_preserved(self):
        now = datetime(2026, 10, 6, 12, tzinfo=ZoneInfo("Asia/Tokyo"))
        plan = {"race_id": "r", "venue": "test", "race_no": 1, "timing_eligible": True,
                "close_at": (now + timedelta(minutes=20)).timestamp(), "first_fixed": True, "fixed_car": 1}
        candidates = pd.DataFrame([{"race_id": "r", "bet_type": "trifecta", "buy": "1-2-3",
                                    "ev": float("nan"), "prob": .1, "main_formation": True}])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            save_snapshots([plan], pd.DataFrame(), now, root, candidate_rows=candidates)
            candidates.loc[0, "ev"] = 100
            stored = json.loads((root / "company/ticket_return_snapshots.jsonl").read_text())
            self.assertIn("1-2-3", stored["candidate_evidence"]["reason_buys"]["odds_missing"])
            stored["actual_trifecta"] = "1-2-3"
            stored["tickets"] = [{"buy": "1-2-4"}]
            self.assertEqual(miss_reason(stored), "odds_missing")
            self.assertEqual(stored["candidate_evidence"]["fixed_car"], 1)


if __name__ == "__main__":
    unittest.main()
