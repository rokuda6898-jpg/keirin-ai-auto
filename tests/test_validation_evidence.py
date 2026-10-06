import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from preserve_validation import save, restore
from verified_live_audit import build_verified_live_audit
from ticket_return_department import save_snapshots, build_ticket_return_department


class ValidationEvidenceTests(unittest.TestCase):
    def test_reset_merges_both_frozen_ledgers_without_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"; backup = Path(tmp) / "backup"
            path = root / "outputs/company/ticket_return_snapshots.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text('{"old":1}\n{"frozen":2}\n')
            save(backup, root)
            path.write_text('{"old":1}\n{"remote":3}\n')
            restore(backup, root); restore(backup, root)
            self.assertEqual(len(path.read_text().splitlines()), 3)
            self.assertIn('{"frozen":2}', path.read_text())

    def test_timestamp_audit_excludes_post_close_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            now = datetime(2026, 10, 6, 12, tzinfo=ZoneInfo("Asia/Tokyo"))
            pd.DataFrame([{"race_id": "pre", "prediction_created_at_jst": now.isoformat(),
                           "close_at": (now + timedelta(minutes=1)).timestamp(), "predicted_winner_car_no": 1},
                          {"race_id": "post", "prediction_created_at_jst": now.isoformat(),
                           "close_at": (now - timedelta(minutes=1)).timestamp(), "predicted_winner_car_no": 1}]).to_csv(root / "top1_prediction_ledger.csv", index=False)
            pd.DataFrame([{"race_id": rid, "actual_trifecta": "1-2-3", "official_result_available": True}
                          for rid in ["pre", "post"]]).to_csv(root / "top1_settled_results.csv", index=False)
            report = build_verified_live_audit(root)
            self.assertEqual(report["legacy_settled_rows"], 2)
            self.assertEqual(report["timestamp_verified_races"], 1)
            self.assertEqual(report["unverified_rows"], 1)

    def test_calibration_and_core_comparison_use_same_frozen_race(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); now = datetime(2026, 10, 6, 12, tzinfo=ZoneInfo("Asia/Tokyo"))
            plan = {"race_id": "r", "timing_eligible": True, "close_at": (now + timedelta(minutes=20)).timestamp(), "venue": "test", "race_no": 1}
            positions = pd.DataFrame([{"race_id": "r", "car_no": car, "p_win": p,
                                       "p_second": 1/3, "p_third": 1/3, "p_core": 1-p} for car,p in [(1,.7),(2,.2),(3,.1)]])
            save_snapshots([plan], pd.DataFrame(), now, root, positions)
            (root / "latest_results.json").write_text(json.dumps([{"race_id": "r", "actual_trifecta": "1-2-3", "official_result_available": True, "actual_trifecta_odds": 10}]))
            report = build_ticket_return_department(root)["calibration_audit"]
            self.assertEqual(report["positions"][0]["races"], 1)
            self.assertEqual(report["same_race_core_vs_final"]["final_hits"], 1)
            self.assertEqual(report["same_race_core_vs_final"]["core_hits"], 0)
