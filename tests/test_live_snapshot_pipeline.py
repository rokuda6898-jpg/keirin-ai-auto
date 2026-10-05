import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import live_snapshot_model
import train_live_snapshot_model


class LiveSnapshotPipelineTests(unittest.TestCase):
    def test_snapshot_contexts_are_scored_separately(self):
        frame = pd.DataFrame([
            {"snapshot_id": "r1::40m", "race_id": "r1", "snapshot_bucket": "40m", "label_finish_pos": 1},
            {"snapshot_id": "r1::40m", "race_id": "r1", "snapshot_bucket": "40m", "label_finish_pos": 2},
            {"snapshot_id": "r1::10m", "race_id": "r1", "snapshot_bucket": "10m", "label_finish_pos": 2},
            {"snapshot_id": "r1::10m", "race_id": "r1", "snapshot_bucket": "10m", "label_finish_pos": 1},
        ])
        top = train_live_snapshot_model.score_top1(
            frame, np.array([0.8, 0.2, 0.3, 0.7])
        )
        self.assertEqual(len(top), 2)
        self.assertEqual(int(top["is_hit"].sum()), 2)

    def test_live_feature_matrix_includes_time_context(self):
        frame = pd.DataFrame([
            {
                "race_id": "r1",
                "snapshot_bucket": "20m",
                "snapshot_bucket_code": 2.0,
                "seconds_to_close": 1200,
                "odds_move_pct": 0.10,
                "odds_move_last_pct": -0.02,
                "odds_snapshot_count": 3,
                "race_no": 1,
                "car_no": 1,
            }
        ])
        X, fills = live_snapshot_model.build_live_matrix_from_snapshot_rows(frame)
        self.assertIn("snapshot_bucket_code", X.columns)
        self.assertIn("seconds_to_close", X.columns)
        self.assertEqual(float(X.iloc[0]["snapshot_bucket_code"]), 2.0)
        self.assertEqual(float(X.iloc[0]["seconds_to_close"]), 1200.0)
        self.assertEqual(set(X.columns), set(live_snapshot_model.LIVE_FEATURES))
        self.assertEqual(set(fills), set(live_snapshot_model.LIVE_FEATURES))

    def test_live_prediction_workflows_restore_central_history(self):
        for path in [
            ".github/workflows/daily.yml",
            ".github/workflows/intraday.yml",
            ".github/workflows/live-snapshot-learning.yml",
        ]:
            text = Path(path).read_text(encoding="utf-8")
            self.assertIn("Restore central history for live features", text)
            self.assertIn("keirin-central-history-v1-", text)
            self.assertIn("live feature history too small", text)

    def test_live_model_keeps_calibration_and_test_blocks_out_of_refit(self):
        text = Path("src/train_live_snapshot_model.py").read_text(encoding="utf-8")
        self.assertIn(
            "chronological_70pct_fit_15pct_calibration_15pct_untouched_test",
            text,
        )
        self.assertIn('"model": live_model', text)
        self.assertIn('"calibrator": calibrator', text)
        self.assertNotIn('"model": final_model', text)


if __name__ == "__main__":
    unittest.main()
