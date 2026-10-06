import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import live_snapshot_store as store
from common import add_pair_history_features


class CaptureReliabilityTests(unittest.TestCase):
    def test_pair_history_is_pre_race_with_exact_line_partner_statistics(self):
        rows = []
        for race, finishes in [("1", [2, 1, 3]), ("2", [1, 3, 2]), ("3", [np.nan]*3)]:
            for car, finish in enumerate(finishes, 1):
                rows.append({"race_id": race, "player_id": str(car), "car_no": car,
                             "date": "2026-10-01", "finish_pos": finish, "line_id": 1, "line_position": car})
        frame = pd.DataFrame(rows).sample(frac=1, random_state=2)
        result = add_pair_history_features(frame)
        pd.testing.assert_index_equal(result.index, frame.index)
        final = result[result.race_id.eq("3")].set_index("player_id")
        self.assertEqual(final.loc["1", "h2h_prior_meetings"], 4)
        self.assertEqual(final.loc["1", "h2h_prior_win_share"], .75)
        self.assertEqual(final.loc["2", "line_pair_prior_races"], 2)
        self.assertEqual(final.loc["2", "line_pair_second_win_rate"], .5)

    def test_raw_capture_never_contains_result_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            now = datetime.now(ZoneInfo("Asia/Tokyo"))
            frame = pd.DataFrame([{"race_id": "test", "player_id": "1", "date": "2026-10-06",
                                  "close_at": (now + timedelta(minutes=20)).timestamp(), "finish_pos": 1}])
            with patch.object(store, "SNAPSHOT_CSV", path / "snapshots.csv"), patch.object(store, "ensure_dirs"):
                self.assertEqual(store.capture_frame(frame, None, now), 1)
                saved = pd.read_csv(path / "snapshots.csv")
                self.assertNotIn("finish_pos", saved)
                self.assertFalse(saved.model_input_ready.iloc[0])
                self.assertTrue(saved.label_finish_pos.isna().all())

    def test_feature_cache_keeps_fresh_odds_and_invalidates_line_or_history_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); history = root / "history.csv"; history.write_text("initial")
            frame = pd.DataFrame([{"race_id": "r1", "player_id": "1", "date": "2026-10-06",
                                   "car_no": 1, "line_position": 1, "odds_win": 2}])
            enriched = frame.copy()
            for col in store.HISTORY_FEATURE_COLUMNS:
                enriched[col] = .25
            with patch.object(store, "FEATURE_BASE_CSV", root / "base.csv"), \
                 patch.object(store, "FEATURE_BASE_META", root / "base.json"), patch.object(store, "HISTORY_CSV", history):
                store._save_feature_base(enriched)
                frame["odds_win"] = 8
                cached = store._load_feature_base(frame)
                self.assertIsNotNone(cached)
                self.assertEqual(store._apply_feature_base(frame, cached).odds_win.iloc[0], 8)
                frame["line_position"] = 2
                self.assertIsNone(store._load_feature_base(frame))
                frame["line_position"] = 1
                history.write_text("changed")
                self.assertIsNone(store._load_feature_base(frame))
