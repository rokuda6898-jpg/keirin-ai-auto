"""Offline validation of the optional licensed racecard CSV path."""
import io
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd
import authorized_today_feed as feed

NOW = datetime(2026, 11, 18, 12, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
CLOSE = datetime(2026, 11, 18, 15, 10, tzinfo=ZoneInfo("Asia/Tokyo")).timestamp()


def rows():
    return pd.DataFrame([
        {
            "race_id": "018120261118",
            "date": "2026-11-18",
            "venue": "小倉",
            "race_no": 1,
            "player_id": f"01{i:04d}",
            "player_name": f"選手{i}",
            "car_no": i,
            "entries_number": 7,
            "declared_entries_number": 7,
            "start_at": CLOSE + 300,
            "close_at": CLOSE,
            "source_url": "https://licensed.example.test/races/01",
            "result_available": 0,
            "finish_pos": None,
            "score": 88.0 + i,
        } for i in range(1, 8)
    ])


class AuthorizedFeedTests(unittest.TestCase):
    def test_accepts_complete_current_day(self):
        frame = feed.validate(rows(), NOW)
        self.assertEqual(len(frame), 7)

    def test_blocks_stale_and_partial(self):
        old = rows()
        old["date"] = "2026-11-17"
        with self.assertRaisesRegex(ValueError, "not exclusively"):
            feed.validate(old, NOW)
        with self.assertRaisesRegex(ValueError, "partial rider"):
            feed.validate(rows().iloc[:-1], NOW)

    def test_blocks_results_and_duplicate_riders(self):
        spoiled = rows()
        spoiled.loc[0, "finish_pos"] = 1
        with self.assertRaisesRegex(ValueError, "result-bearing"):
            feed.validate(spoiled, NOW)
        dup = pd.concat([rows(), rows().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate racer"):
            feed.validate(dup, NOW)

    def test_import_is_atomic_and_never_recycles_odds(self):
        class Response:
            def __init__(self, data):
                self.content = data
            def raise_for_status(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data" / "raw"
            out = root / "outputs"
            raw.mkdir(parents=True)
            out.mkdir(parents=True)
            target = raw / "today_entries.csv"
            odds = raw / "today_odds.csv"
            odds.write_text("old-unverified-odds", encoding="utf-8")
            config = {
                "RAW_DIR": raw,
                "OUTPUT_DIR": out,
                "TODAY_CSV": target,
                "TODAY_ODDS_CSV": odds,
                "SCHEDULE": out / "latest_race_schedule.csv",
                "TRIFECTA": raw / "today_trifecta_odds.csv",
                "ensure_dirs": lambda: None,
            }
            patches = [patch.object(feed, key, value) for key, value in config.items()]
            for p in patches: p.start()
            try:
                valid = rows().to_csv(index=False).encode("utf-8")
                result = feed.load(
                    "https://licensed.example.test/data.csv",
                    now=NOW, requester=lambda *args, **kwargs: Response(valid),
                )
                self.assertEqual(result.race_id.nunique(), 1)
                self.assertEqual(len(pd.read_csv(target)), 7)
                self.assertEqual(len(pd.read_csv(out / "latest_race_schedule.csv")), 1)
                self.assertEqual(len(pd.read_csv(odds)), 0)
                damaged = rows().iloc[:-1].to_csv(index=False).encode("utf-8")
                with self.assertRaises(ValueError):
                    feed.load(
                        "https://licensed.example.test/data.csv",
                        now=NOW, requester=lambda *args, **kwargs: Response(damaged),
                    )
                self.assertEqual(len(pd.read_csv(target)), 7)
            finally:
                for p in reversed(patches): p.stop()


if __name__ == "__main__":
    unittest.main()
