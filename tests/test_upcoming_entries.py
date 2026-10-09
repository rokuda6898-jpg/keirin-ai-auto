import unittest
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch

import pandas as pd

from fetch_upcoming_entries import _entries_complete, select_upcoming_races


class UpcomingEntryTests(unittest.TestCase):
    def test_inside_cutoff_does_not_block_other_refresh_or_remove_base(self):
        import fetch_upcoming_entries as module
        now = datetime.now(timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            schedule = pd.DataFrame([dict(race_id=rid,date=now.astimezone(module.ZoneInfo('Asia/Tokyo')).date().isoformat(),
                close_at=now.timestamp()+seconds,source_url=rid) for rid,seconds in [('early',120),('later',1200)]])
            schedule.to_csv(root/'schedule.csv',index=False)
            base = pd.DataFrame([dict(race_id=rid,car_no=car,entries_number=3)
                                 for rid in ('early','later') for car in (1,2,3)])
            base.to_csv(root/'entries.csv',index=False)
            def save(entries, odds):
                return pd.DataFrame(entries), pd.DataFrame(odds), pd.DataFrame()
            with patch.multiple(module,RACE_SCHEDULE_CSV=root/'schedule.csv',
                                UPCOMING_COUNT_FILE=root/'count',UPCOMING_METADATA_FILE=root/'meta'), \
                 patch('common.TODAY_CSV',root/'entries.csv'), patch('common.TODAY_ODDS_CSV',root/'odds.csv'), \
                 patch.object(module,'ensure_dirs'), patch.object(module,'append_win_odds_history'), \
                 patch.object(module,'capture_after_fetch',return_value=[]), \
                 patch.object(module,'parse_race_page',return_value=(base[base.race_id=='later'].to_dict('records'),[])) as parse, \
                 patch.object(module,'save_today_frames',side_effect=save) as writer:
                self.assertEqual(module.fetch_upcoming(0,1440,sleep_sec=0),1)
                parse.assert_called_once_with('later')
                self.assertEqual({r['race_id'] for r in writer.call_args.args[0]}, {'early','later'})
                self.assertIn('early',(root/'meta').read_text())

    def test_selects_only_near_close_window(self):
        schedule = pd.DataFrame(
            [
                {"race_id": "too_soon", "close_at": 1299},
                {"race_id": "window_start", "close_at": 1300},
                {"race_id": "window_end", "close_at": 3400},
                {"race_id": "too_late", "close_at": 3401},
            ]
        )
        selected = select_upcoming_races(schedule, now_epoch=1000, min_minutes=5, max_minutes=40)
        self.assertEqual(selected["race_id"].tolist(), ["window_start", "window_end"])

    def test_incomplete_tail_riders_are_rejected(self):
        frame = pd.DataFrame([
            {"race_id": "r1", "car_no": n, "entries_number": 9}
            for n in range(1, 8)
        ])
        self.assertFalse(_entries_complete(frame, "r1"))

    def test_complete_nine_rider_field_is_accepted(self):
        frame = pd.DataFrame([
            {"race_id": "r1", "car_no": n, "entries_number": 9}
            for n in range(1, 10)
        ])
        self.assertTrue(_entries_complete(frame, "r1"))


if __name__ == "__main__":
    unittest.main()
