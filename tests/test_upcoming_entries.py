import unittest

import pandas as pd

from fetch_upcoming_entries import _entries_complete, select_upcoming_races


class UpcomingEntryTests(unittest.TestCase):
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
