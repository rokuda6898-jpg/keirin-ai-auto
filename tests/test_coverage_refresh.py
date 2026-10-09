import unittest
from datetime import datetime, timedelta, timezone
from coverage_refresh import refresh_mode


class CoverageRefreshTests(unittest.TestCase):
    def test_missing_evening_forecast_recovers_before_close_window(self):
        now = datetime(2026, 10, 10, 8, tzinfo=timezone(timedelta(hours=9)))
        row = {'date':'2026-10-10', 'race_id':'1', 'close_at':now.timestamp()+10*3600}
        self.assertEqual(refresh_mode([row], [row], set(), now), 'all_upcoming')
        self.assertEqual(refresh_mode([row], [row], {'1'}, now), 'near_close')
        self.assertEqual(refresh_mode([row], [], set(), now), 'daily')
        self.assertEqual(refresh_mode([], [row], set(), now), 'daily')
        row['close_at'] = now.timestamp()-1
        self.assertEqual(refresh_mode([row], [row], set(), now), 'near_close')
