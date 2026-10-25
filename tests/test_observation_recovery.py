import tempfile
import unittest
from pathlib import Path
import pandas as pd
from annual_knowledge import frame_read, save_observations

class ObservationRecoveryTests(unittest.TestCase):
    def test_corrupt_supplement_is_preserved_and_can_be_recollected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'rider_official_observations.csv'
            original = 'race_id,player_id\n1,2\n3,4,5\n'
            path.write_text(original)
            with self.assertWarns(RuntimeWarning):
                self.assertTrue(frame_read(path).empty)
            backup = next(Path(folder).glob('*.corrupt-*'))
            self.assertEqual(backup.read_text(), original)
            save_observations(pd.DataFrame([{'race_id':'7','player_id':'8'}]), path)
            self.assertEqual(frame_read(path).iloc[0].race_id, '7')
            self.assertEqual(backup.read_text(), original)

    def test_primary_history_corruption_still_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'history.csv'
            path.write_text('race_id,player_id\n1,2\n3,4,5\n')
            with self.assertRaises(pd.errors.ParserError):
                frame_read(path)

class DailyFreshnessTests(unittest.TestCase):
    def test_yesterday_is_rejected_in_morning_but_allowed_after_midnight(self):
        from datetime import datetime
        from unittest import mock
        from zoneinfo import ZoneInfo
        import site_manager
        class Clock(datetime):
            hour_value = 8
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 10, 25, cls.hour_value, tzinfo=ZoneInfo('Asia/Tokyo'))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'snapshot.csv'
            path.write_text('date,close_at\n2026-10-24,1\n')
            with mock.patch.multiple(site_manager, TODAY_CSV=path, RACE_SCHEDULE_PATH=path, datetime=Clock):
                self.assertEqual({p['type'] for p in site_manager.audit_freshness()}, {'daily_inputs_stale'})
                Clock.hour_value = 0
                self.assertEqual(site_manager.audit_freshness(), [])
