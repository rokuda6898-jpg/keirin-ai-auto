import tempfile
import unittest
from pathlib import Path
from repair_official_observations import repair

class ObservationRecoveryTests(unittest.TestCase):
    def test_corrupt_supplement_is_preserved_for_official_recollection(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/'outputs/company/rider_official_observations.csv'
            path.parent.mkdir(parents=True)
            original = 'race_id,player_id\n1,2\n3,4,5\n'
            path.write_text(original)
            self.assertTrue(repair(root))
            backup = next(path.parent.glob('*.corrupt-*'))
            self.assertEqual(backup.read_text(), original)
            self.assertFalse(path.exists())
            self.assertFalse(repair(root))
            path.write_text('race_id,player_id\n7,8\n')
            self.assertFalse(repair(root))
            self.assertTrue(path.exists())
            self.assertEqual(backup.read_text(), original)

    def test_primary_history_is_untouched(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'history.csv'
            original='race_id,player_id\n1,2\n3,4,5\n'
            path.write_text(original)
            self.assertFalse(repair(root))
            self.assertEqual(path.read_text(),original)
