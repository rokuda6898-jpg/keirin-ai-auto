import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from fusion_shadow_live import restore_race_metadata
import fusion_shadow_live as shadow


class ShadowRaceMetadataTests(unittest.TestCase):
    def test_concurrent_publish_merge_reads_official_result_file(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            results=root/'latest_results.json'
            results.write_text(json.dumps([{'race_id':'r','official_result_available':True,
                'actual_trifecta':'1-2-3','actual_trifecta_odds':12}]),encoding='utf-8')
            row={'race_id':'r','top12':['1-2-3']}
            with patch.object(shadow,'LATEST_RESULTS_JSON',results), patch.object(shadow,'read_ledger',return_value=[row]), \
                 patch.object(shadow,'write_ledger') as write, patch.object(shadow,'build_report',return_value={}):
                shadow.merge_remote_ledger(root/'no-remote-ledger')
                saved=write.call_args.args[0][0]
                self.assertTrue(saved['top12_hit'])
                self.assertEqual(saved['top12'],['1-2-3'])
                self.assertEqual(saved['hit_odds'],[12])

    def test_restore_close_time_and_race_number_after_model_projection(self):
        entries = pd.DataFrame({
            'race_id': ['race-a'] * 3,
            'close_at': [1_800_000_000] * 3,
            'race_no': [9] * 3,
        })
        projected = pd.DataFrame({
            'race_id': ['race-a'] * 3,
            'car_no': [1, 2, 3],
        })

        restored = restore_race_metadata(projected, entries)

        self.assertEqual(restored.close_at.tolist(), [1_800_000_000] * 3)
        self.assertEqual(restored.race_no.tolist(), [9] * 3)
        self.assertEqual(restored.car_no.tolist(), [1, 2, 3])

    def test_reject_inconsistent_race_level_metadata(self):
        entries = pd.DataFrame({
            'race_id': ['race-a', 'race-a'],
            'close_at': [1_800_000_000, 1_800_000_001],
            'race_no': [9, 9],
        })
        projected = pd.DataFrame({'race_id': ['race-a'], 'car_no': [1]})

        with self.assertRaisesRegex(ValueError, 'race-level'):
            restore_race_metadata(projected, entries)


if __name__ == '__main__':
    unittest.main()
