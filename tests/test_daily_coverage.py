"""Guard today's reference forecasts against retroactive or mislabeled bets."""
import json
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class DailyCoverageReferenceTest(unittest.TestCase):
    def test_all_three_model_pages_show_attributed_reference(self):
        for page,mode in [('index.html','current'),('current.html','current'),
                          ('high-roi.html','high_roi'),('ex.html','ex')]:
            html=(ROOT/'site'/page).read_text(encoding='utf8')
            self.assertIn(f'data-coverage-model="{mode}"', html)
            self.assertIn('daily-coverage.js', html)
            self.assertIn('daily-coverage.css', html)
        script=(ROOT/'site/daily-coverage.js').read_text(encoding='utf8')
        self.assertIn('nexus-formation-5385-v1', script)
        self.assertIn('本日分の専用予想が未掲載です', script)
        self.assertIn('旧方式の回収率実績は適用されません', script)
        for model in ['current471_joint_consensus_MAIN4_20261031',
                      'ROI9891_fixed_EX_20261102',
                      'ROI9234_MAIN6_fixed_EX_20261101']:
            self.assertIn(model,script)

    def test_2026_11_04_snapshot_is_preclose_and_is_not_rebranded(self):
        path=ROOT/'outputs/nexus_forecasts/2026-11-04.json'
        records=json.loads(path.read_text(encoding='utf8'))
        self.assertEqual(len(records),75)
        seen=set()
        for row in records:
            self.assertNotIn(row['race_id'],seen)
            seen.add(row['race_id'])
            self.assertTrue(row['race_id'].endswith('20261104'))
            pred=row['prediction']
            self.assertEqual(pred['model_id'],'nexus-formation-5385-v1')
            self.assertEqual(pred['replay_evidence']['evidence_kind'],'prospective_preclose_inference')
            close=datetime.fromisoformat(row['close_at'])
            self.assertLess(datetime.fromisoformat(pred['captured_at']),close)
            self.assertLess(datetime.fromisoformat(pred['input_captured_at']),close)
            actual=set()
            for b in pred['blocks']:
                for a in b['first']:
                    for second in b['second']:
                        for third in b['third']:
                            if len({a,second,third})==3:
                                actual.add(f'{a}-{second}-{third}')
            self.assertEqual(actual,set(pred['tickets']))
            self.assertEqual(len(actual),pred['count'])


if __name__=='__main__':
    unittest.main()
