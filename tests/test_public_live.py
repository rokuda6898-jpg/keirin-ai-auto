import json
import tempfile
import unittest
from pathlib import Path
from public_live import build, forecast_view, marks


class PublicLiveTests(unittest.TestCase):
    def test_independent_marks_use_own_saved_probabilities(self):
        row={'top12':['1-2-3','2-1-3'],'probabilities':[.1,.8]}
        self.assertEqual(marks(row)[0],{'mark':'◎','car':'2'})

    def test_reject_repeated_or_invalid_car_tickets(self):
        for tickets in [['1-1-2'],['1-2-3','1-2-3'],['1-2-10']]:
            with self.assertRaises(ValueError):
                forecast_view({'top12':tickets,'snapshot_at_jst':'2026-10-09T08:00:00+09:00'})

    def test_all_scheduled_races_remain_and_unknown_payout_stays_unknown(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);out=root/'outputs';(out/'company').mkdir(parents=True)
            (out/'latest_race_schedule.csv').write_text('race_id,date,venue,race_no,start_at,close_at\na,2026-10-09,場,1,1791530000,1791529700\nb,2026-10-09,場,2,1791540000,1791539700\n',encoding='utf-8')
            row=dict(race_id='a',date='2026-10-09',venue='場',race_no=1,close_at=1791529700,
                     snapshot_at_jst='2026-10-09T08:00:00+09:00',top12=['1-2-3'])
            (out/'company/fusion_shadow_live_ledger.jsonl').write_text(json.dumps(row)+'\n',encoding='utf-8')
            (out/'latest_results.json').write_text(json.dumps([dict(race_id='a',official_result_available=True,actual_trifecta='1-2-3')]),encoding='utf-8')
            result=build(root)
            self.assertEqual(len(result['races']),2)
            self.assertEqual(result['races'][0]['shadow']['tickets'],['1-2-3'])
            self.assertIsNone(result['races'][1]['shadow'])
            self.assertEqual(result['races'][0]['payouts'],{})
            self.assertEqual(result['races'][0]['actual'],['1-2-3'])

    def test_refuse_empty_schedule(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                build(Path(temp))
