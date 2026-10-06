import json
import tempfile
import unittest
from datetime import datetime,timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
from betting_logic import STRATEGY_VERSION
from validation_coverage import build_validation_coverage


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.folder=self.root/'company';self.folder.mkdir()
        self.now=datetime(2026,10,6,20,tzinfo=ZoneInfo('Asia/Tokyo'))

    def write(self,name,data):
        (self.folder/name).write_text(json.dumps(data),encoding='utf-8')

    def snapshot(self,rid='one'):
        return {'race_id':rid,'venue':'test','race_no':1,'date':'2026-10-06','strategy_version':STRATEGY_VERSION,
                'snapshot_at':(self.now-timedelta(hours=1)).isoformat(),
                'close_at':(self.now-timedelta(minutes=55)).timestamp(),
                'tickets':[{'buy':'1-2-3','group':'本線','stake_yen':100}]}

    def test_stages_use_schedule_and_reject_postclose(self):
        schedule=[{'date':'2026-10-06','race_id':rid,'venue':'test','race_no':i,
                   'start_at':(self.now-timedelta(minutes=50)).timestamp(),
                   'close_at':(self.now-timedelta(minutes=55)).timestamp()} for i,rid in enumerate(['one','two'],1)]
        schedule.append(dict(schedule[0],date='2026-10-07',race_id='tomorrow'))
        pd.DataFrame(schedule).to_csv(self.root/'latest_race_schedule.csv',index=False)
        good=self.snapshot();bad=self.snapshot('two');bad['snapshot_at']=self.now.isoformat()
        (self.folder/'ticket_return_snapshots.jsonl').write_text('\n'.join(json.dumps(r) for r in [good,good,bad]),encoding='utf-8')
        settled=dict(good,actual_trifecta='1-2-3',payout_per_100yen=500)
        self.write('ticket_return_settled.json',[settled])
        result=build_validation_coverage(self.root,self.now)
        self.assertEqual(result['coverage']['scheduled_races'],2)
        self.assertEqual(result['coverage']['saved_preclose'],1)
        self.assertEqual(result['time_checks']['invalid_snapshot_rows'],1)
        self.assertEqual(result['time_checks']['selected_tickets_without_quote_time'],1)
        self.assertEqual(result['conditions']['venue'][0]['return_rate'],5)
        self.assertEqual(result['conditions']['time'][0]['condition'],'夜（17〜21時）')

    def test_unknown_records_are_not_claimed_as_failures_or_zero_returns(self):
        result=build_validation_coverage(self.root,self.now)
        self.assertEqual(result['coverage']['scheduled_races'],0)
        self.assertEqual(result['conditions']['venue'],[])
        self.assertFalse(result['time_checks']['annual_cutoff_confirmed'])
        self.assertIn('全国の全開催', (self.folder/'validation_coverage.html').read_text(encoding='utf-8'))

    def test_quote_time_after_prediction_is_flagged(self):
        row=self.snapshot();row['tickets'][0]['odds_captured_at_jst']=self.now.isoformat()
        (self.folder/'ticket_return_snapshots.jsonl').write_text(json.dumps(row),encoding='utf-8')
        result=build_validation_coverage(self.root,self.now)
        self.assertEqual(result['time_checks']['quote_after_snapshot_or_close'],1)

    def test_reference_cutoff_and_sparse_counts(self):
        self.write('annual_rider_knowledge.json',{'asof_date':'2026-10-07','window_end_exclusive':'2026-10-07',
            'profiles':{'rider':{'evaluation':{'races':2,'effective_races':1.5}}}})
        result=build_validation_coverage(self.root,self.now)
        self.assertFalse(result['time_checks']['annual_cutoff_confirmed'])
        self.assertEqual(result['sparse_rider_reference']['15走相当未満'],1)
