import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import pandas as pd
from betting_logic import STRATEGY_VERSION
from selection_research import probability_first, chronological_calibration, build_selection_research


class SelectionResearchTests(unittest.TestCase):
    def rows(self,n):
        start=datetime(2026,1,1,tzinfo=ZoneInfo('Asia/Tokyo'))
        return [{'race_id':str(i),'strategy_version':STRATEGY_VERSION,
                 'snapshot_at':(start+timedelta(hours=i)).isoformat(),
                 'close_at':(start+timedelta(hours=i,minutes=10)).timestamp(),
                 'actual_trifecta':'1-2-3','payout_per_100yen':500,
                 'tickets':[{'buy':'1-2-3','prob':.1,'group':'本線'}]} for i in range(n)]

    def test_probability_rank_keeps_gates_budget_and_no_overlap(self):
        frame=pd.DataFrame([
            {'buy':'1-2-3','prob':.2,'ev':1.2,'main_formation':True,'hole_formation':True},
            {'buy':'1-3-2','prob':.01,'ev':10,'main_formation':True,'hole_formation':True},
            {'buy':'2-1-3','prob':.3,'ev':1.0,'main_formation':True,'hole_formation':True}])
        chosen=probability_first(frame,{'main_ev':1.10,'hole_ev':1.25},1,1)
        self.assertEqual([t['buy'] for t in chosen],['1-2-3','1-3-2'])
        self.assertEqual(len({t['buy'] for t in chosen}),len(chosen))
        self.assertEqual(probability_first(frame,{},0,0),[])

    def test_calibration_requires_race_count_not_ticket_count(self):
        rows=self.rows(2)
        rows[0]['tickets']*=200
        self.assertEqual(chronological_calibration(rows)['status'],'collecting')

    def test_holdout_labels_do_not_change_fit(self):
        rows=self.rows(200)
        first=chronological_calibration(rows)
        self.assertEqual(first['status'],'offline_holdout')
        for row in rows[140:]:row['actual_trifecta']='3-2-1'
        second=chronological_calibration(rows)
        self.assertEqual(first['training_bins'],second['training_bins'])
        self.assertLess(first['train_last_close'],first['test_first_close'])

    def test_no_reconstruction_for_old_snapshots_or_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            rows=self.rows(2)
            rows[1]['payout_per_100yen']=None
            report=build_selection_research(rows,Path(directory))
            self.assertEqual(report['paired_races'],0)
            self.assertIsNone(report['probability_first']['return_rate'])
            row=rows[0]
            row['selection_challenger']={'version':'probability_first_v1','snapshot_at':row['snapshot_at'],
                                         'tickets':[{'buy':'3-2-1'}]}
            report=build_selection_research(rows,Path(directory))
            self.assertEqual(report['paired_races'],1)
            self.assertEqual(report['current_ev_first']['return_rate'],5)
            self.assertEqual(report['probability_first']['return_rate'],0)
