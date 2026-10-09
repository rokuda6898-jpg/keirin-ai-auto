import copy
import tempfile
import unittest
from datetime import datetime, timedelta
from itertools import permutations
from pathlib import Path
from unittest.mock import patch
import pandas as pd

import three_year_equations_live as live


class ThreeYearLiveTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 9, 12, tzinfo=live.common.JST)
        self.picks = [{'buy':'-'.join(map(str,t)), 'probability':.02, 'stake_yen':100}
                      for t in list(permutations((1,2,3,4),3))[:12]]
        self.methods = {n:{'basis':'trained','tickets':copy.deepcopy(self.picks)} for n in live.NAMES}
        rows=[]
        live.common.freeze(rows,'three_year_421',{'race_id':'r1','date':'2026-10-09','venue':'test',
            'race_no':1,'close_at':(self.now+timedelta(minutes=20)).timestamp()},self.methods,
            {'three_year':{'training_cutoff_exclusive':'2026-10-09'}},self.now)
        self.forecast=rows[0]
        self.prices={t['buy']:30. for t in self.picks}
        self.stamps={k:self.now.isoformat() for k in self.prices}

    def test_allocations_are_common_budget_immutable_and_preclose(self):
        saved=[]
        with patch.object(live,'quote_view',return_value=(self.prices,self.stamps,{},[])), patch.object(live.common,'clock',return_value=self.now):
            self.assertEqual(live.freeze_allocations(saved,self.forecast,pd.DataFrame(),pd.DataFrame(),self.now),3)
            original=copy.deepcopy(saved)
            self.assertEqual(live.freeze_allocations(saved,self.forecast,pd.DataFrame(),pd.DataFrame(),self.now),0)
            self.assertEqual(saved,original)
            for row in saved:
                self.assertEqual(set(row['methods']),set(live.NAMES))
                self.assertTrue(all(m['spent_yen']==6000 for m in row['methods'].values()))
            later=self.now+timedelta(hours=1)
            self.assertEqual(live.freeze_allocations([],self.forecast,pd.DataFrame(),pd.DataFrame(),later),0)

    def test_mixed_snapshot_and_missing_prices_never_create_allocations(self):
        self.stamps[self.picks[0]['buy']] = (self.now-timedelta(seconds=1)).isoformat()
        with patch.object(live,'quote_view',return_value=(self.prices,self.stamps,{},[])), patch.object(live.common,'clock',return_value=self.now):
            self.assertEqual(live.freeze_allocations([],self.forecast,pd.DataFrame(),pd.DataFrame(),self.now),0)
        with patch.object(live,'quote_view',return_value=({},self.stamps,{},[])):
            self.assertEqual(live.freeze_allocations([],self.forecast,pd.DataFrame(),pd.DataFrame(),self.now),0)

    def test_command_and_weighted_roi_use_saved_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); saved=[]
            self.prices[self.picks[0]['buy']]=100.
            with patch.object(live,'quote_view',return_value=(self.prices,self.stamps,{},[])), patch.object(live.common,'clock',return_value=self.now):
                live.freeze_allocations(saved,self.forecast,pd.DataFrame(),pd.DataFrame(),self.now)
            live.common.save(folder/'allocations.json',saved)
            plan=live.command('first_anchor',3,'r1',folder)
            self.assertEqual(plan['spent_yen'],6000)
            outcomes={'r1':{'winning_buys':[self.picks[0]['buy']], 'payouts':{self.picks[0]['buy']:10000}, 'payout_complete':True}}
            metrics=live.common.metrics([saved[0]],outcomes,'first_anchor',3)
            self.assertEqual(metrics['stake_yen'],6000)
            self.assertEqual(metrics['return_yen'],plan['tickets'][0]['stake_yen']*100)
            self.assertEqual(live.command('first_anchor',3,'missing',folder)['status'],'no_frozen_preclose_allocation')
            saved[0]['methods']['first_anchor']['tickets'][0]['stake_yen']=6000
            live.common.save(folder/'allocations.json',saved)
            with self.assertRaises(ValueError):
                live.allocations(folder)


if __name__=='__main__':
    unittest.main()
