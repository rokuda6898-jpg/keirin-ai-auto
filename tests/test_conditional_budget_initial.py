"""Prospective FIRST allocation; never retroactive forecasts or changed stakes."""
from copy import deepcopy
from datetime import datetime, timedelta
from itertools import permutations
import unittest
from conditional_budget import allocate, settle
from conditional_budget_initial import initialize, eligible

NOW = datetime.fromisoformat('2026-10-11T12:00:00+09:00')
TICKETS = ['-'.join(map(str, p)) for p in permutations(range(1, 5), 3)][:12]

def record():
    return dict(race_id='012120261011', date='2026-10-11', venue='テスト', race_no='1',
        snapshot_at_jst=(NOW-timedelta(hours=4)).isoformat(), close_at=(NOW+timedelta(hours=1)).timestamp(),
        top12=TICKETS[:], probabilities=[.02]*12, actual=[], cancelled_cars=[],
        purchase_authorized=False, source_url='https://example.test/race')

def quote(row, clock):
    return {k:row[k] for k in ('race_id','date','venue','race_no')} | {
        'bet_type':'trifecta', 'provider':'fixture', 'source_url':'https://example.test/odds',
        'captured_at_jst':clock().isoformat(), 'odds':dict.fromkeys(row['top12'],20)}

def invoke(row=None, result=None, scheduled=None, loader=quote, clock=lambda: NOW, persist=None):
    row = record() if row is None else row
    rid = row['race_id']; rows={rid:row}
    report=initialize(rows,{rid:scheduled or deepcopy(row)},{rid:result or {}},loader,clock,persist,workers=1)
    return row, report

class FirstAllocationTests(unittest.TestCase):
    def test_still_open_gets_amounts_with_separate_time_not_reprediction(self):
        original=record(); row, report=invoke(deepcopy(original))
        self.assertEqual(report['allocated'],1)
        self.assertEqual(sum(a['stake_yen'] for a in row['budget']['allocations']),10000)
        self.assertEqual(row['budget']['odds_basis'],'initial_allocation_after_forecast')
        self.assertEqual(row['budget']['prediction_snapshot_at_jst'],original['snapshot_at_jst'])
        self.assertEqual(row['budget']['allocation_frozen_at_jst'],NOW.isoformat())
        for key in original: self.assertEqual(row[key],original[key],key)
        self.assertIsNone(row['budget_settlement']['roi'])

    def test_existing_allocated_infeasible_unavailable_are_never_repriced(self):
        for status in ('allocated','infeasible','unavailable'):
            with self.subTest(status=status):
                row=record(); row['budget']={'status':status}; original=deepcopy(row)
                def forbid(*args): self.fail('existing decision repriced')
                _,r=invoke(row,loader=forbid)
                self.assertEqual(r['attempted'],0);self.assertEqual(row,original)

    def test_closed_and_near_close_get_no_backfill(self):
        for seconds in (-3600,0,60):
            row=record();row['close_at']=(NOW+timedelta(seconds=seconds)).timestamp()
            row,r=invoke(row);self.assertEqual(r['attempted'],0);self.assertNotIn('budget',row)

    def test_known_result_even_with_future_schedule_is_not_used(self):
        for result in ({'official_result_available':True},{'official_result_available':'1'}):
            row,r=invoke(result=result);self.assertEqual(r['attempted'],0)
        row=record();row['actual']=['1-2-3'];_,r=invoke(row);self.assertEqual(r['attempted'],0)

    def test_cross_day_and_identity_mismatch_are_excluded(self):
        for key,value in [('date','2026-10-10'),('venue','別場'),('race_no','2'),('race_id','different')]:
            row=record(); scheduled=deepcopy(row);scheduled[key]=value
            _,r=invoke(row,scheduled=scheduled);self.assertEqual(r['attempted'],0)
        row=record();row['date']='2026-10-10';_,r=invoke(row);self.assertEqual(r['attempted'],0)

    def test_earlier_updated_deadline_is_respected(self):
        row=record(); scheduled=deepcopy(row);scheduled['close_at']=NOW.timestamp()-1
        _,r=invoke(row,scheduled=scheduled);self.assertEqual(r['attempted'],0)

    def test_cancelled_and_future_or_naive_forecast_are_not_allocated(self):
        for patch in ({'cancelled_cars':['1']},{'snapshot_at_jst':(NOW+timedelta(hours=2)).isoformat()},
                      {'snapshot_at_jst':'2026-10-11T08:00:00'}):
            row=record();row.update(patch);_,r=invoke(row);self.assertEqual(r['attempted'],0)

    def test_quote_crossing_deadline_is_not_saved(self):
        now=[NOW]
        def late(row,clock):
            now[0]=NOW+timedelta(hours=2)
            return quote(row,clock)
        row,r=invoke(loader=late,clock=lambda:now[0]);self.assertEqual(r['expired'],1);self.assertNotIn('budget',row)

    def test_missing_quote_is_explicit_not_default_100(self):
        def missing(*args):raise ValueError('missing')
        row,r=invoke(loader=missing);self.assertEqual(r['unavailable'],1)
        self.assertEqual(row['budget']['allocations'],[])
        self.assertIsNone(row['budget_settlement']['roi'])

    def test_checkpoint_and_official_settlement_preserve_amounts(self):
        saved=[];row,r=invoke(persist=lambda rows:saved.append(deepcopy(rows)))
        self.assertEqual(len(saved),1);plan=deepcopy(row['budget']);buy=row['top12'][0]
        result=settle(plan,row['top12'],{'winning_buys':[buy],'payout_complete':True,'payouts':{buy:1100}})
        self.assertEqual(result['payout_yen'],plan['allocations'][0]['stake_yen']//100*1100)
        self.assertEqual(result['profit_yen'],result['payout_yen']-10000)
        self.assertEqual(row['budget'],plan)
        again,r=invoke(row);self.assertEqual(r['attempted'],0);self.assertEqual(again['budget'],plan)

    def test_infeasible_is_still_a_frozen_no_purchase_decision(self):
        def low(row,clock):return quote(row,clock)|{'odds':dict.fromkeys(row['top12'],10)}
        row,r=invoke(loader=low);self.assertEqual(r['infeasible'],1)
        self.assertEqual(row['budget']['allocations'],[])
        _,r=invoke(row);self.assertEqual(r['attempted'],0)

if __name__=='__main__':unittest.main()
