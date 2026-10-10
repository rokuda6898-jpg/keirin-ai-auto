import json
import tempfile
import unittest
from datetime import datetime, timedelta
from itertools import permutations
from pathlib import Path
from unittest.mock import patch

from graded_department import GRADES, JST, grade_of, rank_tickets, forecast, load_ledger, build_feed


def riders():
    return [dict(car_no=c, player_name=f'選手{c}', score=100+c, win_rate=.12,
                 place2_rate=.3, place3_rate=.5, line_verification_status='verified',
                 line_id=(c-1)//2, line_position=(c-1)%2+1, style='逃', front_runner_count=3)
            for c in range(1, 8)]


class GradedDepartmentTests(unittest.TestCase):
    def test_exact_grade_and_supporting_race_exclusion(self):
        for code, expected in [(1,None),(2,None),(3,'G3'),(4,'G2'),(5,'G1'),(6,None),(None,None)]:
            data={'schedule':{'cupId':'today'},'cups':[{'id':'old','grade':5},{'id':'today','grade':code}],
                  'race':{'isGradeRace':True}}
            self.assertEqual(grade_of(data),expected)
            data['race']['isGradeRace']=False
            self.assertIsNone(grade_of(data))

    def test_valid_unique_bounded_tickets_and_no_forced_holes(self):
        rows=riders(); prices={k:40. for k in permutations(range(1,8),3)}
        result=rank_tickets(rows,prices)
        self.assertEqual(len(result['main']),12)
        self.assertEqual(len({r['buy'] for r in result['main']}),12)
        self.assertEqual(result['hole'],[])
        self.assertEqual(result['hole_market_baseline'],[])
        self.assertFalse(result['calibrated'])

    def test_complete_market_required(self):
        with self.assertRaises(ValueError):
            rank_tickets(riders(),{(1,2,3):100.})

    def test_hole_threshold_and_matched_comparator(self):
        prices={k:100. if k[2]==7 else 30. for k in permutations(range(1,8),3)}
        result=rank_tickets(riders(),prices)
        self.assertTrue(all(r['odds']>=100 for r in result['hole']))
        self.assertLessEqual(len(result['hole']),12)
        self.assertEqual(len(result['hole']),len(result['hole_market_baseline']))

    def test_no_result_leakage(self):
        rows=riders(); a=rank_tickets(rows,{})
        for r in rows:
            r['finish_pos']=8-r['car_no'];r['actual_trifecta']='7-6-5'
        self.assertEqual(a,rank_tickets(rows,{}))

    def test_missing_lines_do_not_invent_line_reason(self):
        rows=riders()
        for r in rows:r['line_verification_status']='missing'
        result=rank_tickets(rows,{})
        self.assertFalse(result['verified_lines'])
        self.assertTrue(all(not x['reasons'] for x in result['main']))

    def test_no_after_close_fetch_or_backfill(self):
        now=datetime(2026,10,10,12,tzinfo=JST)
        def fail(url):raise AssertionError('must not fetch')
        records,decisions=forecast([{'race_id':'r','close_at':now.timestamp()-1}],now,fail,{})
        self.assertFalse(records)
        self.assertEqual(decisions[0]['status'],'closed_without_snapshot')

    def test_initial_and_one_priced_upgrade_are_preserved(self):
        now=datetime(2026,10,10,12,tzinfo=JST)
        base={'race_id':'r','grade':'G1','close_at':now.timestamp()+900,'snapshot_at_jst':now.isoformat(),'market_available':False}
        upgrade={**base,'snapshot_at_jst':(now+timedelta(seconds=60)).isoformat(),'market_available':True}
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'ledger'
            p.write_text('\n'.join(json.dumps(r) for r in [base,upgrade]),encoding='utf-8')
            self.assertEqual(load_ledger(p)['r'],upgrade)
            p.write_text(json.dumps({**base,'snapshot_at_jst':(now+timedelta(seconds=1000)).isoformat()}),encoding='utf-8')
            with self.assertRaises(ValueError):load_ledger(p)

    def test_best_equation_may_be_added_once_before_close_without_rewriting_original_snapshot(self):
        now=datetime(2026,10,10,12,tzinfo=JST)
        base={'race_id':'r','grade':'G1','close_at':now.timestamp()+1800,
              'snapshot_at_jst':now.isoformat(),'market_available':True,'main':[],'hole':[]}
        extension={**base,'best_equation':{'snapshot_at':(now+timedelta(seconds=30)).isoformat(),
                   'tickets':['1-2-3']},'best_equation_status':'saved','best_equation_error':None}
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'ledger'
            # Carry-file set sorting can put equal-snapshot records in either order.
            p.write_text('\n'.join(json.dumps(r) for r in [extension,base]),encoding='utf-8')
            self.assertEqual(load_ledger(p)['r'],extension)
            late={**base,'best_equation':{'snapshot_at':datetime.fromtimestamp(base['close_at']+1,JST).isoformat(),
                  'tickets':['1-2-3']},'best_equation_status':'saved'}
            p.write_text('\n'.join(json.dumps(r) for r in [base,late]),encoding='utf-8')
            with self.assertRaises(ValueError):load_ledger(p)

    def test_best_equation_is_saved_for_unpriced_existing_race(self):
        now=datetime(2026,10,10,12,tzinfo=JST)
        schedule=[{'race_id':'r','date':'2026-10-10','venue':'test','race_no':1,
                   'close_at':now.timestamp()+900,'source_url':'https://example.test/r'}]
        data={'cups':[{'id':'cup','grade':5}],'schedule':{'cupId':'cup'},
              'race':{'id':'r','isGradeRace':True,'closeAt':now.timestamp()+900}}
        original={'race_id':'r','grade':'G1','close_at':now.timestamp()+900,
                  'snapshot_at_jst':(now-timedelta(seconds=60)).isoformat(),
                  'market_available':False,'main':['1-2-3'],'hole':[]}
        best={'tickets':['1-2-3','2-1-3'],'probabilities':[.2,.1],
              'snapshot_at':now.isoformat(timespec='seconds'),'model_version':'test'}
        def query(_state,key):return data if key=='FETCH_KEIRIN_RACE' else {}
        with patch('fetch_today_entries.find_query_data',side_effect=query), patch('race_features.build_entry_rows',return_value=riders()), patch('fetch_today_entries._entry_rows_complete',return_value=(True,[],7)), patch('fetch_today_entries.build_odds_rows',return_value=[]), patch('graded_tactics_live.predict',return_value=best):
            additions,decisions=forecast(schedule,now,lambda _url:{},{'r':original},clock=lambda:now)
        self.assertEqual(len(additions),1)
        self.assertEqual(additions[0]['best_equation'],best)
        self.assertFalse(additions[0]['market_available'])
        self.assertEqual(additions[0]['snapshot_at_jst'],original['snapshot_at_jst'])
        self.assertEqual(additions[0]['main'],original['main'])
        self.assertEqual(decisions[0]['best_equation_status'],'saved')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'ledger'
            path.write_text('\n'.join(json.dumps(row) for row in [original,additions[0]]),encoding='utf-8')
            self.assertEqual(load_ledger(path)['r'],additions[0])

    def test_capture_unpriced_then_only_one_valid_market_upgrade(self):
        now=datetime(2026,10,10,12,tzinfo=JST)
        schedule=[{'race_id':'r','date':'2026-10-10','venue':'test','race_no':1,'close_at':now.timestamp()+900,'source_url':'https://example.test/r'}]
        data={'cups':[{'id':'cup','grade':5}],'schedule':{'cupId':'cup'},'race':{'id':'r','isGradeRace':True,'closeAt':now.timestamp()+900}}
        market={'oddsUpdatedAt':now.timestamp()}
        def query(state,key):return data if key=='FETCH_KEIRIN_RACE' else market
        prices=[{'bet_type':'trifecta','buy':'-'.join(map(str,k)),'odds_used':200.} for k in permutations(range(1,8),3)]
        with patch('fetch_today_entries.find_query_data',side_effect=query),patch('race_features.build_entry_rows',return_value=riders()),patch('fetch_today_entries._entry_rows_complete',return_value=(True,[],7)),patch('fetch_today_entries.build_odds_rows',return_value=[]) as odds:
            first,_=forecast(schedule,now,lambda url:{},{},clock=lambda:now)
            self.assertFalse(first[0]['market_available'])
            self.assertEqual(len(first[0]['main']),12)
            self.assertEqual(first[0]['hole'],[])
            odds.return_value=prices
            upgrade,_=forecast(schedule,now,lambda url:{},{'r':first[0]},clock=lambda:now+timedelta(seconds=10))
            self.assertTrue(upgrade[0]['market_available'])
            self.assertEqual(len(upgrade[0]['market_baseline']),12)
            again,_=forecast(schedule,now,lambda url: self.fail('immutable forecast must not re-fetch'),{'r':upgrade[0]},clock=lambda:now)
            self.assertEqual(again,[])
            market['oddsUpdatedAt']=now.timestamp()-400
            stale,_=forecast(schedule,now,lambda url:{},{},clock=lambda:now)
            self.assertFalse(stale[0]['market_available'])

    def test_conditioned_third_changes_with_verified_line(self):
        rows=riders(); first=rank_tickets(rows,{})
        rows[6]['line_id']=rows[0]['line_id'];rows[6]['line_position']=3
        second=rank_tickets(rows,{})
        self.assertNotEqual(first['main'],second['main'])

    def test_tied_result_payouts_are_not_duplicated(self):
        now=datetime(2026,10,10,12,tzinfo=JST)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'outputs').mkdir()
            result={'race_id':'r','official_result_available':True,'actual_trifecta_buys':['1-2-3','2-1-3'],
                    'payouts_trifecta_json':{'1-2-3':12000,'2-1-3':8000}}
            (root/'outputs/latest_results.json').write_text(json.dumps([result]),encoding='utf-8')
            decisions=[{'race_id':'r','grade':'G1','date':'2026-10-10','venue':'test','race_no':1}]
            feed=build_feed(root,{},decisions,now)
            self.assertEqual(feed['races'][0]['payouts'],{'1-2-3':12000.,'2-1-3':8000.})
            self.assertIsNone(feed['races'][0]['grade'])

    def test_best_equation_is_separate_in_full_race_feed(self):
        now=datetime(2026,10,10,12,tzinfo=JST)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'outputs').mkdir()
            saved={'race_id':'r','date':'2026-10-10','venue':'test','race_no':1,
                   'start_at':now.timestamp()+600,'close_at':now.timestamp()+300,
                   'snapshot_at_jst':now.isoformat(),'grade':'G1','market_available':False,
                   'riders':[{'car':'1','name':'選手1'}], 'cancelled_cars':[],
                   'verified_lines':True,'main':[],'hole':[],'market_baseline':[],
                   'hole_market_baseline':[],'best_equation_status':'saved',
                   'best_equation':{'tickets':['1-2-3','2-1-3'],'probabilities':[.2,.1],
                                    'snapshot_at':now.isoformat()}}
            feed=build_feed(root,{'r':saved},[],now)
            race=feed['races'][0]
            self.assertEqual(race['best_equation']['tickets'],['1-2-3','2-1-3'])
            self.assertEqual(race['best_equation']['snapshot_at'],now.isoformat())


if __name__=='__main__':unittest.main()
