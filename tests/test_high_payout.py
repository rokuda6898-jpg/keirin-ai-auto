import unittest
import pandas as pd
from itertools import permutations
from datetime import datetime,timedelta
from zoneinfo import ZoneInfo
from betting_logic import score_riders,select_race
from high_payout_strategy import select_high_payout,VERSION
from high_payout_department import summarize_high_payout,summarize_comparison


class HighPayoutTests(unittest.TestCase):
    def race(self):
        return score_riders(pd.DataFrame({'car_no':range(1,8),'p_win':[.7,.1,.07,.04,.03,.03,.03],
            'line_id':[1,1,1,2,2,2,2],
            'p_second':[.2,.15,.1,.1,.1,.2,.15],'p_third':[.1,.1,.1,.1,.1,.3,.2],
            'position_model_source':'position_specialists','fixed_axis_calibration_passed':True}))

    def market(self):
        return pd.DataFrame([{'buy':'-'.join(map(str,p)),'bet_type':'trifecta',
            'odds_used':1000 if p[0]==2 else 50} for p in permutations(range(1,8),3)])

    def test_independent_heads_and_strict_longshot_gates(self):
        riders=self.race(); market=self.market()
        candidates,plan=select_race(riders,market)
        holes=candidates[candidates.ticket_group.eq('穴')]
        self.assertTrue(plan['first_fixed'])
        self.assertFalse(holes.empty)
        self.assertTrue(holes['head'].eq(2).all())
        self.assertTrue(holes.odds_used.ge(100).all())
        self.assertTrue(holes.ev.ge(1.25).all())
        self.assertTrue(holes.prob.ge(.001).all())
        self.assertLessEqual(len(holes),12)
        self.assertFalse(candidates.buy.duplicated().any())
        self.assertFalse(set(holes.buy)&set(candidates[candidates.ticket_group.eq('本線')].buy))

    def test_comparison_is_frozen_separate_and_requires_preclose(self):
        r=self.race();f,d=select_high_payout(r,self.market(),{'chaos_index':70,'first_gap':50})
        self.assertFalse(d['comparison']['automatic_promotion'])
        for t in d['comparison']['variants']['third_focus']:
            row=f[f.buy.eq(t['buy'])].iloc[0]
            self.assertIn('3着荒れ',row.hole_pattern)
            self.assertGreaterEqual(t['odds'],100)
            self.assertGreaterEqual(t['ev'],1.25)
        base={'race_id':'a','snapshot_at':'2026-10-07T10:00:00+09:00','close_at':datetime(2026,10,7,11,tzinfo=ZoneInfo('Asia/Tokyo')).timestamp(),'payout_per_100yen':15000,'actual_trifecta':'1-2-7','high_payout_department':{'comparison':{'version':'third_focus_v1','variants':{'current':[], 'third_focus':[{'buy':'1-2-7','odds':150}]}}}}
        result=summarize_comparison([base,base])
        self.assertEqual(result['third_focus']['hits'],1)
        self.assertEqual(result['third_focus']['stake_yen'],100)
        self.assertEqual(result['current']['proposed_races'],0)
        self.assertEqual(summarize_comparison([{**base,'snapshot_at':'2026-10-07T12:00:00+09:00'}])['third_focus']['target_races'],0)

    def test_fragmented_and_solo_need_verified_lines_without_score_bonus(self):
        r=self.race();r['line_id']=[1,1,2,3,4,5,6]
        market=self.market();risk={'chaos_index':70,'first_gap':50}
        _,unverified=select_high_payout(r,market,risk)
        r['line_verification_status']='verified'
        f,verified=select_high_payout(r,market,risk)
        self.assertEqual(verified['expectation_score'],unverified['expectation_score'])
        self.assertEqual(verified['recommended_count'],unverified['recommended_count'])
        self.assertIn('細切れ戦',verified['patterns'])
        self.assertIn('単騎',verified['patterns'])
        self.assertNotIn('単騎',unverified['patterns'])
        r.loc[0,'line_id']=None
        _,invalid=select_high_payout(r,market,risk)
        self.assertNotIn('単騎',invalid['patterns'])

    def test_incomplete_market_and_orderly_race_skip(self):
        r=self.race();market=self.market()
        _,d=select_high_payout(r,market.head(5),{'chaos_index':70,'first_gap':50})
        self.assertEqual(d['recommended_count'],0)
        _,d=select_high_payout(r,market,{'chaos_index':10,'first_gap':50})
        self.assertEqual(d['rating'],'見送り')
        self.assertIn('順当',d['skip_reason'])
        _,d=select_high_payout(r,market.assign(odds_used=99.9),{'chaos_index':90,'first_gap':50})
        self.assertEqual(d['recommended_count'],0)

    def test_settlement_separates_old_pending_skips_and_payout_bands(self):
        now=datetime(2026,10,7,19,tzinfo=ZoneInfo('Asia/Tokyo'))
        base={'race_id':'a','snapshot_at':now.isoformat(),'close_at':(now+timedelta(minutes=10)).timestamp(),
              'actual_trifecta':'2-1-6','payout_per_100yen':50000,
              'tickets':[{'buy':'2-1-6','group':'穴'}],
              'high_payout_department':{'version':VERSION,'tickets':[{'buy':'2-1-6','odds':1000,'patterns':['1着荒れ']}]}}
        pending={**base,'race_id':'pending','payout_per_100yen':None}
        skip={**base,'race_id':'skip','tickets':[]}
        old={**base,'race_id':'old','high_payout_department':{}}
        s=summarize_high_payout([base,pending,skip,old])
        self.assertEqual(s['target_races'],2)
        self.assertEqual(s['predicted_races'],1)
        self.assertEqual(s['skipped_races'],1)
        self.assertEqual(s['hits'],1)
        self.assertEqual(s['return_rate'],500)
        self.assertEqual(s['payout_hit_counts'],{'100':1,'300':1,'500':1,'1000':0})
        self.assertEqual(s['pattern_results']['1着荒れ']['hit_rate'],1)
        self.assertEqual(summarize_high_payout([{**base,'snapshot_at':(now+timedelta(minutes=20)).isoformat()}])['target_races'],0)
