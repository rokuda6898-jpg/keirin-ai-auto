import unittest
import tempfile
from pathlib import Path
from itertools import permutations
from unittest.mock import patch
import pandas as pd
from high_payout_history import replay_high_payout,build_high_payout_history


class HistoricalHighPayoutTests(unittest.TestCase):
    def race(self):
        return pd.DataFrame({'race_id':['a']*4,'date':['2026-10-01']*4,'car_no':[1,2,3,4],
            'finish_pos':[1,2,3,4],'p_win':[.4,.3,.2,.1],'p_second':[.1,.4,.3,.2],
            'p_third':[.1,.2,.4,.3],'close_at':[1790848800]*4})

    def quotes(self):
        return pd.DataFrame([{'buy':'-'.join(map(str,p)),'odds_used':500,'bet_type':'trifecta'} for p in permutations(range(1,5),3)])

    def test_temporal_boundary_rejects_leakage(self):
        for day in ['2026-10-01','2026-10-02']:
            with self.assertRaises(ValueError):replay_high_payout(self.race(),self.quotes(),day)

    def test_unverified_odds_and_labels_remain_separate(self):
        chosen=pd.DataFrame({'buy':['1-2-3'],'high_payout_selected':[True]})
        def select(riders,*args):
            self.assertNotIn('finish_pos',riders)
            return chosen,{'rating':'B','patterns':['3着荒れ'],'unknowns':[]}
        with patch('high_payout_history.select_high_payout',side_effect=select):
            result=replay_high_payout(self.race(),self.quotes(),'2026-09-30')
        self.assertEqual(result['scope'],'unverified_odds_simulation')
        self.assertTrue(result['hit'])
        self.assertEqual(result['simulated_return_yen'],50000)
        with tempfile.TemporaryDirectory() as directory:
            report=build_high_payout_history([result],Path(directory),100)
            self.assertEqual(report['groups']['unverified_odds_simulation']['hits'],1)
            self.assertEqual(report['groups']['verified_preclose_odds_replay']['races'],0)
            self.assertFalse(report['prospective_performance_included'])
            self.assertFalse(report['auto_promotion'])

    def test_missing_market_is_not_counted_as_verified(self):
        result=replay_high_payout(self.race(),self.quotes().head(2),'2026-09-30')
        self.assertEqual(result['scope'],'missing_market')
        self.assertEqual(result['count'],0)
        quotes=self.quotes().assign(odds_captured_at_jst='2026-10-01T09:00:00+09:00',odds_verification_status='verified')
        self.assertEqual(replay_high_payout(self.race(),quotes,'2026-09-30')['scope'],'verified_preclose_odds_replay')
