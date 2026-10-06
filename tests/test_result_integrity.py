import unittest
import pandas as pd
from common import valid_finish_mask, add_player_prior_features, add_player_elo_features, add_pair_history_features
from train import clean_training_history
from prediction_quality import reliability, ev_audit, quality_audit

class ResultIntegrityTests(unittest.TestCase):
    def test_published_strategy_refresh_required_after_change(self):
        import json, tempfile
        from pathlib import Path
        from unittest.mock import patch
        from betting_logic import STRATEGY_VERSION
        import site_manager
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(site_manager, 'OUTPUT_DIR', Path(tmp)):
                path=Path(tmp)/'latest_race_strategy.json'
                path.write_text(json.dumps([{'strategy_version':'old'}]))
                self.assertTrue(site_manager.audit_strategy_version())
                path.write_text(json.dumps([{'strategy_version':STRATEGY_VERSION}]))
                self.assertEqual(site_manager.audit_strategy_version(),[])

    def test_invalid_ranks(self):
        data=pd.DataFrame({'finish_pos':[0,-1,99,1.5,float('inf'),None,1,3,4], 'entries_number':[3]*9})
        self.assertEqual(valid_finish_mask(data).tolist(),[False]*6+[True,True,False])

    def test_training_excludes_whole_race_and_keeps_archive(self):
        data=pd.DataFrame({'race_id':['good']*3+['bad']*3+['no_winner']*3,'finish_pos':[1,2,3,0,2,3,2,3,3],'entries_number':[3]*9})
        before=data.copy(deep=True)
        cleaned,audit=clean_training_history(data)
        self.assertEqual(cleaned.race_id.unique().tolist(),['good'])
        self.assertEqual(audit['invalid_finish_rows'],1)
        self.assertEqual(audit['excluded_races'],2)
        pd.testing.assert_frame_equal(data,before)

    def test_priors_ignore_invalid_without_current_result_leak(self):
        data=pd.DataFrame({'race_id':['a','b','c','d'],'date':['2026-01-01','2026-01-02','2026-01-03','2026-01-04'],'player_id':['x']*4,'finish_pos':[1,0,3,2],'entries_number':[3]*4})
        last=add_player_prior_features(data).iloc[-1]
        self.assertEqual(last.player_prior_races,2)
        self.assertEqual(last.player_prior_place2_rate,.5)
        self.assertEqual(last.player_prior_avg_finish,2)

    def test_elo_and_pair_ignore_invalid_competitor(self):
        data=pd.DataFrame({'race_id':['a']*3+['b']*3,'date':['2026-01-01']*3+['2026-01-02']*3,'player_id':['x','y','z']*2,'finish_pos':[0,2,3,1,2,3],'entries_number':[3]*6})
        elo=add_player_elo_features(data)
        self.assertEqual(elo.iloc[3].player_elo,1500)
        pair=add_pair_history_features(data)
        self.assertEqual(pair.iloc[3].h2h_prior_meetings,0)
        self.assertEqual(pair.iloc[4].h2h_prior_meetings,1)

    def test_low_probability_bins_and_ev_returns(self):
        bins=reliability([('r',.005,0),('r',.015,1)],[0,.01,.02,1])['bins']
        self.assertEqual([b['observations'] for b in bins],[1,1,0])
        result=ev_audit([{'race_id':'r','actual_trifecta':'1-2-3','payout_per_100yen':300,'tickets':[{'buy':'1-2-3','ev':1.2},{'buy':'1-3-2','ev':1.3}]}])
        self.assertEqual(result[1]['actual_flat_return_rate'],3)
        self.assertEqual(result[2]['actual_flat_return_rate'],0)

if __name__=='__main__': unittest.main()
