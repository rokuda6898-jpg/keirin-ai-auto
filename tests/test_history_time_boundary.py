import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import predict
import live_snapshot_store as store


class HistoryBoundaryTests(unittest.TestCase):
    def test_same_day_and_target_results_are_not_in_live_prior_features(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'history.csv'
            pd.DataFrame([
                {'race_id':'past','date':'2026-10-05','player_id':'1','car_no':1,'finish_pos':3,'venue':'test'},
                {'race_id':'target','date':'2026-10-06','player_id':'1','car_no':1,'finish_pos':1,'venue':'test'},
                {'race_id':'future','date':'2026-10-07','player_id':'1','car_no':1,'finish_pos':1,'venue':'test'}]).to_csv(path,index=False)
            today=pd.DataFrame([{'race_id':'target','date':'2026-10-06','player_id':'1','car_no':1,'finish_pos':1,'venue':'test'}])
            with patch.object(predict,'HISTORY_CSV',path),patch.object(store,'_load_feature_base',return_value=None),patch.object(store,'_save_feature_base'):
                result=predict.add_today_prior_features(today)
            self.assertEqual(result.player_prior_races.iloc[0],1)
            self.assertTrue(pd.isna(result.finish_pos.iloc[0]))
            self.assertEqual(today.finish_pos.iloc[0],1)
