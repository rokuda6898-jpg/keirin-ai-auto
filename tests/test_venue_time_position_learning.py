import unittest

import pandas as pd

from test_venue_time_learning import fixture
from venue_time_position_learning import FEATURE_SETS, prepare, features, fit_model, score_holdout


class PositionTests(unittest.TestCase):
    def setUp(self):
        self.data=fixture(36)
        self.data['score']=88+self.data.car_no
        self.data['win_rate']=self.data.car_no/100
        self.data['place2_rate']=self.data.car_no/90
        self.data['place3_rate']=self.data.car_no/80
        self.data['back_count']=self.data.car_no

    def test_finish_labels_not_in_training_features(self):
        prepared,races=prepare(self.data,'2025-03-01')
        self.assertEqual(len(races),36)
        self.assertEqual(len(prepared),252)
        for name in FEATURE_SETS:
            self.assertFalse(set(('finish_pos','target_win','target_top3'))&set(features(prepared,name).columns))
        self.assertTrue(prepared.groupby('race_id').target_win.sum().eq(1).all())
        self.assertTrue(prepared.groupby('race_id').target_top3.sum().eq(3).all())

    def test_holdout_labels_cannot_change_fitted_features(self):
        base,_=prepare(self.data,'2025-03-01')
        altered=self.data.copy()
        altered.loc[altered.date.ge('2025-01-26'),'finish_pos'] = (
            altered.loc[altered.date.ge('2025-01-26'),'car_no'].map(lambda k: 8-k))
        other,_=prepare(altered,'2025-03-01')
        for name in FEATURE_SETS:
            a=features(base[base.date.lt('2025-01-26')],name)
            b=features(other[other.date.lt('2025-01-26')],name)
            pd.testing.assert_frame_equal(a,b)

    def test_small_scale_end_to_end_position_candidates(self):
        frame,_=prepare(self.data,'2025-03-01')
        train=frame[frame.date.lt('2025-01-26')]
        test=frame[frame.date.ge('2025-01-26')]
        pred={}
        for variant in FEATURE_SETS:
            win=fit_model(train,variant,'target_win',max_iter=12,min_leaf=5)
            top3=fit_model(train,variant,'target_top3',max_iter=12,min_leaf=5)
            pred[variant]={'win':win.predict_proba(features(test,variant))[:,1],
                           'top3':top3.predict_proba(features(test,variant))[:,1]}
        score=score_holdout(test,pred)
        self.assertEqual(len(score),11)
        for variant in FEATURE_SETS:
            self.assertTrue(score[f'top1_hit_{variant}'].isin((0,1)).all())
            self.assertTrue(score[f'win_logloss_{variant}'].ge(0).all())
            self.assertTrue(score[f'top3_recall_{variant}'].between(0,1).all())
        changed=self.data.copy()
        changed.loc[changed.date.ge('2025-01-26'),'score']=999
        future,_=prepare(changed,'2025-03-01')
        pd.testing.assert_frame_equal(features(train,'venue_time'),
            features(future[future.date.lt('2025-01-26')],'venue_time'))


if __name__=='__main__':unittest.main()
