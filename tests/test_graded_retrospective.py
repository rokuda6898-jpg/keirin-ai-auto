import unittest
import numpy as np
from graded_retrospective import features, distribution


class ConstantModel:
    def predict_proba(self,x):
        return np.tile([.7,.3],(len(x),1))


class GradedRetrospectiveTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{'car_no':c,'score':100+c,'race_type':'予選','grade':'G1',
                    'line_verification_status':'missing','line_id':1} for c in range(1,5)]

    def test_result_fields_never_enter_features(self):
        before=features(self.rows,3,(1,2),['予選','決勝'],True)
        for r in self.rows:r.update(finish_pos=1,actual_trifecta='1-2-3',result_factor=99)
        np.testing.assert_equal(before,features(self.rows,3,(1,2),['予選','決勝'],True))

    def test_stage_is_effective_only_in_contextual_candidates(self):
        before=features(self.rows,1,(),['予選','決勝'],True)
        plain=features(self.rows,1,(),[],False)
        self.rows[0]['race_type']='決勝'
        self.assertNotEqual(before[-5:-3],features(self.rows,1,(),['予選','決勝'],True)[-5:-3])
        np.testing.assert_equal(plain,features(self.rows,1,(),[],False))

    def test_conditional_distributions_are_normalized_and_exclusive(self):
        for method in ('learned_positions','stage_positions','stage_conditional'):
            p=distribution(self.rows,[ConstantModel()]*3,method,['予選'])
            self.assertEqual(len(p),24)
            self.assertAlmostEqual(sum(p.values()),1.)
            self.assertTrue(all(len(set(k))==3 for k in p))

    def test_unverified_lines_do_not_enter_as_confirmed_relation(self):
        x=features(self.rows,2,(1,),[],True)
        self.assertTrue(np.isnan(x[-1]))
        for r in self.rows:r['line_verification_status']='verified'
        self.assertEqual(features(self.rows,2,(1,),[],True)[-1],1.)


if __name__=='__main__':unittest.main()
