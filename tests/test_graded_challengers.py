import unittest
from itertools import permutations
import numpy as np
from graded_challengers import interaction_features, scenario_features, reweight, category, feature_mode, ORIGINAL_FEATURES
import graded_retrospective as base


class ChallengerTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{'car_no':c,'score':100+c,'front_runner_count':c,'race_type':'qualifying','grade':'G1'} for c in range(1,6)]

    def test_opposition_changes_feature_without_candidate_change(self):
        before=interaction_features(self.rows,1,(2,),[],True)
        self.rows[-1]['score']=130
        after=interaction_features(self.rows,1,(2,),[],True)
        self.assertFalse(np.array_equal(before[-54:],after[-54:],equal_nan=True))

    def test_results_cannot_change_features(self):
        before=scenario_features(self.rows,[])
        for r in self.rows:r.update(finish_pos=1,actual_trifecta='1-2-3')
        np.testing.assert_equal(before,scenario_features(self.rows,[]))

    def test_collapse_has_exact_requested_marginals(self):
        orders=list(permutations(range(1,6),3));dist={k:1/len(orders) for k in orders}
        result=reweight(dist,1,[.1,.2,.3,.4])
        self.assertAlmostEqual(sum(result.values()),1)
        for cat,p in enumerate([.1,.2,.3,.4]):self.assertAlmostEqual(sum(v for k,v in result.items() if category(k,1)==cat),p)

    def test_feature_context_restores_on_error(self):
        with self.assertRaises(ValueError):
            with feature_mode(True):raise ValueError()
        self.assertIs(base.features,ORIGINAL_FEATURES)


if __name__=='__main__':unittest.main()
