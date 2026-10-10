import copy
import unittest
from graded_tactics import tactical_inputs, conditional_tactical_features


class TacticsTests(unittest.TestCase):
    def setUp(self):
        self.data={'race':{'id':'r','raceType':'final'},'entries':[{'number':i} for i in range(1,5)],
                   'linePrediction':{'lines':[{'entries':[{'numbers':[1]},{'numbers':[2]}]},
                                              {'entries':[{'numbers':[3]},{'numbers':[4]}]}]}}

    def test_opponent_and_teammate_roles(self):
        x=tactical_inputs(self.data)
        self.assertEqual(x['riders']['2']['opposing_leaders'],[3])
        self.assertEqual(len(x['leader_matchups']),2)
        self.assertEqual(conditional_tactical_features(x,2,(1,))[5],1)
        self.assertEqual(conditional_tactical_features(x,2,(3,))[6],1)

    def test_result_fields_do_not_change_inputs(self):
        before=tactical_inputs(self.data)
        self.data.update(results=[{'order':1}],trifecta=[{'payoff':99999}])
        self.assertEqual(before,tactical_inputs(self.data))

    def test_missing_advancement_not_invented(self):
        self.assertIsNone(tactical_inputs(self.data)['advancement_text'])

    def test_ambiguous_line_not_flattened_into_fact(self):
        self.data['linePrediction']['lines'][0]['entries']=[{'numbers':[1,2]}]
        x=tactical_inputs(self.data)
        self.assertEqual(x['line_status'],'ambiguous')
        self.assertEqual(x['leader_matchups'],[])
        self.assertFalse(x['forecast_available'])


if __name__=='__main__':unittest.main()
