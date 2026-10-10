import copy
import unittest
from graded_tactics import tactical_inputs, conditional_tactical_features, tactical_interaction_features
from race_choice_equation import RaceChoiceModel, distribution as choice_distribution
import numpy as np


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

    def test_interaction_features_capture_team_support_and_rival_pressure(self):
        x=tactical_inputs(self.data)
        rows=[{'car_no':i,'score':float(i*10),'back_count':i} for i in range(1,5)]
        values=tactical_interaction_features(x,rows,2,(1,))
        self.assertEqual(len(values),10)
        self.assertEqual(values[0],0)
        self.assertEqual(values[6],1)

    def test_race_choice_distribution_is_normalized_over_unique_ordered_triples(self):
        # Two raw features plus one missing-value indicator per feature.
        model=RaceChoiceModel(np.zeros(2),np.zeros(2),np.ones(2),np.zeros(4))
        rows=[{'car_no':i} for i in range(1,5)]
        def features(records,candidate,prefix,stages,contextual):
            return [float(candidate),float(len(prefix))]
        result=choice_distribution(rows,[model,model,model],features,[])
        self.assertEqual(len(result),24)
        self.assertEqual(len(set(result)),24)
        self.assertAlmostEqual(sum(result.values()),1.0)
        self.assertTrue(all(len(set(ticket))==3 for ticket in result))


if __name__=='__main__':unittest.main()
