import unittest
import numpy as np
import pandas as pd

from betting_logic import score_riders, race_plan
from position_semantics import exact_place_challenger
from selection_research import exact_position_comparison


class PositionSemanticsTests(unittest.TestCase):
    def test_cumulative_difference_not_cumulative_normalization(self):
        race = pd.DataFrame({'race_id': ['a'] * 3, 'p_win': [.6, .3, .1]})
        result = exact_place_challenger(race, [.7, .8, .5], [1., 1., 1.])
        np.testing.assert_allclose(result.p_second_exact_challenger, [.1, .5, .4])
        np.testing.assert_allclose(result.p_third_exact_challenger, [.3, .2, .5])

    def test_crossing_estimates_and_zero_mass_are_not_fabricated(self):
        race = pd.DataFrame({'race_id': ['a'] * 3 + ['b'] * 3, 'p_win': [.6, .3, .1] * 2})
        result = exact_place_challenger(race, [.5, .2, .1, .7, .8, .5], [.4, .1, .1, 1., 1., 1.])
        self.assertTrue(result.iloc[:3].p_second_exact_challenger.isna().all())
        self.assertAlmostEqual(result.iloc[3:].p_second_exact_challenger.sum(), 1)
        with self.assertRaises(ValueError):
            exact_place_challenger(race, [float('nan')] * 6, [1.] * 6)

    def test_unverified_probability_cannot_fix_axis(self):
        race = pd.DataFrame({'car_no': [1, 2, 3], 'p_win': [.9, .06, .04]})
        plan = race_plan(score_riders(race))
        self.assertFalse(plan['first_fixed'])
        self.assertTrue(plan['risk_veto_fixed'])
        self.assertEqual(plan['fixed_policy_status'], 'calibration_unverified')
        for bad in ['True', 1, float('nan')]:
            race['fixed_axis_calibration_passed'] = bad
            self.assertFalse(race_plan(score_riders(race))['first_fixed'])

    def test_pairing_excludes_retrospective_and_keeps_dates_separate(self):
        rows = []
        for i, day in enumerate(['2026-10-07', '2026-10-07', '2026-10-08']):
            probs = {'1': .2, '2': .5, '3': .3}
            rows.append({'race_id': str(i), 'date': day, 'close_at': i, 'snapshot_at': 'saved',
                         'actual_trifecta': '1-2-3',
                         'position_probabilities': [{'position': p, 'probabilities': probs} for p in [2, 3]],
                         'exact_position_challenger': {'version': 'cumulative_difference_v1', 'snapshot_at': 'saved',
                                                      'probabilities': {'2': probs, '3': probs}}})
        report = exact_position_comparison(rows)
        self.assertEqual(report['earlier'][0]['races'], 2)
        self.assertEqual(report['later'][0]['races'], 1)
        self.assertFalse(report['auto_promotion'])
        rows[0]['exact_position_challenger']['snapshot_at'] = 'after'
        self.assertEqual(exact_position_comparison(rows)['paired_races'], 2)


if __name__ == '__main__':
    unittest.main()
