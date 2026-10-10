import unittest

from graded_tactics_live import predict


class LiveBestEquationTests(unittest.TestCase):
    def test_frozen_best_equation_returns_twelve_unique_valid_predictions(self):
        data = {
            'race': {'id': 'r-best', 'raceType': '決勝'},
            'entries': [{'number': car} for car in range(1, 8)],
            'linePrediction': {'lines': [
                {'entries': [{'numbers': [1]}, {'numbers': [2]}, {'numbers': [3]}]},
                {'entries': [{'numbers': [4]}, {'numbers': [5]}]},
                {'entries': [{'numbers': [6]}, {'numbers': [7]}]},
            ]},
        }
        rows = [{
            'race_id': 'r-best', 'car_no': car, 'grade': 'G1', 'race_type': '決勝',
            'score': 90 + car, 'win_rate': .12, 'place2_rate': .3, 'place3_rate': .5,
            'back_count': car % 3, 'front_runner_count': car % 2,
            'stalker_count': 1, 'deep_closer_count': 1, 'marker_count': 1, 'age': 30,
            'recent_avg_finish': 3.2, 'recent_races_count': 5, 'days_since_last_race': 3,
            'track_win_rate': .1, 'track_place2_rate': .3, 'track_place3_rate': .5,
            'track_races': 12, 'line_verification_status': 'verified',
            'line_id': 1 if car <= 3 else 2 if car <= 5 else 3,
            'line_position': car if car <= 3 else car - 3 if car <= 5 else car - 5,
        } for car in range(1, 8)]
        result = predict(rows, data, 'G1')
        self.assertEqual(len(result['tickets']), 12)
        self.assertEqual(len(set(result['tickets'])), 12)
        self.assertTrue(all(len(buy.split('-')) == 3 and len(set(buy.split('-'))) == 3
                            for buy in result['tickets']))
        self.assertAlmostEqual(result['distribution_mass'], 1.0)
        self.assertEqual(result['line_status'], 'verified')


if __name__ == '__main__':
    unittest.main()
