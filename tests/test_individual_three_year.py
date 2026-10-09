import unittest
import numpy as np
import pandas as pd
import individual_three_year as train


class ThreeYearTrainingTests(unittest.TestCase):
    def test_calendar_boundaries_and_recency(self):
        dates = ['2023-10-09', '2024-10-08', '2024-10-09', '2025-10-08', '2025-10-09', '2026-10-08']
        self.assertEqual(train.weights(dates, '2026-10-09').tolist(), [1, 1, 2, 2, 4, 4])
        for day in ('2023-10-08', '2026-10-09', '2026-10-10'):
            with self.assertRaises(ValueError):
                train.weights([day], '2026-10-09')

    def test_all_three_years_used_for_every_rider(self):
        rows = [{'race_id': str(i), 'player_id': str(car), 'car_no': car,
                 'date': day, 'finish_pos': ((car-1+i)%3)+1}
                for i, day in enumerate(('2024-01-01', '2025-01-01', '2026-01-01')) for car in (1, 2, 3)]
        profiles = train.profiles(pd.DataFrame(rows), '2026-10-09', '2026-10-09')
        self.assertEqual(profiles['1']['evaluation']['races'], 3)
        self.assertEqual(profiles['1']['evaluation']['effective_races'], 7)
        np.testing.assert_allclose(profiles['1']['evaluation']['rates'], [1/7, 2/7, 4/7])

    def test_weighted_pool_changes_toward_recent_evidence(self):
        p = [[.8, .1], [.1, .8]]
        old = train.pool_weights(p, [4, 1])
        recent = train.pool_weights(p, [1, 4])
        self.assertGreater(old[0], recent[0])
        self.assertAlmostEqual(sum(recent), 1)

    def test_weighted_joint_optimizer_changes_direction(self):
        X = np.asarray([[1.], [-1.], [1.], [-1.]], dtype=np.float32)
        old = train.joint_fit(X, [2, 2], [0, 1], [4, 1])
        recent = train.joint_fit(X, [2, 2], [0, 1], [1, 4])
        self.assertGreater(old[0], 0)
        self.assertLess(recent[0], 0)


if __name__ == '__main__':
    unittest.main()
