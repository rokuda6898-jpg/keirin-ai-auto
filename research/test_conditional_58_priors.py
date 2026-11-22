import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from conditional_58_priors import PRIOR, attach_priors, daily_prior_table, predict_enriched, summarize


def history():
    return pd.DataFrame([{'race_id': str(i), 'date': date, 'player_id': str(car),
        'car_no': car, 'finish_pos': car} for i, date in enumerate(
        ['2024-01-02', '2025-01-01', '2025-01-02', '2025-01-03']) for car in range(1, 5)])


class CausalPriorTests(unittest.TestCase):
    def test_expanding_matches_existing_training_priors(self):
        import fixed_year_features as ff
        frame = history()
        reference, _ = ff.daily_priors(frame)
        expected = reference[['player_id', 'date', *PRIOR]].sort_values(['player_id', 'date']).reset_index(drop=True)
        assert_frame_equal(daily_prior_table(frame), expected, check_dtype=False)

    def test_same_day_and_future_labels_cannot_change_todays_features(self):
        frame = history()
        for horizon in (None, 365):
            original = daily_prior_table(frame, horizon)
            changed = frame.copy()
            changed.loc[changed.date.ge('2025-01-02'), 'finish_pos'] = 5 - changed.loc[changed.date.ge('2025-01-02'), 'finish_pos']
            revised = daily_prior_table(changed, horizon)
            assert_frame_equal(original[original.date.le('2025-01-02')], revised[revised.date.le('2025-01-02')])
            self.assertFalse(original[original.date.eq('2025-01-03')].equals(revised[revised.date.eq('2025-01-03')]))

    def test_window_boundary_is_left_inclusive_today_excluded(self):
        table = daily_prior_table(history(), 365)
        counts = table[table.player_id.eq('1')].set_index('date').player_prior_races
        self.assertEqual(counts['2025-01-01'], 1)
        self.assertEqual(counts['2025-01-02'], 1)
        self.assertEqual(counts['2025-01-03'], 2)
        self.assertEqual(counts['2024-01-02'], 0)

    def test_duplicate_and_invalid_horizon_rejected(self):
        with self.assertRaises(ValueError):
            daily_prior_table(pd.concat([history(), history().iloc[:1]]))
        with self.assertRaises(ValueError):
            daily_prior_table(history(), 0)

    def test_alignment_and_outcome_masking(self):
        frame = history().sample(frac=1, random_state=58).reset_index(drop=True)
        enriched = attach_priors(frame, daily_prior_table(history()))
        self.assertEqual(enriched.race_id.tolist(), frame.race_id.tolist())
        self.assertTrue(enriched.finish_pos.isna().all())
        with self.assertRaises(ValueError):
            attach_priors(frame, daily_prior_table(history()).iloc[1:])

    def test_paired_hit_changes_and_cost(self):
        from itertools import permutations
        filler = ['-'.join(map(str, t)) for t in permutations(range(4, 9), 3)][:11]
        baseline = {'a': ['1-2-3', *filler], 'b': ['1-2-3', *filler]}
        candidate = {'a': ['1-2-4', *filler], 'b': ['2-1-3', *filler]}
        summary, rows = summarize(candidate, {'a': '1-2-3', 'b': '2-1-3'},
            {'a': {'date': '2025-01-01', 'field': 9, 'class': 'A'},
             'b': {'date': '2025-01-02', 'field': 9, 'class': 'A'}}, baseline)
        self.assertEqual(summary['overall']['rescued'], 1)
        self.assertEqual(summary['overall']['lost'], 1)
        self.assertEqual(summary['overall']['flat_stake_yen'], 2400)

    def test_inference_matches_unmodified_baseline_and_masks_outcomes(self):
        import trifecta_chain_research as chain
        import fixed_year_features as ff
        path = Path(__file__).resolve().parents[1]/'models/conditional_trifecta_v1.joblib'
        if not path.exists():
            self.skipTest('baseline artifact unavailable')
        bundle = joblib.load(path)
        rows = []
        for n in (4, 7, 9):
            for car in range(1, n+1):
                rows.append({'race_id': str(n), 'date': '2026-01-01', 'player_id': str(car),
                    'car_no': car, 'entries_number': n, 'race_class': 'A級',
                    'score': 90-car, 'recent_avg_finish': car, 'finish_pos': car})
        frame = chain.frame_order(pd.DataFrame(rows))
        expected = list(chain.predict(frame, bundle))
        actual = list(predict_enriched(frame, ff.frozen_priors(frame, bundle['reference']), bundle))
        changed = frame.copy()
        changed['finish_pos'] = 99
        changed['actual_trifecta'] = '9-8-7'
        altered = list(predict_enriched(changed, ff.frozen_priors(changed, bundle['reference']), bundle))
        for a, b, c in zip(expected, actual, altered):
            self.assertEqual(a['race_id'], b['race_id'])
            self.assertEqual(a['keys'], b['keys'])
            for method in ('independent', 'chain'):
                np.testing.assert_allclose(a[method], b[method], atol=1e-14, rtol=1e-14)
                np.testing.assert_allclose(b[method], c[method], atol=1e-14, rtol=1e-14)
                self.assertEqual(len(chain.tickets(b['keys'], b[method])), 12)


if __name__ == '__main__':
    unittest.main()
