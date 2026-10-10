import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import trifecta_chain_research as study
from test_fixed_year_research import fixture


class ChainTests(unittest.TestCase):
    def test_joint_distribution_preserves_first_and_conditional_marginals(self):
        rng = np.random.default_rng(67)
        for n in (4, 7, 9):
            pairs, triples = study.hypotheses(n)
            first, second, third = rng.uniform(.01, 1, (3, n))
            pp, tp = rng.uniform(.01, 1, len(pairs)), rng.uniform(.01, 1, len(triples))
            independent, chain = study.combine(first, second, third, pp, tp, pairs, triples)
            for p in (independent, chain):
                self.assertAlmostEqual(sum(p), 1.)
                self.assertTrue((p > 0).all())
                np.testing.assert_allclose(np.bincount(triples[:, 0], weights=p), first/first.sum())
            for i, j in pairs:
                pair_indices = pairs[:, 0] == i
                triple_indices = (triples[:, 0] == i) & (triples[:, 1] == j)
                expected = first[i]/first.sum()*pp[(pairs == [i, j]).all(axis=1)][0]/pp[pair_indices].sum()
                self.assertAlmostEqual(chain[triple_indices].sum(), expected)
            keys = ['-'.join(str(i+1) for i in t) for t in triples]
            self.assertEqual(len(set(study.tickets(keys, chain))), 12)

    def test_conditional_fit_rows_exclude_prefix_but_keep_one_positive_per_race(self):
        data = fixture(3, n=7)
        for position in (2, 3):
            target, first, second, labels = study.training_indices(data, position)
            self.assertEqual(len(target), 3*(8-position))
            self.assertEqual(labels.sum(), 3)
            self.assertTrue((target != first).all())
            self.assertTrue(data.iloc[first].finish_pos.eq(1).all())
            if position == 3:
                self.assertTrue((target != second).all())
                self.assertTrue((first != second).all())
                self.assertTrue(data.iloc[second].finish_pos.eq(2).all())

    def test_impossible_prefix_and_bad_joint_scores_fail(self):
        data, enriched, _ = study.enrich_training(fixture(4), '2025-01-01')
        X, _ = study.feature_matrix(enriched)
        with self.assertRaisesRegex(ValueError, 'distinct'):
            study.design(X, [0], [0])
        with self.assertRaisesRegex(ValueError, 'distinct'):
            study.design(X, [0], [1], [1])
        with self.assertRaisesRegex(ValueError, 'invalid'):
            study.tickets(['1-2-3']*12, [float('nan')]*12)

    def test_training_rejects_boundary_or_future_before_any_fit(self):
        with self.assertRaisesRegex(ValueError, 'cutoff'):
            study.fit(fixture(start='2025-10-09'), '2025-10-09', 3, Path('.'), fast=True)

    def test_fitted_forecasts_ignore_target_results_and_serialization_order(self):
        with tempfile.TemporaryDirectory() as td, patch('builtins.print'):
            bundle = study.fit(fixture(50, n=7), '2025-01-01', 7, Path(td), fast=True)
            target = fixture(2, start='2025-10-09', n=7)
            original = list(study.predict(target, bundle, batch_size=1))
            changed = target.sample(frac=1., random_state=7).copy()
            changed['finish_pos'] = 1
            changed['official_finish_pos'] = 99
            changed['actual_trifecta'] = '7-6-5'
            changed['result_factor'] = 'changed'
            changed['result_event_back'] = True
            changed['payout'] = 1000000
            np.testing.assert_equal(original, list(study.predict(changed, bundle, batch_size=2)))
            self.assertNotIn('player_id_code', bundle['features'])
            self.assertNotIn('wind_speed', bundle['features'])
            for row in original:
                self.assertEqual(len(row['keys']), 210)
                self.assertEqual(len(study.tickets(row['keys'], row['chain'])), 12)
            with self.assertRaisesRegex(ValueError, 'precede'):
                list(study.predict(fixture(start='2024-01-01'), bundle))


if __name__ == '__main__':
    unittest.main()
