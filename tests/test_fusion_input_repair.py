import gzip
import json
import tempfile
import unittest
from collections import Counter
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

import fixed_year_features as features
import fixed_year_study as old
import fusion_input_repair as repair
from fusion_input_repair_assess import diagnostics, assess
from fixed_year_research import inventory
from test_fixed_year_research import fixture


class InputRepairTests(unittest.TestCase):
    def test_feature_removal_applies_to_fitting_and_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = fixture(12, n=3)
            bundle = repair.fit_base(data, repair.ARMS['both'], Path(tmp), repair.boundary(data), fast=True)
            enriched = features.frozen_priors(fixture(2, start='2026-01-01', n=3), bundle['reference'])
            X, _ = features.matrix(enriched, bundle['feature_config'])
            for classifier in bundle['classifiers'].values():
                expected = classifier.predict_proba(X)
                changed = X.copy()
                changed['player_prior_days_since_last_race'] = np.arange(len(changed))*10000
                changed['player_id_code'] = -999
                np.testing.assert_array_equal(expected, classifier.predict_proba(changed))
                self.assertNotIn('player_id_code', classifier.columns)
                self.assertNotIn('player_prior_days_since_last_race', classifier.columns)
                self.assertIn('days_since_last_race', classifier.columns)
                self.assertIn('player_prior_win_rate', classifier.columns)

    def test_no_stacking_forecast_can_overlap_its_training(self):
        bundle = {'base_training_last': '2024-01-02'}
        for day in ('2024-01-01', '2024-01-02'):
            with self.assertRaisesRegex(ValueError, 'strictly later'):
                repair.later_prediction(fixture(start=day), bundle)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'prediction cutoff'):
                repair.fit_base(fixture(2), (), Path(tmp), '2024-01-02', True)

    def test_anchor_changes_only_first_marginals(self):
        rng = np.random.default_rng(3400)
        keys = ['-'.join(map(str, t)) for t in permutations(range(1, 6), 3)]
        a, b = rng.random((2, len(keys)))
        a /= a.sum()
        b /= b.sum()
        old_p, base = dict(zip(keys, a)), dict(zip(keys, b))
        result = repair.anchor_first(old_p, base)
        target, got = Counter(), Counter()
        for k in keys:
            target[k.split('-')[0]] += base[k]
            got[k.split('-')[0]] += result[k]
        np.testing.assert_allclose(list(got.values()), list(target.values()), atol=1e-15)
        self.assertAlmostEqual(sum(result.values()), 1.)
        for head in got:
            subset = [k for k in keys if k.startswith(head+'-')]
            original_mass = sum(old_p[k] for k in subset)
            np.testing.assert_allclose([result[k]/got[head] for k in subset],
                                       [old_p[k]/original_mass for k in subset])

    def test_top12_diagnostics_partition_every_possible_result(self):
        keys = ['-'.join(map(str, t)) for t in permutations(range(1, 6), 3)]
        mass = sum(range(1, len(keys)+1))
        distribution = {k: (len(keys)-i)/mass for i, k in enumerate(keys)}
        reasons = Counter(diagnostics(distribution, k)['reason'] for k in keys)
        self.assertEqual(reasons['hit'], 12)
        self.assertEqual(sum(reasons.values()), len(keys))
        self.assertEqual(set(reasons), {'hit', 'first_absent_from_top12'})
        third = diagnostics(distribution, keys[12])
        self.assertEqual(third['actual_rank_13_to_24'], 1)
        first12 = ['1-2-3'] + [k for k in keys if k.startswith('2-')][:11]
        ordered = first12 + [k for k in keys if k not in first12]
        mixed = {k: (len(keys)-i)/mass for i, k in enumerate(ordered)}
        self.assertEqual(diagnostics(mixed, '1-2-4')['reason'], 'third_missing_for_true_pair')
        self.assertEqual(diagnostics(mixed, '1-3-4')['reason'], 'true_pair_absent_from_top12')

    def test_all_predefined_arms_separate_fit_predict_score_and_preserve_baselines(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, output = root/'source', root/'repair'
            source.mkdir()
            training = fixture(50, n=3)
            target = fixture(3, start='2026-01-01', n=3)
            data = pd.concat([training, target], ignore_index=True)
            history = root/'history.csv'
            data.to_csv(history, index=False)
            old.write_json(source/'inventory.json', inventory(data, history, '2026-10-09'))
            old.train(training, source, '2025-10-09', fast=True)
            old.predict(target, source)
            baseline = old.assess(target, source)
            baseline_hash = old.digest_file(source/'holdout_predictions.jsonl.gz')
            with self.assertRaisesRegex(ValueError, 'target-year'):
                repair.train(target, source, output/'gap', 'gap', fast=True)
            for arm in repair.ARMS:
                folder = output/arm
                repair.train(training, source, folder, arm, fast=True)
                repair.predict(target, source, folder)
                manifest = repair.read_json(folder/'manifest.json')
                for step in manifest['chronological_base_forecasts']:
                    self.assertLess(step['fit_last'], step['predict_first'])
                self.assertEqual(manifest['final_base_training_last'], max(training.date) if arm == 'refit'
                                 else max(old.phase_split(training)[0].date))
                with gzip.open(folder/'predictions.jsonl.gz', 'rt', encoding='utf-8') as handle:
                    first = handle.read()
                changed = target.copy()
                changed['finish_pos'] = 4-changed.finish_pos
                changed['result_event_back'] = True
                changed['actual'] = '9-8-7'
                changed['payout'] = 999999
                repair.predict(changed, source, folder)
                with gzip.open(folder/'predictions.jsonl.gz', 'rt', encoding='utf-8') as handle:
                    self.assertEqual(first, handle.read())
                for row in map(json.loads, first.splitlines()):
                    self.assertEqual(len(row['department_scenarios']), 7)
                    self.assertNotIn('actual', row)
            result = assess(target, source, output)
            self.assertEqual(result['evaluated_races'], 3)
            self.assertEqual(result['summary']['all']['methods']['risk'],
                             baseline['summary']['all']['methods']['risk'])
            self.assertEqual(old.digest_file(source/'holdout_predictions.jsonl.gz'), baseline_hash)
            self.assertFalse(result['ceo_integration'])
            self.assertFalse(result['main_hole_comparison_available'])
            self.assertIsNone(result['roi'])
            self.assertTrue((output/'fusion_input_repair_report.html').exists())
            with (output/'gap'/'model.joblib').open('ab') as handle:
                handle.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                repair.predict(target, source, output/'gap')


if __name__ == '__main__':
    unittest.main()
