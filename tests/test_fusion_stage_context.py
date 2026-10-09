import copy
import gzip
import json
import tempfile
import unittest
from itertools import permutations
from pathlib import Path
from unittest.mock import patch

import numpy as np

import fusion_equations as fusion
import fusion_stage_context as stage
import fusion_stage_study as extension
import fixed_year_study as old
from fixed_year_research import inventory
from test_fixed_year_research import fixture


def example(n=4, experts=3):
    keys = ['-'.join(map(str, t)) for t in permutations(range(1, n + 1), 3)]
    rng = np.random.default_rng(721)
    matrix = rng.uniform(.1, 1., (experts, len(keys)))
    matrix /= matrix.sum(axis=1)[:, None]
    return keys, matrix


class StageContextTests(unittest.TestCase):
    def test_chain_reconstructs_experts_for_different_fields(self):
        for n in (3, 5, 7, 9):
            keys, matrix = example(n)
            parts = stage.decompose(keys, matrix)
            np.testing.assert_allclose(parts.prod(axis=1), matrix, atol=1e-15)
            model = {'names': ['a', 'b', 'c'], 'global_joint': [.1, .2, .7],
                     'global_stages': [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 'groups': {}}
            got = stage.raw_predictions(model, keys, matrix, 'new_context')
            np.testing.assert_allclose(got['stage_global'], parts[0, 0] * parts[1, 1] * parts[2, 2])
            for p in got.values(): self.assertAlmostEqual(p.sum(), 1., places=12)
            np.testing.assert_array_equal(got['stage_global'], got['stage_context'])

    def test_one_expert_and_duplicate_alias_are_not_extra_votes(self):
        keys, matrix = example(experts=1)
        p = dict(zip(keys, matrix[0]))
        pool = fusion.fit_pool([(keys[0], {'a': p, 'clone': p})])
        keys2, m = stage.expert_matrix(pool, {'a': p, 'clone': p})
        self.assertEqual(m.shape[0], 1)
        np.testing.assert_array_equal(m[0], [p[k] for k in keys2])
        model = {'names': ['a'], 'global_joint': [1], 'global_stages': [[1], [1], [1]], 'groups': {}}
        for out in stage.raw_predictions(model, keys, matrix, 'unknown').values():
            np.testing.assert_allclose(out, matrix[0], atol=1e-15)

    def test_invalid_or_missing_experts_fail_closed(self):
        keys, matrix = example()
        with self.assertRaises(ValueError): stage.decompose(keys[:-1], matrix[:, :-1])
        bad = matrix.copy(); bad[0, 0] = np.nan
        with self.assertRaises(ValueError): stage.decompose(keys, bad)
        with self.assertRaises(ValueError): stage.expert_matrix({'weights': {'a': 1}, 'aliases': {'a': ['a']}}, {})

    def test_optimizer_reproduces_existing_global_pool(self):
        rng = np.random.default_rng(321)
        keys, _ = example(); records = []
        for i in range(140):
            m = rng.uniform(.01, 1., (3, len(keys))); m /= m.sum(axis=1)[:, None]
            records.append((keys[i % len(keys)], {str(e): dict(zip(keys, p)) for e, p in enumerate(m)}))
        pool = fusion.fit_pool(records)
        p = [[es[n][y] for n in sorted(pool['weights'])] for y, es in records]
        np.testing.assert_allclose(stage.fit_weights(p), list(pool['weights'].values()), atol=1e-14)

    def test_contexts_shrink_and_sparse_groups_fall_back(self):
        keys, matrix = example()
        available = {str(e): dict(zip(keys, p)) for e, p in enumerate(matrix)}
        records = [(keys[i % len(keys)], available) for i in range(140)]
        pool = fusion.fit_pool(records)
        parts = stage.decompose(keys, matrix)
        joint = [matrix[:, i % len(keys)] for i in range(140)]
        stages = [parts[:, :, i % len(keys)] for i in range(140)]
        dates = [f'2024-01-{i % 28 + 1:02}' for i in range(140)]
        contexts = ['7|A'] * 120 + ['9|S'] * 20
        fitted = stage.fit(joint, stages, contexts, dates, pool, '2025-10-09')
        self.assertEqual(set(fitted['groups']), {'7|A'})
        self.assertAlmostEqual(fitted['groups']['7|A']['local_fraction'], 120 / 420)
        with self.assertRaisesRegex(ValueError, 'before'):
            stage.fit(joint, stages, contexts, ['2026-01-01'] * 140, pool, '2025-10-09')

    def test_calibration_can_reject_new_component_and_rejects_temporal_leakage(self):
        model = {'fit_last': '2024-01-31', 'cutoff_exclusive': '2025-10-09'}
        rec = {'date': '2024-02-01', 'race_id': '1', 'actual_index': 0, 'baseline': np.array([.9, .1]),
               'raw': {n: np.array([.1, .9]) for n in stage.VARIANTS}}
        cal = stage.calibrate([rec], model)
        for c in cal['variants'].values(): self.assertEqual(c['blend'], 0)
        for date in ('2024-01-31', '2025-10-09', '2026-01-01'):
            with self.assertRaises(ValueError): stage.calibrate([{**rec, 'date': date}], model)
        with self.assertRaises(ValueError): stage.calibrate([rec, rec], model)
        with self.assertRaises(ValueError): stage.calibrate([{**rec, 'raw': {}}], model)

    def test_full_research_flow_preserves_baselines_and_never_uses_prediction_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'original'; source.mkdir()
            folder = Path(tmp) / 'extension'; folder.mkdir()
            training = fixture(50, n=3); target = fixture(3, start='2026-01-01', n=3)
            old.train(training, source, '2025-10-09', fast=True)
            raw = Path(tmp) / 'fixture.csv'; training.to_csv(raw, index=False)
            old.write_json(source / 'inventory.json', inventory(training, raw, '2026-10-09'))
            old.predict(target, source); old.assess(target, source)
            extension.train(training, source, folder)
            extension.predict(target, source, folder)
            path = folder / 'stage_predictions.jsonl.gz'
            with gzip.open(path, 'rt', encoding='utf-8') as h: before = h.read()
            changed = target.copy(); changed['finish_pos'] = 1; changed['official_payout'] = 999999
            with patch.object(old, 'outcome', side_effect=AssertionError('prediction read labels')):
                extension.predict(changed, source, folder)
            with gzip.open(path, 'rt', encoding='utf-8') as h: self.assertEqual(before, h.read())
            result = extension.assess(target, source, folder)
            self.assertEqual(result['evaluated_races'], 3)
            self.assertEqual(len(result['summary']['all']['methods']), 5)
            self.assertFalse(result['ceo_integration']); self.assertIsNone(result['roi'])
            with path.open('ab') as h: h.write(b'modified')
            with self.assertRaisesRegex(ValueError, 'evidence changed'): extension.assess(target, source, folder)


if __name__ == '__main__':
    unittest.main()
