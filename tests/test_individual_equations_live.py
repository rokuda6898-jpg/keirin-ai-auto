import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import individual_equations_live as live


class IndependentEquationTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 9, 12, tzinfo=live.JST)
        self.meta = {'race_id': 'r1', 'date': '2026-10-09', 'venue': 'test',
                     'race_no': 1, 'close_at': (self.now + timedelta(minutes=10)).timestamp()}
        self.methods = {'a': {'basis': 'trained', 'tickets': [
            {'buy': '1-2-3', 'stake_yen': 100}, {'buy': '1-3-2', 'stake_yen': 100}]},
            'b': {'basis': 'trained', 'tickets': [
            {'buy': '2-1-3', 'stake_yen': 100}, {'buy': '3-1-2', 'stake_yen': 100}]}}
        self.models = {'refit': {'training_cutoff_exclusive': '2026-10-09'}}

    def test_independent_tickets_same_budget_and_immutable_first_record(self):
        rows = []
        self.assertTrue(live.freeze(rows, 'archive_models', self.meta, self.methods, self.models, self.now))
        self.assertNotEqual(rows[0]['methods']['a']['tickets'], rows[0]['methods']['b']['tickets'])
        first = rows[0]['sha256']
        self.assertFalse(live.freeze(rows, 'archive_models', self.meta, self.methods, self.models,
                                    self.now + timedelta(minutes=1)))
        self.assertEqual(first, rows[0]['sha256'])

    def test_reject_after_close_unequal_budget_and_future_training(self):
        self.assertFalse(live.freeze([], 'archive_models', self.meta, self.methods, self.models,
                                    self.now + timedelta(minutes=10)))
        with self.assertRaisesRegex(ValueError, 'training'):
            live.freeze([], 'archive_models', self.meta, self.methods,
                        {'refit': {'training_cutoff_exclusive': '2026-10-10'}}, self.now)
        self.methods['b']['tickets'].pop()
        with self.assertRaisesRegex(ValueError, 'budget'):
            live.freeze([], 'archive_models', self.meta, self.methods, self.models, self.now)

    def test_unknown_payout_hits_are_counted_but_not_treated_as_zero_roi(self):
        rows = [{'race_id': 'r1', 'methods': self.methods}]
        outcome = {'r1': {'winning_buys': ['1-2-3'], 'payouts': {}, 'payout_complete': False}}
        a = live.metrics(rows, outcome, 'a', 12)
        self.assertEqual(a['hit_rate'], 1)
        self.assertIsNone(a['roi'])
        self.assertEqual(a['payout_pending_races'], 1)
        self.assertEqual(a['stake_yen'], 0)

    def test_tie_payouts_and_independent_results(self):
        rows = [{'race_id': 'r1', 'methods': self.methods}]
        outcome = {'r1': {'winning_buys': ['1-2-3', '1-3-2'],
                          'payouts': {'1-2-3': 5000, '1-3-2': 10000}, 'payout_complete': True}}
        a = live.metrics(rows, outcome, 'a', 12)
        b = live.metrics(rows, outcome, 'b', 12)
        self.assertEqual(a['return_yen'], 15000)
        self.assertEqual(a['roi'], 75)
        self.assertEqual(a['hits'], 1)
        self.assertEqual(a['hits_at_least_100x'], 1)
        self.assertEqual(b['roi'], 0)
        self.assertEqual(b['hits'], 0)

    def test_conflicting_results_excluded_and_missing_payout_can_be_filled(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            rows = [{'race_id': 'r1', 'close_at': self.now.timestamp(), 'methods': self.methods}]
            result = {'race_id': 'r1', 'official_result_available': True, 'actual_trifecta': '1-2-3'}
            with patch.object(live, 'clock', return_value=self.now + timedelta(hours=1)):
                live.update_results(folder, rows, [result])
                outcome = live.update_results(folder, rows, [{**result, 'actual_trifecta_odds': 70.1}])
                self.assertTrue(outcome['r1']['payout_complete'])
                outcome = live.update_results(folder, rows, [{**result, 'actual_trifecta': '3-2-1'}])
            self.assertTrue(outcome['r1']['conflict'])
            self.assertEqual(live.metrics(rows, outcome, 'a', 12)['settled_races'], 0)

    def test_merge_preserves_published_picks_and_checks_seals(self):
        with tempfile.TemporaryDirectory() as left, tempfile.TemporaryDirectory() as right:
            local, remote = Path(left), Path(right)
            rows = []
            live.freeze(rows, 'archive_models', self.meta, self.methods, self.models, self.now)
            live.save(remote / 'forecasts.json', rows)
            live.save(local / 'forecasts.json', [])
            with patch.object(live, 'report', return_value={}):
                live.merge(remote, local)
            self.assertEqual(live.ledger(local), rows)
            rows[0]['methods']['a']['tickets'][0]['buy'] = '9-8-7'
            live.save(local / 'forecasts.json', rows)
            with self.assertRaisesRegex(ValueError, 'corrupt'):
                live.ledger(local)


if __name__ == '__main__':
    unittest.main()
