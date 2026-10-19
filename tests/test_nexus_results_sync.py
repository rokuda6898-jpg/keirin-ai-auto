import csv
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nexus_results_sync import sync_latest_results


DAY = '2026-10-19'
RACE_ID = '012320261019'
BEFORE = '2026-10-19T11:50:00+09:00'
CLOSE = '2026-10-19T12:00:00+09:00'
OBSERVED = '2026-10-19T12:10:00+09:00'


def prediction():
    return {
        'captured_at': BEFORE,
        'tickets': ['1-2-3', '2-1-3'],
        'formations': [{'first': [1, 2], 'second': [1, 2], 'third': [3]}],
        'odds': {'1-2-3': 12.0, '2-1-3': 50.0},
        'odds_at': BEFORE, 'allocation_saved_at': BEFORE, 'auto_saved_at': BEFORE,
        'allocations': {
            '5000': {'available': True, 'stakes': {'1-2-3': 4000, '2-1-3': 1000}},
            '6000': {'available': False, 'reason_code': 'no_break_even'},
        },
        'auto_budget': 5000,
    }


def race(race_id=RACE_ID, **overrides):
    value = {
        'race_id': race_id, 'date': DAY, 'venue': '取手', 'race_no': 1,
        'close_at': CLOSE, 'closed': False, 'prediction': prediction(),
        'result_source_sha256': 'original-evidence-sha',
    }
    value.update(overrides)
    return value


def outcome(race_id=RACE_ID, winners=('1-2-3',), payouts=None, **overrides):
    value = {
        'race_id': race_id, 'official_result_available': True,
        'actual_trifecta': '|'.join(winners), 'actual_trifecta_buys': list(winners),
        'payouts_trifecta_json': json.dumps({'1-2-3': 1200} if payouts is None else payouts),
        'result_source': 'winticket', 'result_observed_at_jst': OBSERVED,
        'source_url': 'https://example.test/official-result',
    }
    value.update(overrides)
    return value


class NexusResultSync(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.outputs = Path(self.temporary.name)

    def write(self, relative, value):
        path = self.outputs / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')

    def read(self, relative):
        return json.loads((self.outputs / relative).read_text(encoding='utf-8'))

    def prepare(self, rows=None, results=None, ledger=None):
        rows = [race()] if rows is None else rows
        live = {
            'schema_version': 1, 'date': DAY, 'updated_at': BEFORE,
            'policy_updated_at': '2026-10-19T11:55:00+09:00',
            'history_through': '2026-10-18', 'model': {'name': 'saved-model'},
            'coverage': {'races': len(rows)}, 'races': rows,
        }
        self.write('nexus_live.json', live)
        self.write('latest_results.json', [outcome()] if results is None else results)
        if ledger is not None:
            self.write('nexus_forecasts/' + DAY + '.json', ledger)
        return deepcopy(live)

    def sync(self):
        summary = sync_latest_results(self.outputs)
        return summary, self.read('nexus_live.json')

    def test_complete_result_settles_saved_flat_and_auto_without_reallocation(self):
        original = self.prepare()
        with patch('nexus_model.publication.allocation_snapshot', side_effect=AssertionError('must not allocate')):
            summary, live = self.sync()
        self.assertTrue(summary['changed'])
        self.assertEqual(summary['updated_races'], 1)
        row = live['races'][0]
        self.assertEqual(row['prediction'], original['races'][0]['prediction'])
        self.assertEqual(row['result_source_sha256'], 'original-evidence-sha')
        for key in original.keys() - {'races'}:
            self.assertEqual(live[key], original[key])
        self.assertEqual(live['results_updated_at'], OBSERVED)
        self.assertEqual(row['winners'], ['1-2-3'])
        self.assertEqual(row['payouts'], {'1-2-3': 1200})
        self.assertTrue(row['result_complete'])
        self.assertEqual(row['settlement']['flat']['stake'], 200)
        self.assertEqual(row['settlement']['flat']['returned'], 1200)
        self.assertEqual(row['settlement']['5000']['returned'], 48000)
        self.assertEqual(row['settlement']['auto']['budget'], 5000)
        self.assertEqual(row['settlement']['auto']['returned'], 48000)
        self.assertNotIn('6000', row['settlement'])
        self.assertEqual(self.read('nexus_forecasts/' + DAY + '.json'), live['races'])

    def test_result_without_prediction_is_still_published(self):
        row = race()
        del row['prediction']
        self.prepare([row])
        _, live = self.sync()
        row = live['races'][0]
        self.assertTrue(row['result_complete'])
        self.assertEqual(row['payouts']['1-2-3'], 1200)
        self.assertNotIn('prediction', row)
        self.assertNotIn('settlement', row)

    def test_missing_dead_heat_payout_is_pending_and_removes_stale_settlement(self):
        self.prepare([race(settlement={'flat': {'stake': 200, 'returned': 0}})],
                     [outcome(winners=('1-2-3', '2-1-3'), payouts={'1-2-3': 1200})])
        _, live = self.sync()
        row = live['races'][0]
        self.assertEqual(row['winners'], ['1-2-3', '2-1-3'])
        self.assertFalse(row['result_complete'])
        self.assertTrue(row['payout_pending'])
        self.assertNotIn('settlement', row)

    def test_complete_dead_heat_adds_both_winning_ticket_returns(self):
        self.prepare(results=[outcome(winners=('1-2-3', '2-1-3'),
                                      payouts={'1-2-3': 1200, '2-1-3': 1500})])
        _, live = self.sync()
        row = live['races'][0]
        self.assertTrue(row['result_complete'])
        self.assertEqual(row['settlement']['flat']['returned'], 2700)
        self.assertEqual(row['settlement']['auto']['returned'], 63000)

    def test_newer_incomplete_result_cannot_downgrade_complete_result(self):
        self.prepare()
        self.sync()
        before = self.read('nexus_live.json')['races'][0]
        self.write('latest_results.json', [outcome(winners=('1-2-3', '2-1-3'),
            payouts={'1-2-3': 1200}, result_observed_at_jst='2026-10-19T12:20:00+09:00')])
        _, live = self.sync()
        self.assertEqual(live['races'][0], before)

    def test_older_result_cannot_overwrite_newer_official_result(self):
        self.prepare(results=[outcome(result_observed_at_jst='2026-10-19T12:30:00+09:00')])
        self.sync()
        before = self.read('nexus_live.json')['races'][0]
        self.write('latest_results.json', [outcome(winners=('3-2-1',), payouts={'3-2-1': 2000})])
        summary, live = self.sync()
        self.assertFalse(summary['changed'])
        self.assertEqual(live['races'][0], before)
        self.assertEqual(live['results_updated_at'], '2026-10-19T12:30:00+09:00')

    def test_partial_refunds_and_original_receipt_are_preserved(self):
        self.prepare([race(refunded=['2-1-3'], result_captured_at='2026-10-19T12:05:00+09:00')])
        _, live = self.sync()
        row = live['races'][0]
        self.assertEqual(row['refunded'], ['2-1-3'])
        self.assertEqual(row['result_source_sha256'], 'original-evidence-sha')
        self.assertEqual(row['result_captured_at'], '2026-10-19T12:05:00+09:00')
        auto = row['settlement']['auto']
        self.assertEqual((auto['stake'], auto['net_stake'], auto['refund']), (5000, 4000, 1000))
        self.assertEqual((auto['winnings'], auto['returned']), (48000, 49000))
        self.assertTrue(all(isinstance(auto[key], int) for key in ('stake', 'net_stake', 'refund', 'winnings', 'returned')))

    def test_stale_ledger_cannot_roll_back_a_newer_live_result(self):
        original = self.prepare(results=[outcome(result_observed_at_jst='2026-10-19T12:30:00+09:00')])
        self.sync()
        latest = self.read('nexus_live.json')['races'][0]
        self.write('nexus_forecasts/' + DAY + '.json', original['races'])
        self.write('latest_results.json', [outcome(winners=('3-2-1',), payouts={'3-2-1': 2000})])
        _, live = self.sync()
        self.assertEqual(live['races'][0], latest)
        self.assertEqual(self.read('nexus_forecasts/' + DAY + '.json'), [latest])

    def test_legacy_complete_result_is_not_downgraded_by_missing_payouts(self):
        original = race(winners=['1-2-3'], payouts={'1-2-3': 1200},
                        result_captured_at='2026-10-19T12:05:00+09:00')
        self.prepare([original], results=[outcome(payouts={})])
        _, live = self.sync()
        self.assertEqual(live['races'][0], original)

    def test_full_refund_takes_precedence_over_compact_outcome(self):
        for field in ('refund_all', 'cancelled'):
            with self.subTest(field=field):
                self.prepare([race(**{field: True}, payouts={}, winners=[],
                                   refund_reason='official cancellation')], ledger=[])
                _, live = self.sync()
                row = live['races'][0]
                self.assertTrue(row[field])
                self.assertEqual(row['refund_reason'], 'official cancellation')
                self.assertEqual(row['payouts'], {})
                auto = row['settlement']['auto']
                self.assertEqual((auto['stake'], auto['returned'], auto['net_stake']), (5000, 5000, 0))
                self.assertEqual(auto['winnings'], 0)
                self.assertTrue(auto['void'])

    def test_postclose_or_unknown_prediction_has_no_settlement(self):
        for saved_at in (CLOSE, '2026-10-19T12:01:00+09:00', None):
            with self.subTest(saved_at=saved_at):
                row = race()
                row['prediction']['captured_at'] = saved_at
                self.prepare([row], ledger=[])
                _, live = self.sync()
                self.assertNotIn('settlement', live['races'][0])
                self.assertEqual(live['races'][0]['prediction'], row['prediction'])
                self.assertTrue(live['races'][0]['result_complete'])

    def test_postclose_allocation_scores_only_existing_flat_tickets(self):
        for key in ('odds_at', 'allocation_saved_at', 'auto_saved_at'):
            with self.subTest(key=key):
                row = race()
                row['prediction'][key] = CLOSE
                self.prepare([row], ledger=[])
                _, live = self.sync()
                self.assertEqual(set(live['races'][0]['settlement']), {'flat'})
                self.assertEqual(live['races'][0]['prediction'], row['prediction'])

    def test_second_run_does_not_rewrite_files(self):
        self.prepare()
        self.sync()
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in self.outputs.rglob('*.json')}
        with patch('nexus_results_sync._write', side_effect=AssertionError('must be idempotent')):
            summary, _ = self.sync()
        self.assertFalse(summary['changed'])
        self.assertEqual(summary['updated_races'], 0)
        self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mtime_ns)
                                 for path in self.outputs.rglob('*.json')})

    def test_multiple_days_and_ledger_only_forecasts_are_retained(self):
        old_id, extra_id = '014620261018', '022320261019'
        old = race(old_id, date='2026-10-18')
        extra = race(extra_id, race_no=2)
        self.prepare(ledger=[race(), extra], results=[outcome(), outcome(old_id)])
        self.write('nexus_forecasts/2026-10-18.json', [old])
        _, live = self.sync()
        by_id = {row['race_id']: row for row in live['races']}
        self.assertEqual(set(by_id), {RACE_ID, old_id, extra_id})
        self.assertEqual(by_id[extra_id], extra)
        self.assertEqual(by_id[old_id]['prediction'], old['prediction'])
        self.assertEqual(self.read('nexus_forecasts/2026-10-18.json'), [by_id[old_id]])
        self.assertEqual(len(self.read('nexus_forecasts/' + DAY + '.json')), 2)

    def test_new_result_only_race_uses_schedule_metadata(self):
        self.prepare(rows=[])
        with (self.outputs / 'latest_race_schedule.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['race_id', 'date', 'venue', 'race_no', 'close_at', 'start_at', 'source_url'])
            writer.writeheader()
            writer.writerow({'race_id': RACE_ID, 'date': DAY, 'venue': '取手', 'race_no': 1,
                             'close_at': 1792378800, 'start_at': 1792379100,
                             'source_url': 'https://example.test/race'})
        _, live = self.sync()
        row = live['races'][0]
        self.assertEqual((row['venue'], row['race_no'], row['date']), ('取手', 1, DAY))
        self.assertEqual(row['source_url'], 'https://example.test/race')
        self.assertTrue(row['close_at'].endswith('+09:00'))
        self.assertTrue(row['result_complete'])
        self.assertNotIn('settlement', row)

    def test_exact_legacy_odds_and_fractional_payout_rejection(self):
        self.prepare(results=[outcome(payouts={}, actual_trifecta_odds='19.9')])
        _, live = self.sync()
        self.assertEqual(live['races'][0]['payouts'], {'1-2-3': 1990})
        self.prepare(results=[outcome(payouts={'1-2-3': '1200.5'})], ledger=[])
        _, live = self.sync()
        self.assertFalse(live['races'][0]['result_complete'])
        self.assertNotIn('settlement', live['races'][0])

    def test_missing_feeds_noop_and_corrupt_feed_raises(self):
        self.assertEqual(sync_latest_results(self.outputs)['reason'], 'missing_results')
        self.write('latest_results.json', [])
        self.assertEqual(sync_latest_results(self.outputs)['reason'], 'missing_nexus_feed')
        (self.outputs / 'nexus_live.json').write_text('{', encoding='utf-8')
        with self.assertRaises(json.JSONDecodeError):
            sync_latest_results(self.outputs)


if __name__ == '__main__':
    unittest.main()
