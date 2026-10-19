"""Recorded final payoff must settle unchanged plans after the date rolls over.

Fixtures are unchanged public repository snapshots from main 610d8bac:
  official page blob f3a6bd3276176dbb2f4cf940dd5c134bf504c3c1
  near-close feed blob 72b388054949812122724af41a98473de7815cbc
The page really has decidedAt=0 and a final 1-2-4 payoff of JPY 2,270.
"""
import ast
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import conditional_budget_near_close as near
from conditional_budget import settle
from nexus_results_sync import read_saved_page_outcome, sync_latest_results
from official_outcomes import normalize_outcome

FIXTURES = ROOT / 'tests/fixtures'
RID = '096220261019'
DAY = '2026-10-19'
NEXT_DAY = datetime.fromisoformat('2026-10-20T02:43:40+09:00')


def conditional_helpers():
    # Isolate result-only helpers without loading model dependencies/assets.
    tree = ast.parse((ROOT / 'src/conditional_live.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('saved_outcome_evidence', 'apply_result_evidence')]
    scope = dict(datetime=datetime, normalize_outcome=normalize_outcome, settle_budget=settle)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'conditional_result_helpers', 'exec'), scope)
    return scope['saved_outcome_evidence'], scope['apply_result_evidence']


class SavedFinalPayoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = json.loads((FIXTURES / 'saved_outcome_hiroshima_r9_20261019.json').read_text())
        self.baseline = json.loads((FIXTURES / 'conditional_near_close_20261019.json').read_text())
        self.path = self.root / 'data/raw/nexus_page_snapshots' / DAY / 'outcomes' / (RID + '.json')
        self.path.parent.mkdir(parents=True)
        self.write_raw(self.raw)
        self.originals = [dict(race_id=r['id'], date=r['date'], venue=r['venue'], race_no=r['number'],
            close_at=r['close_at'], snapshot_at_jst=r['prediction_snapshot_at_jst'],
            top12=copy.deepcopy(r['tickets']), budget=copy.deepcopy(r['initial_budget']),
            cancelled_cars=[], actual=[], payouts={},
            budget_official_outcome=copy.deepcopy(r.get('official_outcome')))
            for r in self.baseline['races']]
        self.row = next(r for r in self.originals if r['race_id'] == RID)
        self.outputs = self.root / 'outputs'
        self.outputs.mkdir()
        self.write_json('latest_results.json', [])
        self.write_json('conditional_near_close.json', self.baseline)
        ledger = self.root / near.LEDGER
        ledger.parent.mkdir(parents=True)
        ledger.write_text(''.join(json.dumps(r) + '\n' for r in self.originals))

    def write_raw(self, value):
        self.path.write_text(json.dumps(value))

    def write_json(self, name, value):
        path = self.outputs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_actual_final_page_certifies_payoff_without_decided_timestamp(self):
        self.assertEqual(self.raw['data']['race']['decidedAt'], 0)
        with patch('socket.socket', side_effect=AssertionError('network forbidden')):
            evidence = read_saved_page_outcome(self.root, self.row)
        self.assertEqual(evidence['outcome']['winning_buys'], ['1-2-4'])
        self.assertEqual(evidence['outcome']['payouts'], {'1-2-4': 2270})
        self.assertTrue(evidence['outcome']['payout_complete'])
        self.assertIsNone(evidence['outcome']['decided_at'])
        self.assertEqual(evidence['cancelled_cars'], [])
        self.assertEqual(evidence['refunded'], [])

    def test_next_day_empty_results_settles_all_five_saved_comparisons_without_reallocation(self):
        ledger_bytes = (self.root / near.LEDGER).read_bytes()
        raw_bytes = self.path.read_bytes()
        before = copy.deepcopy(self.baseline['races'])
        with patch('socket.socket', side_effect=AssertionError('network forbidden')), \
                patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')), \
                patch.object(near, 'allocate', side_effect=AssertionError('must not reallocate')):
            payload = near.run(self.root, clock=lambda: NEXT_DAY)
        self.assertEqual(payload['comparison']['paired_races'], 5)
        for name in ('initial', 'near_close'):
            stats = payload['comparison'][name]
            self.assertEqual((stats['races'], stats['hits'], stats['invested_yen'], stats['payout_yen']),
                             (5, 0, 25000, 0))
        frozen_keys = ('id', 'date', 'number', 'close_at', 'prediction_snapshot_at_jst',
                       'tickets', 'initial_budget', 'budget', 'registered_at_jst', 'source_sha256')
        for original, updated in zip(before, payload['races']):
            self.assertEqual({k: original.get(k) for k in frozen_keys},
                             {k: updated.get(k) for k in frozen_keys})
        self.assertEqual((self.root / near.LEDGER).read_bytes(), ledger_bytes)
        self.assertEqual(self.path.read_bytes(), raw_bytes)
        self.assertEqual(json.loads((self.outputs / 'latest_results.json').read_text()), [])

    def test_same_official_evidence_settles_initial_conditional_plan_without_mutating_forecast(self):
        get_evidence, apply_evidence = conditional_helpers()
        row = copy.deepcopy(self.row)
        # A probability sentinel verifies that the result-only path never ranks.
        row['probabilities'] = list(range(len(row['top12'])))
        frozen = {k: copy.deepcopy(row[k]) for k in ('top12', 'probabilities', 'budget', 'snapshot_at_jst')}
        evidence = get_evidence(self.root, row)
        apply_evidence(row, None, evidence)
        self.assertEqual(row['actual'], ['1-2-4'])
        self.assertEqual(row['payouts'], {'1-2-4': 2270})
        self.assertEqual(row['budget_settlement']['status'], 'settled')
        self.assertFalse(row['budget_settlement']['hit'])
        self.assertEqual({k: row[k] for k in frozen}, frozen)
        # Missing raw and an empty next-day compact feed cannot erase it.
        self.path.unlink()
        apply_evidence(row, None, get_evidence(self.root, row))
        self.assertEqual(row['budget_settlement']['status'], 'settled')
        self.assertEqual(row['actual'], ['1-2-4'])

    def test_nexus_finds_prior_day_saved_ledger_when_new_live_feed_and_results_omit_it(self):
        prediction = dict(captured_at='2026-10-19T23:00:00+09:00',
            tickets=['1-2-4', '2-1-4'], odds={'1-2-4': 20, '2-1-4': 30},
            odds_at='2026-10-19T23:00:00+09:00', allocation_saved_at='2026-10-19T23:00:00+09:00',
            auto_saved_at='2026-10-19T23:00:00+09:00', auto_budget=5000,
            allocations={'5000': {'available': True, 'stakes': {'1-2-4': 3000, '2-1-4': 2000}}})
        row = dict(race_id=RID, date=DAY, race_no=9, close_at=self.row['close_at'],
                   prediction=prediction, result_source_sha256='original-receipt')
        self.write_json('nexus_live.json', dict(date='2026-10-20', races=[]))
        self.write_json('nexus_forecasts/' + DAY + '.json', [row])
        with patch('socket.socket', side_effect=AssertionError('network forbidden')), \
                patch('nexus_model.publication.allocation_snapshot', side_effect=AssertionError('must not allocate')):
            result = sync_latest_results(self.outputs)
        self.assertEqual(result['explicit_source_results'], 1)
        saved = json.loads((self.outputs / 'nexus_forecasts' / (DAY + '.json')).read_text())[0]
        self.assertEqual(saved['prediction'], prediction)
        self.assertEqual(saved['result_source_sha256'], 'original-receipt')
        self.assertEqual(saved['winners'], ['1-2-4'])
        self.assertEqual(saved['settlement']['flat']['winnings'], 2270)
        self.assertEqual(saved['settlement']['auto']['winnings'], 68100)

    def test_in_progress_mismatched_incomplete_or_unconfirmed_refund_pages_remain_pending(self):
        def winner(page):
            return next(q for q in page['data']['trifecta'] if q['id'] in page['data']['trifectaWinningOddsIds'])
        changes = [
            ('wrong race', lambda p: p['meta'].update(race_id='086220261019')),
            ('wrong day', lambda p: p['data']['schedule'].update(date='20261020')),
            ('wrong deadline', lambda p: p['data']['race'].update(closeAt=self.row['close_at']+1)),
            ('before close capture', lambda p: p.update(captured_at='2026-10-19T23:24:59+09:00')),
            ('timezone missing', lambda p: p.update(captured_at='2026-10-19T23:37:26')),
            ('decision after capture', lambda p: p['data']['race'].update(decidedAt=NEXT_DAY.timestamp())),
            ('payoff not final', lambda p: p['data'].update(trifectaPayoffStatus=0)),
            ('missing explicit winner', lambda p: p['data'].update(trifectaWinningOddsIds=[])),
            ('duplicate explicit winner', lambda p: p['data']['trifectaWinningOddsIds'].append(p['data']['trifectaWinningOddsIds'][0])),
            ('zero payoff', lambda p: winner(p).update(payoffUnitPrice=0)),
            ('fractional yen', lambda p: winner(p).update(payoffUnitPrice=2270.5)),
            ('wrong payoff unit', lambda p: winner(p).update(unitPrice=1000)),
            ('missing entrant', lambda p: p['data']['entries'].pop()),
            ('missing ticket evidence', lambda p: p['data']['trifecta'].pop()),
            ('withdrawal unconfirmed', lambda p: p['data']['entries'][0].pop('absent')),
            ('refund contradicts entrant', lambda p: winner(p).update(absent=True)),
            ('cancellation', lambda p: p['data']['race'].update(cancel=True)),
            ('cancellation unknown', lambda p: p['meta'].pop('cancel')),
            ('unlisted positive payoff', lambda p: p['data']['trifecta'][0].update(payoffUnitPrice=100)),
        ]
        for name, change in changes:
            with self.subTest(name=name):
                page = copy.deepcopy(self.raw)
                change(page)
                self.write_raw(page)
                self.assertIsNone(read_saved_page_outcome(self.root, self.row))
        # The final invalid page cannot promote the fifth pending comparison.
        payload = near.run(self.root, clock=lambda: NEXT_DAY)
        self.assertEqual(payload['comparison']['paired_races'], 4)
        row = next(r for r in payload['races'] if r['id'] == RID)
        self.assertEqual(row['budget_settlement']['status'], 'pending')


if __name__ == '__main__':
    unittest.main()
