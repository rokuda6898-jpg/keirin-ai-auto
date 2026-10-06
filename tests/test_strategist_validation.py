import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
from betting_logic import STRATEGY_VERSION
from strategist_validation import consensus, build_strategist_validation, preclose


class StrategistTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.folder = self.root / 'company'
        self.folder.mkdir()
        self.now = datetime(2026, 10, 7, 10, tzinfo=ZoneInfo('Asia/Tokyo'))

    def proposals(self):
        base = {'race_id': 'r1', 'venue': 'test', 'race_no': 1,
                'snapshot_at': self.now.isoformat(), 'close_at': (self.now+timedelta(minutes=10)).timestamp(),
                'winner_car': 1, 'tickets': [{'buy': '1-2-3', 'group': '本線', 'prob': .1, 'ev': 1.5}]}
        return [dict(base, department=name) for name in ['data_department', 'line_department']]

    def write(self, name, value):
        (self.folder / name).write_text(json.dumps(value), encoding='utf-8')

    def test_consensus_is_conservative_and_deduplicated(self):
        views = self.proposals()
        views[1]['tickets'] = [{'buy': '1-2-3', 'group': '本線', 'prob': .08, 'ev': 1.2}]*2
        tickets = consensus(views)
        self.assertEqual(len(tickets), 1)
        self.assertEqual(tickets[0]['ev'], 1.2)
        self.assertEqual(tickets[0]['votes'], 2)
        views[1]['tickets'][0]['ev'] = .9
        self.assertEqual(consensus(views), [])

    def test_conflicting_prices_and_single_votes_are_excluded(self):
        views = self.proposals()
        self.assertEqual(consensus(views[:1]), [])
        views[1]['tickets'] = [{'buy': '1-2-3', 'group': '本線', 'prob': .1, 'ev': 3}]
        self.assertEqual(consensus(views), [])

    def test_preclose_rejects_future_and_naive_timestamps(self):
        row = self.proposals()[0]
        self.assertTrue(preclose(row, self.now))
        row['snapshot_at'] = (self.now+timedelta(minutes=1)).isoformat()
        self.assertFalse(preclose(row, self.now))
        row['snapshot_at'] = '2026-10-07T10:00:00'
        self.assertFalse(preclose(row))

    def test_frozen_comparison_settles_and_retains_results(self):
        self.write('annual_department_predictions.json', {'proposals': self.proposals()})
        first = build_strategist_validation(self.root, self.now)
        self.assertEqual(len(first['latest_plans']), 1)
        self.assertEqual(first['paired_races'], 0)
        build_strategist_validation(self.root, self.now)
        ledger = self.folder / 'annual_strategist_ledger.jsonl'
        self.assertEqual(len(ledger.read_text(encoding="utf-8").splitlines()), 1)
        after = self.now+timedelta(minutes=11)
        result = {'race_id': 'r1', 'official_result_available': True, 'actual_trifecta': '1-2-3', 'actual_trifecta_odds': 15}
        (self.root / 'latest_results.json').write_text(json.dumps([result]))
        baseline = dict(self.proposals()[0], strategy_version=STRATEGY_VERSION,
                        actual_trifecta='1-2-3', payout_per_100yen=1500)
        self.write('ticket_return_settled.json', [baseline])
        settled = build_strategist_validation(self.root, after)
        self.assertEqual(settled['paired_races'], 1)
        self.assertEqual(settled['paired_strategist']['return_rate'], 15)
        self.assertFalse(settled['auto_promotion'])
        (self.root / 'latest_results.json').write_text('[]')
        self.assertEqual(build_strategist_validation(self.root, after)['paired_races'], 1)
        self.assertEqual(len(ledger.read_text(encoding="utf-8").splitlines()), 1)

    def test_mixed_batches_and_postclose_do_not_freeze(self):
        views = self.proposals()
        views[1]['snapshot_at'] = (self.now-timedelta(minutes=1)).isoformat()
        self.write('annual_department_predictions.json', {'proposals': views})
        self.assertEqual(build_strategist_validation(self.root,self.now)['latest_plans'], [])
        self.write('annual_department_predictions.json', {'proposals': self.proposals()})
        self.assertEqual(build_strategist_validation(self.root,self.now+timedelta(hours=1))['latest_plans'], [])

    def test_missing_payout_is_pending(self):
        self.write('annual_department_predictions.json', {'proposals': self.proposals()})
        build_strategist_validation(self.root,self.now)
        (self.root/'latest_results.json').write_text(json.dumps([{'race_id':'r1','official_result_available':True,'actual_trifecta':'1-2-3'}]))
        self.assertEqual(build_strategist_validation(self.root,self.now+timedelta(hours=1))['settled']['settled_races'],0)

    def test_creation_time_does_not_backdate_department_timestamp(self):
        self.write('annual_department_predictions.json', {'proposals': self.proposals()})
        later = self.now + timedelta(minutes=2)
        row = build_strategist_validation(self.root,later)['latest_plans'][0]
        self.assertEqual(row['snapshot_at'],later.isoformat(timespec='seconds'))
        self.assertEqual(row['source_snapshot_at'],self.now.isoformat())
