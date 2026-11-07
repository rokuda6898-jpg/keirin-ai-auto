"""Focused deadline, evidence, immutability, budget and settlement checks."""
import copy
import hashlib
import itertools
import json
from datetime import datetime, timedelta
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import high_roi_cloud as cloud
from nexus_model import replay_evidence


class CloudTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime.fromisoformat('2026-11-07T12:00:00+09:00')
        self.close = self.now + timedelta(hours=1)
        self.rid = '012120261107'
        self.manifest = {'id': 'test-trained-base', 'train': ['2024-01-01', '2025-01-01']}
        self.manifest_bytes = json.dumps(self.manifest).encode()
        self.manifest_hash = hashlib.sha256(self.manifest_bytes).hexdigest()
        cars = list(range(1, 8))
        triples = list(itertools.permutations(cars, 3))
        raw = [1 / (1 + i) for i in range(len(triples))]
        q = [v / sum(raw) for v in raw]
        meta = dict(race_id=self.rid, date='2026-11-07', venue='テスト', race_number=1,
                    close_at=int(self.close.timestamp()), start_at=int(self.close.timestamp()) + 300,
                    race_class='A級', race_type='予選', graded=False)
        data = dict(race={'id': self.rid, 'distance': 1600, 'startAt': meta['start_at'],
                          'closeAt': meta['close_at'], 'decidedAt': 0},
                    schedule={'date': '20261107', 'cupId': '2026110721', 'index': 1, 'day': 1},
                    players=[], entries=[{'number': n, 'absent': False} for n in cars],
                    records=[], linePrediction=[], results=[], finalOdds=False,
                    trifectaWinningOddsIds=[], trifecta=[{
                        'key': list(t), 'raceId': self.rid, 'unitPrice': 100,
                        'payoffUnitPrice': 0, 'odds': 10 + (i % 300), 'absent': False,
                    } for i, t in enumerate(triples)])
        self.snap = dict(meta=meta, data=data, captured_at=(self.now - timedelta(minutes=1)).isoformat(),
                         source_kind='existing_public_page_response', sha256='a' * 64)
        self.snap['input_sha256'] = cloud.input_digest(self.snap)
        original = {'tickets': [triples[0]], 'blocks': [dict(first=[1], second=[2], third=[3], count=1)]}
        pred = dict(model_id=self.manifest['id'], captured_at=(self.now - timedelta(hours=1)).isoformat(),
                    input_captured_at=(self.now - timedelta(hours=1, minutes=1)).isoformat(),
                    input_sha256=self.snap['input_sha256'], source_sha256='b' * 64,
                    history_through='2026-10-18', band_probabilities=dict(MAIN=.7, MEDIUM=.2, HIGH=.1),
                    replay_evidence=replay_evidence.from_inference(cars, q, original, self.manifest, self.manifest_hash))
        replay_evidence.bind_prediction(pred, race_id=self.rid, race_date='2026-11-07', close_at=self.close.isoformat())
        self.row = dict(race_id=self.rid, date='2026-11-07', close_at=self.close.isoformat(),
                        riders=[{'car': c, 'name': str(c)} for c in cars], lines=[], prediction=pred)

    def build(self, row=None, snap=None, clock=None):
        return cloud.build_forecast(row or self.row, snap or self.snap, self.manifest,
                                    self.manifest_hash, clock=clock or (lambda: self.now))

    def test_exact_budgets_full_formations_and_odds(self):
        for category, graded, expected in [('MAIN', False, 8), ('MEDIUM', False, 8),
                                           ('HIGH', False, 10), ('MAIN', True, 18)]:
            with self.subTest(category=category, graded=graded):
                row, snap = copy.deepcopy(self.row), copy.deepcopy(self.snap)
                row['prediction']['band_probabilities'] = {b: .8 if b == category else .1 for b in cloud.BANDS}
                # Graded status is part of the hashed model input.
                snap['meta']['graded'] = graded
                snap['input_sha256'] = cloud.input_digest(snap)
                row['prediction']['input_sha256'] = snap['input_sha256']
                evidence = row['prediction']['replay_evidence']
                evidence['input_sha256'] = snap['input_sha256']
                evidence['evidence_sha256'] = cloud.digest({k: v for k, v in evidence.items() if k != 'evidence_sha256'})
                result = self.build(row, snap)
                f = result['forecast']
                self.assertEqual((f['count'], f['total_yen']), (expected, expected * 100))
                self.assertEqual(len(set(f['tickets'])), expected)
                self.assertLessEqual(len(f['blocks']), 4)
                self.assertEqual(set(f['odds']), set(f['tickets']))
                self.assertEqual(f['confidence']['status'], 'assessed')
                self.assertEqual(f['history_through'], '2026-10-18')
                self.assertEqual({mark['car'] for mark in f['marks']}, set(range(1, 8)))
                self.assertEqual(result['venue_id'], '21')
                self.assertFalse(result['performance_eligible'])
                self.assertNotIn('first_published_at', result)

    def test_leakage_hash_and_deadline_rejections_precede_selection(self):
        cases = []
        for key, value in [('finalOdds', True), ('results', [{'position': 1}]), ('trifectaWinningOddsIds', ['winning'])]:
            snap = copy.deepcopy(self.snap); snap['data'][key] = value; cases.append((self.row, snap))
        snap = copy.deepcopy(self.snap); snap['data']['records'] = [{'new': 1}]; cases.append((self.row, snap))
        row = copy.deepcopy(self.row); row['prediction']['replay_evidence']['q'][0] += .01; cases.append((row, self.snap))
        snap = copy.deepcopy(self.snap); snap['captured_at'] = (self.now - timedelta(hours=1)).isoformat(); cases.append((self.row, snap))
        with patch.object(cloud.optimizer, 'select', side_effect=AssertionError('selection must not be reached')):
            for row, snap in cases:
                with self.assertRaises(ValueError):
                    self.build(row, snap)
            with self.assertRaises(ValueError):
                self.build(clock=lambda: self.close - timedelta(minutes=5))

    def test_missing_or_sentinel_quotes_are_explicit_probability_fallback(self):
        snap = copy.deepcopy(self.snap)
        for quote in snap['data']['trifecta']:
            quote['odds'] = 9999.9
        result = self.build(snap=snap)['forecast']
        self.assertEqual(result['source_mode'], 'probability_only')
        self.assertNotIn('odds', result)
        self.assertNotIn('odds_at', result)
        self.assertEqual(result['selected_odds_count'], 0)

    def test_optimizer_crossing_deadline_cannot_create_a_forecast(self):
        times = iter([self.now, self.close - timedelta(minutes=4)])
        with self.assertRaisesRegex(ValueError, 'deadline passed during optimization'):
            self.build(clock=lambda: next(times))

    def test_missing_snapshot_preserves_schedule_without_forecast_or_credit(self):
        row = copy.deepcopy(self.row)
        row.update(race_no=1, class_name='A級', stage='予選', session='day')
        result = cloud.schedule_row(row, None, self.now, 'pre-close snapshot missing')
        self.assertEqual(result['race_number'], 1)
        self.assertEqual(result['race_class'], 'A級')
        self.assertEqual(result['race_type'], '予選')
        self.assertEqual(result['session'], 'day')
        self.assertIsNone(result['forecast'])
        self.assertFalse(result['performance_eligible'])

    def test_persistence_receipts_and_official_settlement_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'models/nexus53').mkdir(parents=True)
            (root / 'models/nexus53/manifest.json').write_bytes(self.manifest_bytes)
            cloud.save_json(root / 'outputs/nexus_live.json', {'races': [self.row]})
            cloud.save_json(root / 'data/raw/nexus_page_snapshots/2026-11-07/preclose' / (self.rid + '.json'), self.snap)
            first = cloud.run(root, clock=lambda: self.now, confirm_existing=True)
            original = copy.deepcopy(first['races'][0]['forecast'])
            self.assertFalse(first['races'][0]['performance_eligible'])
            # Changed/new source quotes may never rewrite the first forecast.
            self.snap['data']['trifecta'] = []
            cloud.save_json(root / 'data/raw/nexus_page_snapshots/2026-11-07/preclose' / (self.rid + '.json'), self.snap)
            confirmed = cloud.run(root, clock=lambda: self.now + timedelta(seconds=5), confirm_existing=True)
            self.assertEqual(confirmed['races'][0]['forecast'], original)
            self.assertTrue(confirmed['races'][0]['performance_eligible'])
            self.assertEqual(confirmed['observed']['pending'], 1)
            won = original['tickets'][0]
            # Pure official outcome helpers are exercised through the publisher.
            record = {'void': False, 'winners': [won], 'payoffs': {won: 12340},
                      'observed_at': (self.close + timedelta(minutes=10)).isoformat()}
            with patch.object(cloud, '_load_official', return_value={self.rid: record}):
                settled = cloud.run(root, clock=lambda: self.close + timedelta(minutes=11))
            self.assertEqual(settled['races'][0]['forecast'], original)
            self.assertEqual(settled['observed']['races'], 1)
            self.assertEqual(settled['observed']['returned'], 12340)
            self.assertEqual(settled['observed']['model_id'], cloud.MODEL_ID)
            # A late first receipt cannot retrospectively enter performance.
            frozen = cloud.read_json(root / cloud.OUTPUT)
            frozen['races'][0].pop('first_published_at')
            frozen['races'][0]['performance_eligible'] = False
            cloud.save_json(root / cloud.OUTPUT, frozen)
            late = cloud.run(root, clock=lambda: self.close + timedelta(minutes=12), confirm_existing=True)
            self.assertFalse(late['races'][0]['performance_eligible'])
            self.assertEqual(late['observed']['races'], 0)
            with self.assertRaises(ValueError):
                cloud.run(root, clock=lambda: self.now, race_date='2026-11-06')


if __name__ == '__main__':
    unittest.main()
