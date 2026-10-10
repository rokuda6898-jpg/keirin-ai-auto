import json
import tempfile
import unittest
from datetime import datetime, timedelta
from itertools import permutations
from pathlib import Path

import pandas as pd

from conditional_live import LEDGER, atomic_json, run


NOW = datetime.fromisoformat('2026-10-10T09:00:00+09:00')


def fixture(root):
    (root/'outputs/company').mkdir(parents=True)
    (root/'data/raw').mkdir(parents=True)
    schedule, entries = [], []
    for rid in ('open', 'late', 'missing'):
        close = (NOW+timedelta(hours=-1 if rid == 'late' else 1)).timestamp()
        schedule.append(dict(race_id=rid, date='2026-10-10', venue='テスト', race_no=len(schedule)+1,
                             close_at=close, start_at=close+300))
        if rid != 'missing':
            entries.extend(dict(race_id=rid, date='2026-10-10', car_no=i, player_id=str(i),
                                entries_number=7, close_at=close, player_name='選手'+str(i)) for i in range(1, 8))
    pd.DataFrame(schedule).to_csv(root/'outputs/latest_race_schedule.csv', index=False)
    pd.DataFrame(entries).to_csv(root/'data/raw/today_entries.csv', index=False)
    return schedule


def fake_model(root):
    return {'chain_weight': 1, 'model_sha256': 'fixture'}, {'cutoff': '2025-10-09'}


def fake_predict(entries, bundle):
    for rid, race in entries.groupby('race_id'):
        keys = ['-'.join(map(str, p)) for p in permutations(range(1, 8), 3)]
        yield dict(race_id=rid, keys=keys, independent=[1/210]*210, chain=[1/210]*210)


def execute(root, **kwargs):
    return run(root, clock=kwargs.pop('clock', lambda: NOW),
               model_loader=fake_model, predictor=kwargs.pop('predictor', fake_predict), **kwargs)


class ConditionalLiveTests(unittest.TestCase):
    def test_all_races_remain_but_closed_and_missing_are_never_backfilled(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root)
            feed, missing = execute(root)
            self.assertEqual(len(feed['races']), 3)
            self.assertEqual(missing, ['missing'])
            by_id = {r['id']: r for r in feed['races']}
            self.assertEqual(len(by_id['open']['shadow']['tickets']), 12)
            self.assertEqual(len(by_id['open']['shadow']['marks']), 5)
            self.assertIsNone(by_id['late']['shadow'])
            self.assertIsNone(by_id['missing']['shadow'])
            self.assertFalse((root/'outputs/company/company_decision_ledger.jsonl').exists())

    def test_repeated_run_preserves_original_snapshot_and_tickets(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root)
            execute(root)
            original = (root/LEDGER).read_bytes()
            def must_not_repredict(*args):
                raise AssertionError('a frozen race was regenerated')
            execute(root, predictor=must_not_repredict)
            self.assertEqual(original, (root/LEDGER).read_bytes())

    def test_clock_checked_after_calculation_not_only_at_start(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root)
            clock = iter([NOW, NOW+timedelta(hours=2), NOW+timedelta(hours=2)])
            feed, _ = execute(root, clock=lambda: next(clock))
            self.assertTrue(all(r['shadow'] is None for r in feed['races']))

    def test_official_ties_and_unknown_payout_remain_separate_from_forecasts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root); execute(root)
            before = json.loads((root/LEDGER).read_text(encoding='utf-8'))
            atomic_json(root/'outputs/latest_results.json', [dict(race_id='open',
                official_result_available=True, actual_trifecta='1-2-3|2-1-3',
                payouts_trifecta_json={'1-2-3': 5100})])
            feed, _ = execute(root, predict_new=False)
            r = next(r for r in feed['races'] if r['id'] == 'open')
            self.assertEqual(r['actual'], ['1-2-3', '2-1-3'])
            self.assertEqual(r['payouts'], {'1-2-3': 5100})
            self.assertEqual(r['shadow']['tickets'], before['top12'])
            self.assertEqual(r['shadow']['snapshot_at'], before['snapshot_at_jst'])
            atomic_json(root/'outputs/latest_results.json', [dict(race_id='open', official_result_available=False)])
            corrected, _ = execute(root, predict_new=False)
            self.assertEqual(next(r for r in corrected['races'] if r['id'] == 'open')['actual'], [])

    def test_outcome_already_published_and_incomplete_fields_not_predicted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root)
            atomic_json(root/'outputs/latest_results.json', [dict(race_id='open',
                official_result_available=True, actual_trifecta='1-2-3')])
            feed, _ = execute(root)
            self.assertTrue(all(r['shadow'] is None for r in feed['races']))
            atomic_json(root/'outputs/latest_results.json', [])
            entries = pd.read_csv(root/'data/raw/today_entries.csv')
            entries = entries[entries.car_no.ne(7)]
            entries.to_csv(root/'data/raw/today_entries.csv', index=False)
            feed, _ = execute(root)
            self.assertTrue(all(r['shadow'] is None for r in feed['races']))
            self.assertIn('incomplete_field', feed['coverage']['errors']['open'])

    def test_source_close_time_and_cancelled_cars_are_respected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); fixture(root)
            entries = pd.read_csv(root/'data/raw/today_entries.csv')
            entries['cancelled_car_numbers'] = '[7]'
            entries.to_csv(root/'data/raw/today_entries.csv', index=False)
            feed, _ = execute(root)
            self.assertEqual(next(r for r in feed['races'] if r['id'] == 'open')['cancelled_cars'], ['7'])
            entries.loc[entries.race_id.eq('missing'), 'close_at'] = NOW.timestamp()-60
            # An inconsistent within-race close must never be accepted.
            entries.loc[0, 'close_at'] = NOW.timestamp()-60
            (root/LEDGER).unlink()
            entries.to_csv(root/'data/raw/today_entries.csv', index=False)
            feed, _ = execute(root)
            self.assertTrue(all(r['shadow'] is None for r in feed['races']))


if __name__ == '__main__':
    unittest.main()
