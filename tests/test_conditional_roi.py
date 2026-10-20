import copy
import csv
from datetime import datetime, timedelta
from itertools import permutations
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import conditional_roi as roi
from conditional_budget import allocate_range, check_saved_plan


class ConditionalRoiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = datetime(2026, 10, 20, 8, 0, tzinfo=roi.JST)
        self.close = self.now + timedelta(hours=1)
        self.rid = '016220261020'
        self.url = 'https://www.winticket.jp/keirin/hiroshima/racecard/2026102062/1/1'
        self.tickets = ['1-2-3', '1-3-2', '2-1-3', '2-3-1', '3-1-2', '3-2-1',
                        '1-2-4', '1-4-2', '2-1-4', '2-4-1', '4-1-2', '4-2-1']
        self.q = [.1, .08, .07, .06, .05, .04, .035, .03, .025, .02, .015, .01]
        self.original = dict(race_id=self.rid, date='2026-10-20', venue='広島', race_no='1',
                             close_at=self.close.timestamp(), start_at=(self.close + timedelta(minutes=5)).timestamp(),
                             source_url=self.url, snapshot_at_jst=(self.now - timedelta(minutes=30)).isoformat(),
                             top12=self.tickets, probabilities=self.q, model_sha256='b' * 64,
                             training_cutoff_exclusive='2026-10-06',
                             riders=[dict(car=str(i), name=f'選手{i}') for i in range(1, 8)],
                             actual=[], payouts={}, cancelled_cars=[])
        self.original['budget'] = self.plan(20)
        self.write_inputs()

    def plan(self, price, stamp=None):
        stamp = stamp or self.now - timedelta(minutes=30)
        plan = allocate_range(self.tickets, dict.fromkeys(self.tickets, price))
        plan.update(odds_captured_at_jst=(stamp - timedelta(seconds=1)).isoformat(),
                    allocation_frozen_at_jst=stamp.isoformat(), source_url=self.url, provider='winticket')
        return plan

    def source(self):
        r = self.original
        return dict(id=r['race_id'], date=r['date'], venue=r['venue'], number=int(float(r['race_no'])),
                    start_at=r['start_at'], close_at=r['close_at'], riders=r['riders'],
                    shadow=dict(tickets=r['top12'], snapshot_at=r['snapshot_at_jst']),
                    budget=r['budget'], actual=r.get('actual', []), payouts=r.get('payouts', {}),
                    cancelled_cars=r.get('cancelled_cars', []))

    def write_inputs(self, extra_schedule=()):
        path = self.root / roi.LEDGER
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.original) + '\n')
        roi.save_json(self.root / 'outputs/conditional_live.json', dict(schema=1, races=[self.source()]))
        fields = ['race_id', 'date', 'venue', 'race_no', 'close_at', 'start_at']
        with (self.root / 'outputs/latest_race_schedule.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerow({key: self.original[key] for key in fields})
            for row in extra_schedule:
                writer.writerow(row)

    def at(self, when=None):
        return roi.run(self.root, clock=lambda: when or self.now)

    def row(self, payload):
        return next(r for r in payload['races'] if r['id'] == self.rid)

    def result(self, prices=None, complete=True):
        prices = prices or {'1-2-3': 1700}
        value = dict(race_id=self.rid, date='2026-10-20', venue='広島', race_no=1,
                     official_result_available=True, actual_trifecta_buys=list(prices),
                     payouts_trifecta_json=prices if complete else {},
                     result_observed_at_jst=(self.close + timedelta(minutes=2)).isoformat())
        roi.save_json(self.root / 'outputs/latest_results.json', [value])
        return value

    def test_actual_preclose_eight_and_same_budget_are_frozen_without_network_or_source_writes(self):
        paths = [self.root / roi.LEDGER, self.root / 'outputs/conditional_live.json', self.root / 'outputs/latest_race_schedule.csv']
        before = {p: p.read_bytes() for p in paths}
        with patch('socket.socket', side_effect=AssertionError('network forbidden')), \
                patch('urllib.request.urlopen', side_effect=AssertionError('network forbidden')):
            payload = self.at()
        row = self.row(payload)
        selected = [self.tickets[i - 1] for i in roi.SELECTED_RANKS]
        self.assertEqual(row['shadow']['tickets'], selected)
        self.assertEqual(row['shadow']['snapshot_at'], self.now.isoformat())
        self.assertEqual(row['original_snapshot_at_jst'], self.original['snapshot_at_jst'])
        self.assertEqual(row['original_tickets'], self.tickets)
        self.assertEqual(row['probabilities'], [self.q[i - 1] for i in roi.SELECTED_RANKS])
        self.assertEqual(row['shadow']['confidence']['grade'], 'B')
        self.assertAlmostEqual(row['shadow']['confidence']['model_mass'], .38)
        self.assertTrue(check_saved_plan(row['budget'], selected))
        self.assertEqual(row['budget']['budget_yen'], self.original['budget']['budget_yen'])
        self.assertEqual(row['initial_budget'], self.original['budget'])
        self.assertEqual(payload['comparison']['flat_100']['paired_races'], 0)
        self.assertIsNone(payload['comparison']['allocation']['roi_eight']['roi'])
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        frozen = {k: copy.deepcopy(row[k]) for k in ('shadow', 'probabilities', 'original_tickets', 'budget', 'initial_budget', 'snapshot_at_jst')}
        self.original['budget'] = self.plan(50, self.now + timedelta(minutes=1))
        self.original['top12'] = list(reversed(self.tickets))
        self.original['probabilities'] = list(reversed(self.q))
        self.write_inputs()
        later = self.row(self.at(self.now + timedelta(minutes=2)))
        for key, value in frozen.items():
            self.assertEqual(later[key], value)

    def test_every_scheduled_race_remains_including_missing_bad_time_and_pre_policy(self):
        extra = [dict(race_id='026220261020', date='2026-10-20', venue='広島', race_no=2,
                      close_at=self.close.timestamp() + 1200, start_at=self.close.timestamp() + 1500),
                 dict(race_id='036220261020', date='2026-10-20', venue='広島', race_no=3,
                      close_at='', start_at=''),
                 dict(race_id='046220261019', date='2026-10-19', venue='広島', race_no=4,
                      close_at=self.close.timestamp() - 86400, start_at=self.close.timestamp() - 86100)]
        self.write_inputs(extra)
        rows = {r['id']: r for r in self.at()['races']}
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows['026220261020']['forecast_status'], 'missing')
        self.assertEqual(rows['036220261020']['forecast_status'], 'invalid')
        self.assertEqual(rows['046220261019']['forecast_status'], 'pre_policy')
        self.assertTrue(all(rows[r['race_id']]['shadow'] is None for r in extra))

    def test_no_closed_or_last_minute_backfill(self):
        for stamp in (self.close - timedelta(seconds=59), self.close, self.close + timedelta(days=1)):
            with self.subTest(stamp=stamp):
                (self.root / roi.FEED).unlink(missing_ok=True)
                row = self.row(self.at(stamp))
                self.assertIsNone(row['shadow'])
                self.assertIsNone(row['budget'])
                self.assertEqual(row['flat_settlement']['status'], 'not_predicted')

    def test_invalid_identity_probabilities_and_future_original_cannot_create_eight(self):
        original = copy.deepcopy(self.original)
        changes = [lambda r: r.update(model_sha256='unknown'),
                   lambda r: r.update(snapshot_at_jst=(self.now + timedelta(minutes=1)).isoformat()),
                   lambda r: r.update(snapshot_at_jst=self.close.isoformat()),
                   lambda r: r.update(probabilities=[.2] * 12),
                   lambda r: r.update(probabilities=list(reversed(self.q))),
                   lambda r: r.update(top12=self.tickets[:11] + [self.tickets[0]]),
                   lambda r: r.update(training_cutoff_exclusive='2026-10-21')]
        for change in changes:
            with self.subTest(change=changes.index(change)):
                (self.root / roi.FEED).unlink(missing_ok=True)
                self.original = copy.deepcopy(original)
                change(self.original)
                self.write_inputs()
                row = self.row(self.at())
                self.assertIsNone(row['shadow'])
                self.assertEqual(row['forecast_status'], 'invalid')
        self.original = original
        self.write_inputs()
        feed = roi.read_json(self.root / 'outputs/conditional_live.json')
        feed['races'][0]['venue'] = '別会場'
        roi.save_json(self.root / 'outputs/conditional_live.json', feed)
        (self.root / roi.FEED).unlink(missing_ok=True)
        self.assertEqual(self.row(self.at())['forecast_status'], 'invalid')

    def test_missing_quote_does_not_block_eight_and_first_later_allocation_is_immutable(self):
        self.original['budget'] = dict(status='unavailable')
        self.write_inputs()
        first = self.row(self.at())
        self.assertEqual(first['forecast_status'], 'saved')
        self.assertIsNone(first['budget'])
        self.original['budget'] = self.plan(20, self.now + timedelta(minutes=1))
        self.write_inputs()
        allocated = self.row(self.at(self.now + timedelta(minutes=2)))
        self.assertEqual(allocated['snapshot_at_jst'], first['snapshot_at_jst'])
        self.assertEqual(allocated['budget']['status'], 'allocated')
        saved = copy.deepcopy(allocated['budget'])
        self.original['budget'] = self.plan(200, self.now + timedelta(minutes=3))
        self.write_inputs()
        self.assertEqual(self.row(self.at(self.now + timedelta(minutes=4)))['budget'], saved)

    def test_quote_timing_identity_checksum_unit_and_amount_rejections_keep_flat_prediction(self):
        good = copy.deepcopy(self.original['budget'])
        mutations = [lambda b: b.update(source_url=self.url[:-1] + '2'),
                     lambda b: b.update(provider='other'), lambda b: b.update(ticket_sha256='a' * 64),
                     lambda b: b.update(unit_yen=1000),
                     lambda b: b['allocations'][0].update(stake_yen=1),
                     lambda b: b.update(budget_yen=5100),
                     lambda b: b.update(odds_captured_at_jst=self.close.isoformat()),
                     lambda b: b.update(allocation_frozen_at_jst=self.close.isoformat()),
                     lambda b: b.update(odds_captured_at_jst=(self.now - timedelta(days=1)).isoformat()),
                     lambda b: b['snapshot_odds'].update({self.tickets[0]: '9999.9'})]
        for mutate in mutations:
            with self.subTest(mutation=mutations.index(mutate)):
                (self.root / roi.FEED).unlink(missing_ok=True)
                self.original['budget'] = copy.deepcopy(good)
                mutate(self.original['budget'])
                self.write_inputs()
                row = self.row(self.at())
                self.assertEqual(row['forecast_status'], 'saved')
                self.assertIsNone(row['budget'])
                self.assertTrue(row['budget_error'])

    def test_infeasible_is_frozen_and_original_infeasible_can_yield_smallest_feasible_eight(self):
        self.original['budget'] = self.plan(1.1)
        self.write_inputs()
        first = self.row(self.at())
        self.assertEqual(first['budget']['status'], 'infeasible')
        self.original['budget'] = self.plan(50, self.now + timedelta(minutes=1))
        self.write_inputs()
        self.assertEqual(self.row(self.at(self.now + timedelta(minutes=2)))['budget'], first['budget'])
        (self.root / roi.FEED).unlink()
        self.original['budget'] = self.plan(10)
        self.write_inputs()
        row = self.row(self.at())
        self.assertEqual(row['initial_budget']['status'], 'infeasible')
        self.assertEqual(row['budget']['status'], 'allocated')
        self.assertEqual(row['budget']['budget_yen'], 5000)
        self.assertEqual(row['budget']['budget_selection'], 'smallest_feasible_no_loss')

    def test_official_payouts_not_quote_prices_settle_ties_and_survive_rollover(self):
        first = self.at()
        self.result({'1-2-3': 1700, '2-1-4': 2200}, complete=False)
        pending = self.at(self.close + timedelta(minutes=2))
        self.assertEqual(pending['comparison']['allocation']['paired_races'], 0)
        self.result({'1-2-3': 1700, '2-1-4': 2200})
        settled = self.at(self.close + timedelta(minutes=3))
        row = self.row(settled)
        self.assertEqual(row['flat_settlement']['payout_yen'], 3900)
        self.assertEqual(row['initial_flat_settlement']['payout_yen'], 3900)
        stakes = {a['buy']: a['stake_yen'] for a in row['budget']['allocations']}
        self.assertEqual(row['budget_settlement']['payout_yen'], 1700 * (stakes['1-2-3'] // 100) + 2200 * (stakes['2-1-4'] // 100))
        self.assertEqual(settled['comparison']['allocation']['paired_races'], 1)
        self.assertEqual(settled['comparison']['flat_100']['paired_races'], 1)
        self.assertEqual(row['budget'], self.row(first)['budget'])
        (self.root / roi.LEDGER).write_text('')
        roi.save_json(self.root / 'outputs/conditional_live.json', dict(schema=1, races=[]))
        roi.save_json(self.root / 'outputs/latest_results.json', [])
        later = self.row(self.at(self.close + timedelta(days=1)))
        self.assertEqual(later['flat_settlement'], row['flat_settlement'])
        self.assertEqual(later['budget_settlement'], row['budget_settlement'])

    def test_explicit_saved_refunds_override_contaminated_compact_result_and_persist(self):
        self.at()
        quotes = []
        for key in permutations(range(1, 8), 3):
            absent = 4 in key
            quotes.append(dict(id=self.rid + ':2:' + '.'.join(map(str, key)), key=list(key),
                               raceId=self.rid, absent=absent, unitPrice=100,
                               payoffUnitPrice=1700 if key == (1, 2, 3) else 100 if absent else 0))
        snapshot = dict(schema=1, source_kind='existing_public_page_response',
                        captured_at=(self.close + timedelta(minutes=2)).isoformat(), sha256='a' * 64, url=self.url,
                        meta=dict(race_id=self.rid, date='2026-10-20', venue_id='62', venue='広島', race_number=1,
                                  close_at=self.close.timestamp(), cancel=False),
                        data=dict(race=dict(id=self.rid, number=1, closeAt=self.close.timestamp(), entriesNumber=7, cancel=False, decidedAt=0),
                                  schedule=dict(date='20261020'),
                                  entries=[dict(number=i, raceId=self.rid, absent=i == 4) for i in range(1, 8)],
                                  trifecta=quotes, trifectaPayoffStatus=1,
                                  trifectaWinningOddsIds=[self.rid + ':2:1.2.3']))
        path = self.root / 'data/raw/nexus_page_snapshots/2026-10-20/outcomes' / (self.rid + '.json')
        roi.save_json(path, snapshot)
        contaminated = {'-'.join(map(str, q['key'])): q['payoffUnitPrice'] for q in quotes if q['payoffUnitPrice']}
        self.result(contaminated)
        row = self.row(self.at(self.close + timedelta(minutes=3)))
        self.assertEqual(row['actual'], ['1-2-3'])
        self.assertEqual(row['payouts'], {'1-2-3': 1700})
        self.assertEqual(row['cancelled_cars'], ['4'])
        for key in ('flat_settlement', 'initial_flat_settlement', 'budget_settlement', 'initial_budget_settlement'):
            self.assertEqual(row[key]['status'], 'refund_pending')
            self.assertIsNone(row[key]['hit'])
        path.unlink()
        roi.save_json(self.root / 'outputs/latest_results.json', [])
        later = self.row(self.at(self.close + timedelta(days=1)))
        self.assertEqual(later['actual'], ['1-2-3'])
        self.assertEqual(later['budget_settlement']['status'], 'refund_pending')

    def test_gate_discards_private_forecast_or_private_first_allocation_but_keeps_public_plan(self):
        private = self.at()
        dropped = roi.gate_publication(self.root, {}, clock=lambda: self.close - timedelta(seconds=59))
        self.assertEqual(dropped, [self.rid])
        discarded = self.row(roi.read_json(self.root / roi.FEED))
        self.assertIsNone(discarded['shadow'])
        self.assertIsNone(discarded['budget'])
        self.assertEqual(discarded['flat_settlement']['status'], 'not_predicted')
        (self.root / roi.FEED).unlink()
        self.original['budget'] = dict(status='unavailable')
        self.write_inputs()
        public = self.at()
        self.original['budget'] = self.plan(20, self.now + timedelta(minutes=1))
        self.write_inputs()
        self.at(self.now + timedelta(minutes=2))
        dropped = roi.gate_publication(self.root, public, clock=lambda: self.close - timedelta(seconds=59))
        row = self.row(roi.read_json(self.root / roi.FEED))
        self.assertEqual(dropped, [self.rid])
        self.assertEqual(row['shadow'], self.row(public)['shadow'])
        self.assertIsNone(row['budget'])
        roi.save_json(self.root / roi.FEED, private)
        self.assertEqual(roi.gate_publication(self.root, private, clock=lambda: self.close + timedelta(minutes=1)), [])
        self.assertEqual(self.row(roi.read_json(self.root / roi.FEED))['budget'], self.row(private)['budget'])

    def test_processing_that_crosses_deadline_cannot_publish_prediction_or_allocation(self):
        times = iter([self.now, self.now, self.now, self.close])
        payload = roi.run(self.root, clock=lambda: next(times, self.close))
        row = self.row(payload)
        self.assertIsNone(row['shadow'])
        self.assertIsNone(row['budget'])
        self.assertEqual(payload['comparison']['registered'], 0)

    def test_bad_persisted_amount_is_rejected_and_result_identity_cannot_cross_races(self):
        payload = self.at()
        result = self.result()
        result['date'] = '2026-10-21'
        roi.save_json(self.root / 'outputs/latest_results.json', [result])
        self.assertEqual(self.row(self.at(self.close + timedelta(minutes=3)))['budget_settlement']['status'], 'pending')
        self.row(payload)['budget']['allocations'][0]['stake_yen'] += 100
        roi.save_json(self.root / roi.FEED, payload)
        with self.assertRaisesRegex(ValueError, '配分'):
            self.at()

    def test_saved_forecast_must_retain_sixty_second_lead_and_allocation_timestamps_are_frozen(self):
        good = self.at()
        changed = copy.deepcopy(good)
        row = self.row(changed)
        row['snapshot_at_jst'] = (self.close - timedelta(seconds=59)).isoformat()
        row['shadow']['snapshot_at'] = row['snapshot_at_jst']
        row['forecast_fingerprint'] = roi._frozen_signature(row)
        roi.save_json(self.root / roi.FEED, changed)
        with self.assertRaisesRegex(ValueError, '8点予想'):
            self.at()
        changed = copy.deepcopy(good)
        self.row(changed)['budget']['allocation_frozen_at_jst'] = (self.now + timedelta(seconds=1)).isoformat()
        roi.save_json(self.root / roi.FEED, changed)
        with self.assertRaisesRegex(ValueError, '時刻・内容'):
            self.at()

    def test_existing_oddspark_quote_is_accepted_only_for_exact_race_date_and_bet_type(self):
        self.original['budget'].update(provider='oddspark', source_url=
            'https://www.oddspark.com/keirin/Odds.do?betType=9&joCode=62&kaisaiBi=20261020&raceNo=1&viewType=1')
        self.write_inputs()
        row = self.row(self.at())
        self.assertEqual(row['budget']['status'], 'allocated')
        (self.root / roi.FEED).unlink()
        self.original['budget']['source_url'] = self.original['budget']['source_url'].replace('20261020', '20261019')
        self.write_inputs()
        row = self.row(self.at())
        self.assertIsNone(row['budget'])
        self.assertEqual(row['forecast_status'], 'saved')

    def test_missing_preclose_future_or_inconsistent_result_times_never_settle(self):
        clean = self.at()
        observed = (self.close + timedelta(minutes=2)).isoformat()
        cases = [dict(result_observed_at_jst=''),
                 dict(result_observed_at_jst=(self.now - timedelta(minutes=1)).isoformat()),
                 dict(result_observed_at_jst=self.close.isoformat()),
                 dict(result_observed_at_jst=(self.close + timedelta(minutes=10)).isoformat()),
                 dict(result_observed_at_jst=observed, decided_at=self.close.timestamp()),
                 dict(result_observed_at_jst=observed, decided_at=(self.close + timedelta(minutes=3)).timestamp())]
        for case in cases:
            with self.subTest(case=case):
                roi.save_json(self.root / roi.FEED, clean)
                result = self.result()
                result.update(case)
                roi.save_json(self.root / 'outputs/latest_results.json', [result])
                payload = self.at(self.close + timedelta(minutes=4))
                row = self.row(payload)
                self.assertEqual(row['flat_settlement']['status'], 'pending')
                self.assertEqual(row['budget_settlement']['status'], 'pending')
                self.assertEqual(row['actual'], [])
                self.assertEqual(payload['comparison']['flat_100']['paired_races'], 0)
        self.result()
        settled = self.at(self.close + timedelta(minutes=4))
        self.assertEqual(self.row(settled)['budget_settlement']['status'], 'settled')
        result = self.result()
        result['result_observed_at_jst'] = (self.now - timedelta(minutes=1)).isoformat()
        roi.save_json(self.root / 'outputs/latest_results.json', [result])
        preserved = self.at(self.close + timedelta(minutes=5))
        self.assertEqual(self.row(preserved)['budget_settlement'], self.row(settled)['budget_settlement'])


if __name__ == '__main__':
    unittest.main()
