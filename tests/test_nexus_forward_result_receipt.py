"""Strict saved-page receipts reach forward settlement without rewriting plans.

The six completed examples and the pending 京王閣7R come from the saved outcome
paths at commit 22e71cd822f9dc57669739e04fc499b6d29298ac. Fixtures retain their
identity, receipt, explicit winner and payoff, with minimal complete quote rows.
The ten-ticket fixture plans exercise settlement transport, not historical ROI.
"""
import copy
from datetime import datetime, timedelta
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import nexus_forward as forward
import nexus_results_sync as sync
from nexus_input_snapshots import save_json

JST = ZoneInfo('Asia/Tokyo')
CURRENT = datetime.fromisoformat('2026-10-20T21:21:00+09:00')
# race ID, close epoch, decided epoch, entries, capture, source SHA, winner, yen.
CASES = (
    ('042720261020', 1792481340, 1792482390, 9, '2026-10-20T17:57:35.429330+09:00',
     '5ae3fe4c1a367fa51d380e4aa7ccf2f98d9fab06c949eb15f1dc9817cdd1c641', '4-3-5', 6660),
    ('043420261020', 1792482180, 1792483071, 7, '2026-10-20T17:58:38.963380+09:00',
     'bf703efc144d52c75828e609b68d186723ff7069164d5e9e499528f7d988be65', '4-3-7', 59710),
    ('052720261020', 1792483020, 1792484096, 9, '2026-10-20T17:57:37.117322+09:00',
     '731b6b132ca8575dbad85141e60aea67c9a47634ea92fc46163f6d94aa6500e6', '5-2-9', 12160),
    ('053420261020', 1792483860, 1792484685, 7, '2026-10-20T17:58:40.680838+09:00',
     'a2c8d0b05135ac70b6a1ae485ab95a7929d3956b777b6588fe3bc781119d94ee', '4-7-3', 550),
    ('062720261020', 1792484700, 1792486146, 9, '2026-10-20T17:57:38.843033+09:00',
     'e9a9eb066417429fdff5e3daff6f70ce783540132e877d9a71cc727f76249fbc', '4-2-6', 16080),
    ('063420261020', 1792485540, 1792486401, 7, '2026-10-20T17:58:42.567659+09:00',
     '7d52a5ced3c26275d59b90cc805d74cd420ac0e2e9a90ff3a7d100605977868d', '2-4-7', 5300),
    ('072720261020', 1792486440, 0, 9, '2026-10-20T17:57:40.545571+09:00',
     'f2bac982f07ed3d364d832bef68b59f442e29aa722c0449c47acecfcfa2c0689', None, None),
)


class ForwardResultReceiptTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def example(self, case=CASES[0], *, synchronize=True):
        rid, epoch, decided, count, captured, sha, winner, payoff = case
        close = datetime.fromtimestamp(epoch, JST)
        saved = close-timedelta(minutes=4)
        cars = list(range(1, count+1))
        a, b, _ = map(int, (winner or '1-2-3').split('-'))
        if count == 7:
            second = next(c for c in cars if c not in (a, b))
            blocks = [dict(first=[a], second=[b, second], third=cars)]
        else:
            blocks = [dict(first=[a], second=[b], third=cars),
                      dict(first=[b], second=[a], third=[c for c in cars if c not in (a, b)][:3])]
        tickets = set()
        for block in blocks:
            current = {'-'.join(map(str, t)) for t in itertools.product(
                block['first'], block['second'], block['third']) if len(set(t)) == 3}
            block['count'] = len(current)
            tickets.update(current)
        self.assertEqual(len(tickets), 10)
        plan = dict(count=10, tickets=sorted(tickets), blocks=blocks, allocations={}, auto_budget=None,
                    captured_at=saved.isoformat(), input_captured_at=(saved-timedelta(seconds=20)).isoformat())
        meta = dict(race_id=rid, date='2026-10-20', race_no=int(rid[:2]),
                    venue='京王閣' if rid[2:4] == '27' else '川崎', session='night',
                    close_at=close.isoformat(), start_at=(close+timedelta(minutes=5)).isoformat(),
                    riders=[dict(car=c, name=f'選手{c}', pid=str(100+c)) for c in cars],
                    source_url='https://example.test/race/'+rid)
        frozen = dict(meta, saved_at=saved.isoformat(), baseline=copy.deepcopy(plan),
            candidate=dict(copy.deepcopy(plan), policy_id=forward.POLICY_ID),
            source_prediction_sha256='a'*64, source_evidence_sha256='b'*64,
            allocation_status='missing_quotes', allocation_reason='fixture has no saved quotes',
            status='result_pending', result={}, settlement={})
        frozen['forecast_sha256'] = forward._frozen_signature(frozen)
        forward._validate_saved(frozen, saved-timedelta(minutes=1))
        source = dict(copy.deepcopy(meta), prediction=copy.deepcopy(plan), cancelled=False,
            winners=[], payouts={}, refunded=[], result_complete=False,
            result_captured_at=(close+timedelta(minutes=1)).isoformat(),
            result_source_sha256='f'*64)
        quote_rows = [dict(id=rid+':2:{}'.format('.'.join(map(str, t))), raceId=rid,
            key=list(t), unitPrice=100, absent=False,
            payoffUnitPrice=payoff if '-'.join(map(str, t)) == winner else 0)
            for t in itertools.permutations(cars, 3)]
        snapshot = dict(schema=1, source_kind='existing_public_page_response',
            captured_at=captured, sha256=sha, url=meta['source_url'],
            meta=dict(race_id=rid, date='2026-10-20', venue_id=rid[2:4],
                      race_number=int(rid[:2]), close_at=epoch, cancel=False),
            data=dict(race=dict(id=rid, number=int(rid[:2]), closeAt=epoch, decidedAt=decided,
                                entriesNumber=count, cancel=False),
                      schedule=dict(date='20261020'),
                      entries=[dict(number=c, raceId=rid, absent=False) for c in cars],
                      trifecta=quote_rows, trifectaPayoffStatus=int(winner is not None),
                      trifectaWinningOddsIds=[rid+':2:'+winner.replace('-', '.')] if winner else []))
        path = self.root/'data/raw/nexus_page_snapshots/2026-10-20/outcomes'/(rid+'.json')
        save_json(path, snapshot)
        if synchronize:
            self.assertEqual(sync._apply_saved_outcome(self.root/'outputs', source), winner is not None)
        return frozen, source, path

    def test_six_complete_saved_original_examples_settle_without_replacing_receipts_or_plans(self):
        rows = []
        for case in CASES[:6]:
            with self.subTest(race_id=case[0]):
                frozen, source, _ = self.example(case)
                source_before = copy.deepcopy(source)
                before = forward._frozen_signature(frozen)
                forward._update_result(frozen, source, CURRENT)
                self.assertEqual(frozen['status'], 'settled')
                self.assertEqual(frozen['result']['source_sha256'], case[5])
                self.assertEqual(frozen['result']['captured_at'], case[4])
                for side in forward.SIDES:
                    self.assertEqual(frozen['settlement'][side]['flat']['returned'], case[7])
                self.assertEqual(source, source_before)
                self.assertEqual(source['result_source_sha256'], 'f'*64)
                self.assertNotEqual(source['result_captured_at'], case[4])
                self.assertEqual(forward._frozen_signature(frozen), before)
                rows.append(frozen)
        report = dict(races=rows)
        forward._summarize(report)
        self.assertEqual(report['coverage']['settled'], 6)
        self.assertEqual(report['coverage']['allocated_common'], 0)

    def test_keio_7_incomplete_original_remains_pending(self):
        frozen, source, _ = self.example(CASES[6])
        self.assertIsNone(sync.read_saved_page_outcome(self.root, source))
        forward._update_result(frozen, source, CURRENT)
        self.assertEqual(frozen['status'], 'result_pending')
        self.assertEqual(frozen['settlement'], {})

    def test_partial_page_receipt_never_borrows_complete_legacy_fields(self):
        for missing in ('result_sync_captured_at', 'result_sync_source_sha256'):
            with self.subTest(missing=missing):
                frozen, source, _ = self.example()
                source.pop(missing)
                forward._update_result(frozen, source, CURRENT)
                self.assertEqual(frozen['status'], 'invalid_result')
                self.assertEqual(frozen['settlement'], {})

    def test_sync_fields_need_explicit_verified_page_source(self):
        frozen, source, _ = self.example()
        source['result_sync_source'] = 'latest_results.json'
        source.pop('result_source_sha256')
        forward._update_result(frozen, source, CURRENT)
        self.assertEqual(frozen['status'], 'invalid_result')

    def test_legacy_receipt_without_page_marker_keeps_existing_behavior(self):
        frozen, source, _ = self.example()
        source.pop('result_sync_source')
        forward._update_result(frozen, source, CURRENT)
        self.assertEqual(frozen['status'], 'settled')
        self.assertEqual(frozen['result']['source_sha256'], 'f'*64)
        self.assertEqual(frozen['result']['captured_at'], source['result_captured_at'])

    def test_page_receipt_retains_time_identity_and_complete_payout_guards(self):
        for invalid in ('future', 'before_close', 'bad_hash', 'race_number', 'payouts'):
            with self.subTest(invalid=invalid):
                frozen, source, _ = self.example()
                if invalid == 'future':
                    source['result_sync_captured_at'] = (CURRENT+timedelta(seconds=1)).isoformat()
                elif invalid == 'before_close':
                    source['result_sync_captured_at'] = (datetime.fromisoformat(source['close_at'])-timedelta(seconds=1)).isoformat()
                elif invalid == 'bad_hash':
                    source['result_sync_source_sha256'] = 'missing'
                elif invalid == 'race_number':
                    source['race_no'] += 1
                else:
                    source['payouts'] = {}
                forward._update_result(frozen, source, CURRENT)
                self.assertEqual(frozen['status'], 'invalid_result')
                self.assertEqual(frozen['settlement'], {})

    def test_result_ordering_uses_the_same_page_receipt_time(self):
        frozen, source, _ = self.example()
        forward._update_result(frozen, source, CURRENT)
        saved = copy.deepcopy(frozen)
        older = copy.deepcopy(source)
        older['result_sync_captured_at'] = '2026-10-20T17:56:00+09:00'
        older['result_captured_at'] = '2026-10-20T19:00:00+09:00'
        older['payouts'] = {'4-3-5': 9999}
        forward._update_result(frozen, older, CURRENT)
        self.assertEqual(frozen, saved)
        newer = copy.deepcopy(source)
        newer['result_sync_captured_at'] = '2026-10-20T18:00:00+09:00'
        newer['result_captured_at'] = '2026-10-20T17:00:00+09:00'
        newer['result_sync_source_sha256'] = 'e'*64
        newer['payouts'] = {'4-3-5': 6670}
        forward._update_result(frozen, newer, CURRENT)
        self.assertEqual(frozen['result']['captured_at'], newer['result_sync_captured_at'])
        self.assertEqual(frozen['settlement']['candidate']['flat']['returned'], 6670)

    def test_strict_original_identity_rejection_cannot_create_a_receipt(self):
        frozen, source, path = self.example(synchronize=False)
        snapshot = json.loads(path.read_text())
        snapshot['meta']['race_id'] = '052720261020'
        save_json(path, snapshot)
        self.assertFalse(sync._apply_saved_outcome(self.root/'outputs', source))
        self.assertNotIn('result_sync_source_sha256', source)
        forward._update_result(frozen, source, CURRENT)
        self.assertEqual(frozen['status'], 'result_pending')

    def test_refunds_remain_excluded_for_both_plans(self):
        frozen, source, _ = self.example()
        source['refunded'] = ['1-2-3']
        forward._update_result(frozen, source, CURRENT)
        self.assertEqual(frozen['status'], 'excluded')
        self.assertEqual(frozen['settlement'], {})


if __name__ == '__main__':
    unittest.main()
