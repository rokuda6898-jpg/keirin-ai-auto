"""Recorded withdrawal evidence must never be counted as winning payouts."""
import ast
import copy
import itertools
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import unittest
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from conditional_budget import allocate, RANGE_VERSION, settle
from official_outcomes import normalize_outcome


def finite(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


# These result-only helpers require no model training assets or network imports.
tree = ast.parse((ROOT/'src/conditional_live.py').read_text())
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('saved_outcome_evidence', 'apply_result_evidence')]
scope = dict(re=re, datetime=datetime, read_json=read_json, finite=finite,
             normalize_outcome=normalize_outcome, settle_budget=settle)
exec(compile(ast.Module(body=functions, type_ignores=[]), 'conditional_live_result_helpers', 'exec'), scope)
saved_outcome_evidence = scope['saved_outcome_evidence']
apply_result_evidence = scope['apply_result_evidence']


class ConditionalRefundEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.rid = '082720261019'
        self.path = self.root/'data/raw/nexus_page_snapshots/2026-10-19/outcomes'/f'{self.rid}.json'
        self.path.parent.mkdir(parents=True)
        refunds = ['-'.join(map(str, t)) for t in itertools.permutations(range(1, 10), 3) if 3 in t]
        self.tickets = refunds[:12]
        plan = allocate(self.tickets, {t: '100' for t in self.tickets}, budget=5000)
        plan['version'] = RANGE_VERSION
        close = datetime.fromisoformat('2026-10-19T17:30:00+09:00').timestamp()
        self.row = dict(race_id=self.rid, date='2026-10-19', race_no='8',
                        close_at=close,
                        snapshot_at_jst='2026-10-19T07:00:00+09:00',
                        top12=self.tickets, probabilities=[.025]*12,
                        budget=plan, cancelled_cars=[], actual=refunds+['9-1-2'])
        self.result = dict(official_result_available=True,
                           actual_trifecta_buys=refunds+['9-1-2'],
                           payouts_trifecta_json={**{t: 100 for t in refunds}, '9-1-2': 3120},
                           result_observed_at_jst='2026-10-19T23:00:00+09:00')
        quotes = [dict(id=f'{self.rid}:2:'+'.'.join(map(str, key)), raceId=self.rid,
                       key=list(key), unitPrice=100,
                       payoffUnitPrice=3120 if key == (9, 1, 2) else 100 if 3 in key else 0,
                       absent=3 in key) for key in itertools.permutations(range(1, 10), 3)]
        self.snapshot = dict(schema=1, source_kind='existing_public_page_response',
            sha256='a'*64,
            captured_at='2026-10-19T18:00:00+09:00', url='https://example.invalid/recorded-page',
            meta=dict(race_id=self.rid, date='2026-10-19', race_number=8, venue_id='27',
                      cancel=False, close_at=close),
            data=dict(race=dict(id=self.rid, number=8, entriesNumber=9, closeAt=close, cancel=False,
                               decidedAt=close+900),
                      schedule=dict(date='20261019'),
                      entries=[dict(raceId=self.rid, number=c, absent=c == 3) for c in range(1, 10)],
                      trifectaWinningOddsIds=[f'{self.rid}:2:9.1.2'], trifectaPayoffStatus=1,
                      trifecta=quotes))
        self.path.write_text(json.dumps(self.snapshot))

    def test_official_winning_ids_override_168_refund_payoffs_without_changing_bets(self):
        frozen = {key: copy.deepcopy(self.row[key]) for key in
                  ('top12', 'probabilities', 'snapshot_at_jst', 'budget')}
        self.assertEqual(len(normalize_outcome(self.result)['winning_buys']), 169)
        evidence = saved_outcome_evidence(self.root, self.row)
        apply_result_evidence(self.row, self.result, evidence)
        self.assertEqual(self.row['actual'], ['9-1-2'])
        self.assertEqual(self.row['payouts'], {'9-1-2': 3120})
        self.assertEqual(self.row['cancelled_cars'], ['3'])
        self.assertEqual(self.row['budget_official_outcome']['winning_buys'], ['9-1-2'])
        self.assertEqual(self.row['budget_settlement']['status'], 'refund_pending')
        self.assertIsNone(self.row['budget_settlement']['hit'])
        self.assertIsNone(self.row['budget_settlement']['payout_yen'])
        self.assertEqual({key: self.row[key] for key in frozen}, frozen)

    def test_mismatched_race_date_or_number_cannot_supply_evidence(self):
        for field, value in (('race_id', '092720261019'), ('date', '2026-10-18'), ('race_number', 9)):
            with self.subTest(field=field):
                bad = copy.deepcopy(self.snapshot)
                bad['meta'][field] = value
                self.path.write_text(json.dumps(bad))
                self.assertIsNone(saved_outcome_evidence(self.root, self.row))

    def test_later_generic_result_and_empty_csv_do_not_erase_confirmed_withdrawal(self):
        apply_result_evidence(self.row, self.result, saved_outcome_evidence(self.root, self.row))
        self.path.unlink()
        apply_result_evidence(self.row, self.result, saved_outcome_evidence(self.root, self.row), [])
        self.assertEqual(self.row['cancelled_cars'], ['3'])
        self.assertEqual(self.row['actual'], ['9-1-2'])
        self.assertEqual(self.row['payouts'], {'9-1-2': 3120})
        self.assertEqual(self.row['budget_settlement']['status'], 'refund_pending')


if __name__ == '__main__':
    unittest.main()
