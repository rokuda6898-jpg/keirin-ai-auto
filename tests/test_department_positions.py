import copy
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from itertools import permutations
from pathlib import Path
from zoneinfo import ZoneInfo
from unittest.mock import patch

import numpy as np
import pandas as pd

from betting_logic import score_riders, select_race, race_plan, generate_formations
from position_market import position_market
from annual_knowledge import forecast_departments
from department_position_experiment import (DEPARTMENTS, VARIANTS, LEDGER, POLICY, VERSION,
    make_bundle, append_bundle, valid_bundle, latest_bundles, summarize, build_report)


class DepartmentPositionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.now = datetime(2026, 10, 8, 12, tzinfo=ZoneInfo('Asia/Tokyo'))
        self.race = pd.DataFrame({'race_id': 'fixture', 'date': '2026-10-08', 'race_no': 1,
            'car_no': range(1, 8), 'player_id': [str(c) for c in range(1, 8)],
            'p_win': [.3, .2, .15, .12, .10, .08, .05], 'p_second': 1/7, 'p_third': 1/7,
            'department_score_second': [2, 3, 4, 5, 6, 60, 20],
            'department_score_third': [2, 3, 4, 5, 6, 20, 60],
            'line_id': [1, 1, 2, 2, 3, 3, 4], 'line_position': [1, 2, 1, 2, 1, 2, 1],
            'close_at': (self.now + timedelta(minutes=20)).timestamp()})
        scored = score_riders(self.race)
        cand = generate_formations(scored, race_plan(scored))
        self.market = cand[['buy', 'bet_type']].assign(race_id='fixture', odds_used=1.4/cand.prob,
                                                     odds_captured_at_jst=self.now.isoformat())

    def tearDown(self):
        self.tmp.cleanup()

    def bundle(self):
        inputs = {d: self.race.copy() for d in (*DEPARTMENTS, 'risk_department')}
        return make_bundle(self.race, inputs, self.market, self.now, '2026-10-08')[0]

    def simple_bundle(self):
        ticket = {'buy': '1-2-3', 'group': '穴', 'stake_yen': 100, 'odds': 200}
        arms = {'risk'} | {d + ':' + v for d in DEPARTMENTS for v in VARIANTS}
        return {'version': VERSION, 'snapshot_policy': POLICY, 'race_id': 'fixture',
                'snapshot_at': self.now.isoformat(), 'quote_snapshot_at': self.now.isoformat(),
                'close_at': (self.now + timedelta(minutes=20)).timestamp(),
                'arms': {a: {'tickets': [dict(ticket)]} for a in arms}}

    def test_preservation_is_explicit_and_survives_model_source_and_row_order(self):
        raw = self.race.assign(position_model_source='position_specialists').sample(frac=1, random_state=1)
        fixed = score_riders(raw, self.market, preserve_position_scores=True)
        self.assertEqual(fixed.loc[fixed.rank_second.eq(1), 'car_no'].item(), 6)
        self.assertEqual(fixed.loc[fixed.rank_third.eq(1), 'car_no'].item(), 7)
        self.assertEqual(fixed.score_third.tolist(), [2, 3, 4, 5, 6, 20, 60])
        old = score_riders(raw, self.market)
        self.assertNotEqual(old.score_third.tolist(), fixed.score_third.tolist())
        a, _ = select_race(old, self.market)
        b, _ = select_race(fixed, self.market)
        self.assertNotEqual(set(a.loc[a.is_selected, 'buy']), set(b.loc[b.is_selected, 'buy']))
        with self.assertRaises(ValueError):
            score_riders(raw.assign(department_score_third=np.nan), preserve_position_scores=True)

    def test_market_support_uses_each_position(self):
        market = self.market.copy()
        market['odds_used'] = market.buy.map(lambda b: 10 if b.split('-')[1] == '7' else 1000)
        context = position_market(score_riders(self.race), market)
        self.assertEqual(context['ranks'][1][7], 1)
        self.assertEqual(context['ranks'][0][7], 7)
        self.assertEqual(context['ranks'][2][7], 7)

    def test_third_depends_on_ordered_prefix_not_only_exclusion(self):
        market = self.market.assign(odds_used=2000.)
        market.loc[market.buy.isin(['1-2-3', '2-1-4']), 'odds_used'] = 30
        riders = score_riders(self.race.assign(department_score_second=10, department_score_third=10),
                              preserve_position_scores=True)
        ctx = position_market(riders, market)
        self.assertGreater(ctx['third_given_first_second'][1, 2][3], ctx['third_given_first_second'][1, 2][4])
        self.assertGreater(ctx['third_given_first_second'][2, 1][4], ctx['third_given_first_second'][2, 1][3])
        self.assertAlmostEqual(sum(ctx['probabilities'].values()), 1)
        for a in range(1, 8):
            self.assertAlmostEqual(sum(v for (x, _, _), v in ctx['probabilities'].items() if x == a),
                                   riders.set_index('car_no').at[a, 'p_win'])
        with self.assertRaises(ValueError):
            position_market(riders, market.head(10))

    def test_four_arms_are_frozen_together_and_risk_is_unchanged(self):
        bundle = self.bundle()
        self.assertTrue(valid_bundle(bundle))
        self.assertEqual(len(bundle['arms']), 13)
        legacy, _ = select_race(score_riders(self.race, self.market), self.market)
        self.assertEqual([t['buy'] for t in bundle['arms']['risk']['tickets']], legacy.loc[legacy.is_selected, 'buy'].tolist())
        for arm in bundle['arms'].values():
            self.assertLessEqual(sum(t['group'] == '本線' for t in arm['tickets']), 12)
            self.assertLessEqual(sum(t['group'] == '穴' for t in arm['tickets']), 12)
            self.assertTrue(all(t['odds'] >= 100 for t in arm['tickets'] if t['group'] == '穴'))
        self.assertTrue(append_bundle(bundle, self.now + timedelta(seconds=5), self.root))
        self.assertFalse(append_bundle(bundle, self.now + timedelta(minutes=16), self.root))

    def test_strict_quote_time_and_missing_data_never_fabricated(self):
        inputs = {d: self.race for d in (*DEPARTMENTS, 'risk_department')}
        for market, reason in [
            (self.market.drop(columns='odds_captured_at_jst'), 'odds_schema_missing'),
            (self.market.head(10), 'incomplete_quotes'),
            (self.market.assign(odds_captured_at_jst=(self.now-timedelta(minutes=6)).isoformat()), 'stale_quotes'),
            (self.market.assign(odds_captured_at_jst=(self.now+timedelta(seconds=1)).isoformat()), 'future_capture_time'),
        ]:
            row, actual = make_bundle(self.race, inputs, market, self.now, '2026-10-08')
            self.assertIsNone(row)
            self.assertEqual(actual, reason)
        mixed = self.market.copy()
        mixed.loc[0, 'odds_captured_at_jst'] = (self.now-timedelta(seconds=1)).isoformat()
        self.assertEqual(make_bundle(self.race, inputs, mixed, self.now, '2026-10-08')[1], 'mixed_quote_snapshots')
        self.assertIsNone(make_bundle(self.race, inputs, self.market, self.now+timedelta(minutes=30), '2026-10-08')[0])

    def test_latest_whole_snapshot_zero_tickets_replaces_earlier_win(self):
        first = self.simple_bundle()
        later = copy.deepcopy(first)
        later['snapshot_at'] = (self.now+timedelta(seconds=40)).isoformat()
        for a in later['arms'].values():
            a['tickets'] = []
        path = self.root / LEDGER
        path.write_text('\n'.join(json.dumps(x) for x in [later, first, later]), encoding='utf-8')
        selected, invalid = latest_bundles(path)
        self.assertEqual(selected['fixture']['arms']['risk']['tickets'], [])
        self.assertEqual(invalid, 0)
        invalid_row = copy.deepcopy(first)
        invalid_row['snapshot_at'] = (self.now+timedelta(minutes=30)).isoformat()
        self.assertFalse(valid_bundle(invalid_row))

    def test_same_counts_missing_payout_and_outlier_dependency(self):
        hit = {**self.simple_bundle(), 'actual': '1-2-3', 'payout_per_100yen': 20000}
        loss = {**self.simple_bundle(), 'race_id': 'loss', 'actual': '7-6-5', 'payout_per_100yen': 3000}
        pending = {**self.simple_bundle(), 'race_id': 'pending', 'actual': '1-2-3', 'payout_per_100yen': None}
        unequal = copy.deepcopy(hit)
        unequal['race_id'] = 'unequal'
        unequal['arms']['data_department:combined']['tickets'] = []
        summary, risk = summarize([hit, loss, pending, unequal])
        paired = summary['data_department:穴']
        self.assertEqual(paired['paired_races'], 3)
        self.assertEqual(paired['excluded']['unequal_ticket_counts'], 1)
        self.assertEqual(paired['arms']['risk']['hits'], 2)
        self.assertEqual(paired['arms']['risk']['stake_yen'], 200)
        self.assertEqual(paired['arms']['risk']['return_rate'], 100)
        _, dependency = summarize([hit, loss])
        self.assertEqual(dependency['hole_return_share'], 1)
        self.assertEqual(dependency['groups']['穴']['largest_hit_return_share'], 1)
        self.assertEqual(dependency['groups']['穴']['return_rate_without_largest_hit'], 0)

    def test_no_legacy_backfill_and_official_outcomes_survive_daily_overwrite(self):
        folder = self.root / 'company'
        folder.mkdir()
        (self.root / 'latest_results.json').write_text(json.dumps([{'race_id': 'fixture',
            'official_result_available': True, 'actual_trifecta': '1-2-3', 'actual_trifecta_odds': 200}]))
        self.assertEqual(build_report(self.root)['settled_races'], 0)
        row = self.simple_bundle()
        (folder / LEDGER).write_text(json.dumps(row)+'\n')
        report = build_report(self.root)
        self.assertEqual(report['settled_races'], 1)
        (self.root / 'latest_results.json').write_text('[]')
        self.assertEqual(build_report(self.root)['settled_races'], 1)
        # An output reset can remove the derived map; the append-only official
        # outcome ledger must still recover the same settled evidence.
        (folder / 'annual_position_experiment_results.json').unlink()
        self.assertEqual(build_report(self.root)['settled_races'], 1)

    def test_validation_restore_preserves_both_new_ledgers_without_duplicates(self):
        from preserve_validation import save, restore
        folder = self.root / 'outputs/company'
        folder.mkdir(parents=True)
        for name in (LEDGER, 'annual_position_experiment_outcomes.jsonl'):
            (folder / name).write_text('{"record":1}\n', encoding='utf-8')
        backup = self.root / 'saved'
        save(backup, self.root)
        for name in (LEDGER, 'annual_position_experiment_outcomes.jsonl'):
            (folder / name).write_text('{"record":2}\n', encoding='utf-8')
        restore(backup, self.root)
        restore(backup, self.root)
        for name in (LEDGER, 'annual_position_experiment_outcomes.jsonl'):
            self.assertEqual(len((folder / name).read_text(encoding='utf-8').splitlines()), 2)

    def test_annual_forecast_keeps_risk_path_and_specialist_lower_places(self):
        profiles = {str(c): {'races': 50, 'rates': [0.2, .6 if c == 6 else .02, .6 if c == 7 else .02]}
                    for c in range(1, 8)}
        report = {'asof_date': '2026-10-08', 'window_end_exclusive': '2026-10-08',
                  'annual_races': 50, 'profiles': profiles}
        calls = []
        def score(*args, **kwargs):
            result = score_riders(*args, **kwargs)
            calls.append((kwargs.get('preserve_position_scores'), result))
            return result
        with patch('annual_knowledge.score_riders', side_effect=score):
            rows = forecast_departments(self.race, self.market, report, self.now, self.root)
        self.assertEqual([v for v, _ in calls], [True, True, True, False])
        self.assertTrue(all(f.loc[f.rank_second.eq(1), 'car_no'].item() == 6 for _, f in calls[:3]))
        risk = next(r for r in rows if r['department'] == 'risk_department')
        self.assertEqual(risk['ticket_strategy'], 'risk_legacy_unchanged')
        before = (self.root / 'company' / LEDGER).read_bytes()
        forecast_departments(self.race, self.market, report, self.now+timedelta(hours=1), self.root)
        self.assertEqual((self.root / 'company' / LEDGER).read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
