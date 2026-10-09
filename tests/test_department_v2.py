import copy
import json
import math
import tempfile
import unittest
from datetime import datetime, timedelta
from itertools import permutations
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from betting_logic import score_riders, select_race, race_plan
from department_ticket_v2 import candidate_portfolios, distributions, take, public_preserved
from department_context import opponent_profiles, context_for
from department_experiment_v2 import (PRODUCER, RULES, LEDGER, OUTCOMES, SOURCES,
    make_record, append_record, valid_record, seal, digest, latest_records,
    build_report, comparisons, outcomes_for, evaluation, metrics)
from official_outcomes import normalize_outcome, ticket_return, winning_orders, winning_ticket_values


class DepartmentV2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.now = datetime(2026, 10, 8, 12, tzinfo=ZoneInfo('Asia/Tokyo'))
        self.race = pd.DataFrame({'race_id': 'fixture', 'date': '2026-10-08', 'race_no': 1,
            'venue': 'fixture', 'car_no': range(1, 8), 'player_id': [str(c) for c in range(1, 8)],
            'p_win': [.45, .20, .13, .08, .06, .05, .03], 'p_second': 1/7, 'p_third': 1/7,
            'department_score_second': [10, 30, 15, 10, 8, 20, 7],
            'department_score_third': [5, 10, 10, 10, 15, 20, 30],
            'line_id': [1, 1, 1, 2, 2, 3, 3], 'line_position': [1, 2, 3, 1, 2, 1, 2],
            'line_verification_status': 'verified', 'style': ['逃', '追', '追', '逃', '追', '逃', '追'],
            'close_at': (self.now + timedelta(minutes=20)).timestamp()})
        self.departments = ('data_department', 'pace_department', 'line_department')
        self.inputs = {d: self.race.copy() for d in (*self.departments, 'risk_department')}
        self.scored = score_riders(self.race)
        self.scores = self.pack(self.scored)
        joint = distributions(self.scores)[3]
        self.quotes = {f'{a}-{b}-{c}': min(9000, 1.6 / p) for (a,b,c),p in joint.items()}
        self.market = pd.DataFrame([{'race_id': 'fixture', 'bet_type': 'trifecta', 'buy': buy,
            'odds_used': price, 'odds_captured_at_jst': self.now.isoformat()} for buy,price in self.quotes.items()])
        self.knowledge = {'window_end_exclusive': '2026-10-08', 'fingerprint': 'test', 'profiles': {}}
        self.provenance = {'producer': PRODUCER, 'models': {'fixture.joblib': 'a'*64},
                           'model_source': 'fixture', 'position_source': ['fixture']}

    def pack(self, frame):
        return [{'car_no':int(r.car_no), 'first':r.score_first, 'second':r.score_second,
                 'third':r.score_third} for r in frame.itertuples()]

    def record(self):
        row, reason = make_record(self.race, self.inputs, self.market, self.now, self.knowledge, self.provenance)
        self.assertEqual(reason, 'eligible')
        self.assertTrue(valid_record(row))
        return row

    def test_scale_alone_cannot_change_candidates_probabilities_or_counts(self):
        policy = race_plan(self.scored)
        before = candidate_portfolios(self.scores, self.quotes, policy)
        scaled = [{**r, 'second': r['second']*123, 'third':r['third']/1000} for r in self.scores]
        after = candidate_portfolios(scaled, self.quotes, policy)
        for buy,p in before['distribution'].items():
            self.assertAlmostEqual(p, after['distribution'][buy])
        for group in ('本線', '穴'):
            self.assertEqual([t['buy'] for t in before['pools'][group]], [t['buy'] for t in after['pools'][group]])

    def test_prefix_probability_controls_candidate_gate_and_is_not_blended_twice(self):
        scores = [{'car_no':c, 'first':60 if c==1 else 40/6,
                   'second':100 if c==2 else 1, 'third':.9 if c==7 else 1} for c in range(1,8)]
        quotes = {'-'.join(map(str,t)): 5 if t==(1,2,7) else 6000 for t in permutations(range(1,8),3)}
        policy = {'risk_score':0, 'first_fixed':False}
        old = candidate_portfolios(scores, quotes, policy)
        new = candidate_portfolios(scores, quotes, policy, conditional=True)
        self.assertNotIn('1-2-7', [t['buy'] for t in old['pools']['本線']])
        selected = next(t for t in new['pools']['本線'] if t['buy']=='1-2-7')
        self.assertEqual(selected['prob'], new['distribution']['1-2-7'])
        self.assertAlmostEqual(sum(new['distribution'].values()), 1)
        market = candidate_portfolios(scores, quotes, policy, market_only=True)
        self.assertAlmostEqual(sum(market['distribution'].values()), 1)
        self.assertEqual(market['pools']['本線'][0]['buy'], '1-2-7')

    def test_live_preservation_never_increases_either_existing_ticket_count(self):
        preserved = score_riders(self.race, self.market, preserve_position_scores=True)
        frame, policy = select_race(preserved, self.market)
        tickets = public_preserved(preserved, self.market, frame, policy)
        for group in ('本線', '穴'):
            self.assertLessEqual(sum(t['group']==group for t in tickets), int((frame.is_selected & frame.ticket_group.eq(group)).sum()))
        profiles = {str(c): {'line_positions':{str(int(self.race.iloc[c-1].line_position)):{'races':40}}} for c in range(1,8)}
        context,_=context_for('line_department',self.race,profiles)
        contextual=public_preserved(preserved,self.market,frame,policy,context=context)
        for group in ('本線','穴'):
            self.assertLessEqual(sum(t['group']==group for t in contextual),int((frame.is_selected & frame.ticket_group.eq(group)).sum()))
        self.assertTrue(all(t['odds']>=100 for t in contextual if t['group']=='穴'))

    def test_capture_keeps_exact_legacy_risk_and_equal_budget_views(self):
        row = self.record()
        self.assertTrue(valid_record(json.loads(json.dumps(row))))
        self.assertEqual(row['evidence']['quotes'][0]['odds_used'], self.market.iloc[0].odds_used)
        frame, _ = select_race(score_riders(self.inputs['risk_department'], self.market), self.market)
        expected = [{'buy':str(t.buy), 'group':str(t.ticket_group), 'stake_yen':100,
                     'odds':float(t.odds_used), 'prob':float(t.prob), 'ev':float(t.ev)} for t in frame[frame.is_selected].itertuples()]
        self.assertEqual(row['risk_tickets'], expected)
        for view, groups in row['views'].items():
            for group, arms in groups.items():
                count = 6 if view=='fixed_six' else sum(t['group']==group for t in expected)
                self.assertTrue(all(len(t)==count and sum(x['stake_yen'] for x in t)==count*100 for t in arms.values() if t))
        self.assertIsNone(take({'available':True,'pools':{'穴':[]}}, '穴', 6))

    def test_position_market_support_changes_the_relevant_prefix_only(self):
        scores = [{'car_no':c, 'first':1, 'second':1, 'third':1} for c in range(1,5)]
        uniform = {p:1/24 for p in permutations(range(1,5),3)}
        changed = {p:v*(9 if p==(1,2,4) else 1) for p,v in uniform.items()}
        before = distributions(scores, uniform)
        after = distributions(scores, changed)
        self.assertEqual(before[0], after[0])
        self.assertGreater(after[1][1][2], before[1][1][2])
        self.assertGreater(after[2][1,2][4], before[2][1,2][4])
        self.assertEqual(before[2][2,1], after[2][2,1])

    def test_mixed_snapshot_and_capped_prices_preserve_nonmarket_capture(self):
        market = self.market.copy()
        market.loc[0,'odds_captured_at_jst'] = (self.now-timedelta(seconds=20)).isoformat()
        row, reason = make_record(self.race,self.inputs,market,self.now,self.knowledge,self.provenance)
        self.assertEqual(reason, 'partial_market_recorded')
        self.assertTrue(valid_record(row))
        self.assertFalse(row['arms']['data_department:conditional']['available'])
        market.loc[0,'odds_used'] = 9999.9
        row, reason = make_record(self.race,self.inputs,market,self.now,self.knowledge,self.provenance)
        self.assertTrue(valid_record(row))
        self.assertTrue(row['arms']['data_department:preserve']['available'])
        self.assertNotIn(market.iloc[0].buy, row['evidence']['usable_quotes'])

    def test_canonical_forecast_integration_captures_v2_without_touching_v1(self):
        from annual_knowledge import forecast_departments
        pred = self.race.copy()
        pred.attrs['department_provenance'] = self.provenance
        knowledge = {**self.knowledge,'asof_date':'2026-10-08','annual_races':0}
        forecasts = forecast_departments(pred,self.market,knowledge,self.now,self.out)
        self.assertEqual(len(forecasts),4)
        rows,bad = latest_records(self.out/'company')
        self.assertEqual((len(rows),bad),(1,0))
        risk = next(f for f in forecasts if f['department']=='risk_department')
        self.assertEqual([t['buy'] for t in risk['tickets']], [t['buy'] for t in rows[0]['risk_tickets']])
        self.assertFalse((self.out/'company/annual_position_experiment_ledger.jsonl').exists())

    def test_v2_ledgers_survive_restore_without_duplication(self):
        from preserve_validation import save, restore
        row = self.record()
        append_record(row,self.now,self.out/'outputs')
        saved = self.out/'backup'
        save(saved,self.out)
        path = self.out/'outputs/company'/LEDGER
        evidence = path.read_bytes()
        restore(saved,self.out)
        self.assertEqual(path.read_bytes(),evidence)
        path.write_bytes(b'')
        restore(saved,self.out)
        self.assertEqual(path.read_bytes(),evidence)

    def test_missing_quote_records_nonmarket_arms_without_inventing_odds(self):
        market = self.market.iloc[1:].copy()
        row, reason = make_record(self.race, self.inputs, market, self.now, self.knowledge, self.provenance)
        self.assertEqual(reason, 'partial_market_recorded')
        self.assertTrue(valid_record(row))
        self.assertTrue(row['arms']['data_department:preserve']['available'])
        self.assertFalse(row['arms']['data_department:conditional']['available'])
        self.assertFalse(row['arms']['data_department:market']['available'])
        self.assertNotIn(self.market.iloc[0].buy, row['evidence']['usable_quotes'])

    def test_input_route_is_explicit_and_old_uniform_route_cannot_enter(self):
        row, reason = make_record(self.race, self.inputs, self.market, self.now, self.knowledge,
                                   {**self.provenance, 'producer':'uniform_standalone'})
        self.assertIsNone(row)
        self.assertEqual(reason, 'noncanonical_producer')

    def test_hash_roster_price_and_model_version_validation(self):
        row = self.record()
        wrong = copy.deepcopy(row)
        wrong['evidence']['inputs']['risk_department'][0]['p_win'] = .99
        self.assertFalse(valid_record(wrong))
        self.assertFalse(valid_record(seal(wrong)))
        wrong = copy.deepcopy(row)
        wrong['models'] = {}
        self.assertFalse(valid_record(seal(wrong)))
        wrong = copy.deepcopy(row)
        wrong['evidence']['usable_quotes']['1-2-9'] = 200
        wrong['evidence']['quote_times']['1-2-9'] = self.now.isoformat()
        wrong['input_sha256'] = digest(wrong['evidence'])
        self.assertFalse(valid_record(seal(wrong)))
        self.assertTrue({'position_market.py','high_payout_strategy.py','official_outcomes.py'} <= set(row['code']['files']))

    def test_expiry_no_backfill_and_evaluation_start_does_not_slide(self):
        row = self.record()
        self.assertFalse(append_record(row, self.now+timedelta(minutes=20), self.out))
        self.assertFalse((self.out/'company'/LEDGER).exists())
        self.assertTrue(append_record(row, self.now, self.out))
        self.assertTrue(append_record(row, self.now+timedelta(minutes=1), self.out))
        latest, bad = latest_records(self.out/'company')
        self.assertEqual(bad, 0)
        self.assertEqual(latest[0]['window_start'], self.now.isoformat())
        self.assertEqual(latest[0]['snapshot_at'], (self.now+timedelta(minutes=1)).isoformat())
        prior = self.out/'company/annual_position_experiment_ledger.jsonl'
        prior.write_bytes(b'old frozen evidence\n')
        build_report(self.out, self.now)
        self.assertEqual(prior.read_bytes(), b'old frozen evidence\n')

    def test_two_arm_comparison_not_blocked_by_an_unrelated_variant(self):
        row = self.record()
        arms = row['views']['fixed_six']['穴']
        pool = row['arms']['data_department:baseline']['pools']['穴']
        self.assertGreaterEqual(len(pool), 6)
        arms['data_department:baseline'] = pool[:6]
        arms['data_department:preserve'] = pool[:6]
        arms['data_department:context'] = None
        report = comparisons([row], {}, self.now)
        self.assertEqual(report['data_department:穴:fixed_six:baseline:preserve']['preclose_matched'], 1)
        self.assertEqual(report['data_department:穴:fixed_six:baseline:context']['preclose_matched'], 0)

    def test_multiple_winning_tickets_settle_individually_and_sum_returns(self):
        outcome = normalize_outcome({'official_result_available':True,'actual_trifecta':'1-2-3',
            'payouts_trifecta_json':json.dumps({'1-2-3':10000,'1-2-4':30000})})
        tickets = [{'buy':'1-2-3','stake_yen':100},{'buy':'1-2-4','stake_yen':200}]
        self.assertEqual(ticket_return(tickets, outcome), (True, 70000))
        self.assertEqual(winning_orders([(1,1),(2,2),(3,3),(3,4)]), ['1-2-3','1-2-4'])
        self.assertEqual(winning_orders([(1,1),(1,2),(3,3)]), ['1-2-3','2-1-3'])
        self.assertIn('1-2-4', winning_ticket_values({'actual_trifecta':'1-2-3','payouts_trifecta_json':{'1-2-3':10000,'1-2-4':30000}}))
        incomplete = normalize_outcome({'official_result_available':True,'actual_trifecta':'1-2-3',
                        'actual_trifecta_buys':['1-2-3','1-2-4'],'actual_trifecta_odds':100})
        self.assertEqual(ticket_return(tickets, incomplete), (True, None))

    def test_stale_official_observation_does_not_overwrite_a_correction(self):
        folder = self.out/'company'
        folder.mkdir()
        result = {'race_id':'fixture','official_result_available':True,'actual_trifecta':'1-2-3',
                  'actual_trifecta_odds':100,'result_observed_at_jst':'2026-10-08T13:00:00+09:00'}
        path = self.out/'latest_results.json'
        path.write_text(json.dumps([result]))
        outcomes_for(folder, {'fixture'})
        path.write_text(json.dumps([{**result,'actual_trifecta':'1-2-4','result_observed_at_jst':'2026-10-08T13:01:00+09:00'}]))
        outcomes_for(folder, {'fixture'})
        path.write_text(json.dumps([result]))
        outcomes, conflicts = outcomes_for(folder, {'fixture'})
        self.assertEqual(outcomes['fixture']['winning_buys'], ['1-2-4'])
        self.assertEqual(conflicts, ['fixture'])
        self.assertEqual(len((folder/OUTCOMES).read_text().splitlines()), 2)

    def test_code_revisions_are_separate_cohorts(self):
        row = self.record()
        append_record(row, self.now, self.out)
        second = copy.deepcopy(row)
        second['code']['files']['betting_logic.py'] = 'b'*64
        second['code']['sha256'] = digest(second['code']['files'])
        append_record(seal(second), self.now, self.out)
        report = build_report(self.out, self.now)
        self.assertEqual(len(report['cohorts']), 2)
        self.assertEqual(report['frozen_races'], 1)

    def test_fixed_deadline_insufficient_data_never_auto_promotes(self):
        end = self.now+timedelta(days=56)
        self.assertEqual(evaluation([], 'fixed_six','穴','a','b',end,self.now,True)['status'], 'collecting')
        report = evaluation([], 'fixed_six','穴','a','b',end,end,True)
        self.assertEqual(report['status'], 'insufficient_at_fixed_deadline')
        self.assertFalse(report['automatic_promotion'])

    def test_fixed_deadline_evaluates_hit_and_roi_on_distinct_evidence(self):
        row = {'views':{'fixed_six':{'穴':{'a':[{'buy':'1-2-3','stake_yen':100}]*6,
                                           'b':[{'buy':'1-2-4','stake_yen':100}]*6}}},
               'outcome':{'winning_buys':['1-2-4'],'payouts':{},'payout_complete':False},
               'snapshot_at':self.now.isoformat()}
        samples = [{**row,'date':(self.now+timedelta(days=i%28)).date().isoformat()} for i in range(504)]
        end = self.now+timedelta(days=56)
        result = evaluation(samples,'fixed_six','穴','a','b',end,end,True)
        self.assertEqual(result['status'],'fixed_deadline_evaluated')
        self.assertTrue(result['hit_improvement_supported'])
        self.assertIsNone(result['return_rate_difference_interval'])
        self.assertFalse(result['roi_improvement_supported'])

    def test_opponent_context_excludes_future_and_updates_ratings_after_whole_day(self):
        history = pd.DataFrame([{'race_id':rid,'date':day,'player_id':str(c),'finish_pos':c,
                                'player_class':1.0,'entries_number':3}
            for rid,day in [('a','2026-10-06'),('b','2026-10-06'),('future','2026-10-08')] for c in (1,2,3)])
        profiles, coverage = opponent_profiles(history, '2026-10-08')
        self.assertEqual(coverage['complete_races'], 2)
        self.assertEqual(profiles['1']['buckets'][0]['opponent_rating'], 1500)
        self.assertEqual(profiles['1']['buckets'][0]['races'], 2)
        self.assertEqual(profiles['1']['buckets'][0]['class'], '1')
        reordered, _ = opponent_profiles(history.sample(frac=1, random_state=7), '2026-10-08')
        self.assertEqual(profiles, reordered)
        missing, excluded = opponent_profiles(history.drop(columns='entries_number'),'2026-10-08')
        self.assertEqual(missing,{})
        self.assertEqual(excluded['complete_races'],0)

    def test_specialist_context_uses_ordered_line_survivors_and_missing_data_is_explicit(self):
        profiles = {str(c): {'line_positions':{str(int(self.race.iloc[c-1].line_position)):{'races':40}},
                             'events':{'back':{'observed':40}}} for c in range(1,8)}
        fn, evidence = context_for('line_department', self.race, profiles)
        self.assertEqual(evidence['status'], 'available')
        self.assertNotEqual(fn(1,2,3,3), fn(2,1,3,3))
        self.assertNotEqual(fn(1,2,3,3), fn(1,2,4,3))
        fn, evidence = context_for('pace_department', self.race, profiles)
        self.assertNotEqual(fn(1,2,3,3), fn(1,2,4,3))
        fn, evidence = context_for('line_department', self.race.assign(line_verification_status='unknown'), profiles)
        self.assertIsNone(fn)
        self.assertEqual(evidence['status'], 'verified_line_required')


if __name__ == '__main__':
    unittest.main()
