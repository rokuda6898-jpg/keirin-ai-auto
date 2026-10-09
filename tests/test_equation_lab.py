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

import equation_lab as lab
import equation_models as models
import test_department_v2 as fixtures
from department_experiment_v2 import append, seal
from official_outcomes import ticket_return


def small_sample(day='2026-10-01', rid='training'):
    riders = [{'car_no': c, 'p_win': p, 'score': 90-c*2, 'back_count': 8-c,
               'place2_rate': .2, 'place3_rate': .1, 'recent_avg_finish': c,
               'line_id': 1 if c<3 else 2, 'line_verification_status': 'verified'}
              for c, p in enumerate((.5, .3, .15, .05), 1)]
    return {'race_id': rid, 'date': day, 'riders': riders, 'actual': '1-2-3',
            'quotes': {'-'.join(map(str,t)):100.0 for t in permutations(range(1,5),3)}}


class EquationModelTests(unittest.TestCase):
    def test_distinct_models_fit_without_target_day_and_normalize(self):
        samples = [small_sample(f'2026-10-0{i+1}',str(i)) for i in range(7)]
        with patch.dict(models.CONFIG, minimum_training_races=7, iterations=30):
            fitted=models.fit(samples)
        distributions=[]
        for name in ('market_residual','pairwise_order'):
            p=models.predict(name,fitted['models'][name],samples[0]['riders'],samples[0]['quotes'])
            self.assertAlmostEqual(sum(p.values()),1)
            self.assertGreater(p['1-2-3'],p['4-3-2'])
            distributions.append(p)
        self.assertNotEqual(distributions[0],distributions[1])
        self.assertIsNone(fitted['models']['state_paths'])

    def test_no_model_is_invented_for_insufficient_training(self):
        fitted=models.fit([small_sample()])
        self.assertTrue(all(v is None for v in fitted['models'].values()))

    def test_independent_provisional_picks_are_separate_and_labelled(self):
        from equation_preview import provisional_distribution, standalone_tickets
        sample = small_sample()
        keys = list(sample['quotes'])
        quotes = {k: (150.0 if i % 2 else 24.0) for i, k in enumerate(keys)}
        distributions = {m: provisional_distribution(m, sample['riders'], quotes)
                         for m in models.METHODS}
        for name, distribution in distributions.items():
            self.assertEqual(set(distribution), set(quotes))
            self.assertAlmostEqual(sum(distribution.values()), 1.0)
            picks = standalone_tickets(distribution, quotes)
            self.assertEqual(len(picks['本線']), 12)
            self.assertEqual(len(picks['穴']), 12)
            self.assertTrue(all(t['odds'] >= 100 for t in picks['穴']))
            self.assertTrue(all(t['odds'] < 100 for t in picks['本線']))
        self.assertNotEqual(distributions['market_residual'], distributions['pairwise_order'])
        self.assertNotEqual(distributions['state_paths'], distributions['pairwise_order'])

    def test_pairwise_equation_is_invariant_to_car_renaming(self):
        s=small_sample(); beta={'beta':[.3,-.2,.1,.4,.2,-.5,0,0]}
        p=models.predict('pairwise_order',beta,s['riders'],s['quotes'])
        mapping={1:4,2:1,3:2,4:3}
        riders=[{**r,'car_no':mapping[r['car_no']]} for r in s['riders']]
        renamed=lambda t:'-'.join(str(mapping[int(c)]) for c in t.split('-'))
        q=models.predict('pairwise_order',beta,riders,{renamed(k):v for k,v in s['quotes'].items()})
        for k,v in p.items():self.assertAlmostEqual(v,q[renamed(k)])

    def test_future_result_fields_do_not_enter_features(self):
        s=small_sample()
        original=models.joint_features(s['riders'],(1,2,3))
        for r in s['riders']:
            r.update(finish_pos=1,actual_trifecta='4-3-2',result_event_leftBehind=True)
        self.assertEqual(original,models.joint_features(s['riders'],(1,2,3)))

    def test_path_formula_requires_observations_and_learns_transition(self):
        samples=[small_sample(f'2026-10-0{i+1}',str(i)) for i in range(7)]
        paths=[{'race_id':str(i),'orders':{'start':[3,2,1],'bell':[2,3,1],
                'back':[1,3,2],'finish':[1,2,3]}} for i in range(7)]
        with patch.dict(models.CONFIG, minimum_training_races=7):
            model=models.fit_paths(samples,paths)
        p=models.predict('state_paths',model,samples[0]['riders'],samples[0]['quotes'])
        self.assertAlmostEqual(sum(p.values()),1)
        self.assertEqual(max(p,key=p.get),'1-2-3')
        self.assertIsNone(models.fit_paths(samples,[]))
        paths[0]['orders']['finish']=[4,3,2]
        with patch.dict(models.CONFIG,minimum_training_races=7):
            self.assertIsNone(models.fit_paths(samples,paths))

    def test_partial_and_capped_markets_are_rejected(self):
        s=small_sample(); q=dict(s['quotes']);q.pop('1-2-3')
        with self.assertRaises(ValueError):models.predict('pairwise_order',{'beta':[0]*8},s['riders'],q)
        q=dict(s['quotes']);q['1-2-3']=9999.9
        with self.assertRaises(ValueError):models.predict('pairwise_order',{'beta':[0]*8},s['riders'],q)


class EquationLedgerTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.DepartmentV2Tests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.out=self.fixture.out
        self.folder=self.out/'company'
        self.now=self.fixture.now
        self.row=self.fixture.record()
        append(self.folder/lab.INPUTS,self.row)

    def capture(self):
        return lab.capture(self.row['input_sha256'],self.row['snapshot_at'],self.out,self.now)

    def test_shadow_waits_without_touching_risk_or_backfilling(self):
        before=(self.folder/lab.INPUTS).read_bytes()
        self.assertEqual(self.capture(),'collecting_training_data')
        self.assertEqual((self.folder/lab.INPUTS).read_bytes(),before)
        rows,_=lab.validated_rows(self.folder)
        self.assertEqual(len(rows),1)
        self.assertFalse(rows[0]['ceo_integration'])
        self.assertFalse(rows[0]['purchase_authorized'])
        self.assertEqual(set(rows[0]['standalone']), set(models.METHODS))
        for method in models.METHODS:
            independent = rows[0]['standalone'][method]
            self.assertEqual(independent['basis'], 'untrained_preclose_proxy')
            self.assertTrue(independent['picks']['本線'] or independent['picks']['穴'])
        report = lab.build_report(self.out, self.now)
        self.assertEqual(len(report['standalone_by_race']), 1)
        page = (self.folder / 'annual_equation_report.html').read_text()
        self.assertIn('方程式ごとの単独買い目', page)
        for g in lab.GROUPS:
            expected=[t for t in self.row['risk_tickets'] if t['group']==g]
            self.assertEqual(rows[0]['views'][g]['risk'],expected or None)
        self.assertEqual(lab.capture(self.row['input_sha256'],self.row['snapshot_at'],self.out,
                         self.now+timedelta(hours=1)),'outside_capture_window')
        self.assertEqual(len(list(lab.checked(self.folder/lab.LEDGER))),1)

    def test_report_does_not_generate_missing_past_forecasts(self):
        self.assertFalse((self.folder/lab.LEDGER).exists())
        report=lab.build_report(self.out,self.now+timedelta(days=1))
        self.assertEqual(report['recorded_races'],0)
        self.assertFalse((self.folder/lab.LEDGER).exists())
        self.assertFalse(report['ceo_integration'])

    def test_asof_training_excludes_same_day_late_labels_and_ties(self):
        outcome={'winning_buys':['1-2-3'],'payouts':{'1-2-3':1000},'payout_complete':True}
        append(self.folder/lab.OUTCOMES,{'race_id':'fixture','outcome':outcome,
             'observed_at':(self.now+timedelta(hours=1)).isoformat()})
        samples,_,_=lab.prior_samples(self.folder,self.now.date())
        self.assertEqual(samples,[])
        samples,_,_=lab.prior_samples(self.folder,(self.now+timedelta(days=1)).date())
        self.assertEqual(len(samples),1)
        # Conflicting labels cannot be used to manufacture a fit.
        append(self.folder/lab.OUTCOMES,{'race_id':'fixture','outcome':{**outcome,'winning_buys':['2-1-3']},
             'observed_at':(self.now+timedelta(hours=2)).isoformat()})
        samples,_,info=lab.prior_samples(self.folder,(self.now+timedelta(days=1)).date())
        self.assertEqual(samples,[]);self.assertEqual(info['conflicting_results'],['fixture'])

    def test_daily_model_is_frozen(self):
        a=lab.prepare_models(self.out,self.now)
        with patch('equation_lab.fit',side_effect=AssertionError('no refit')):
            b=lab.prepare_models(self.out,self.now+timedelta(minutes=1))
        self.assertEqual(a,b)

    def test_training_does_not_accept_unverified_or_postcutoff_paths(self):
        append(self.folder/lab.OUTCOMES,{'race_id':'fixture','outcome':{'winning_buys':['1-2-3'],
            'payouts':{'1-2-3':1000},'payout_complete':True},'observed_at':(self.now+timedelta(hours=1)).isoformat()})
        path={'race_id':'fixture','schema':'observed_top3_stages_v1','verified':False,
              'source_url':'https://example.test/video','observed_at':(self.now+timedelta(hours=2)).isoformat(),
              'orders':{'start':[1,2,3],'bell':[1,2,3],'back':[1,2,3],'finish':[1,2,3]}}
        (self.folder/lab.PATHS).write_text(json.dumps(path)+'\n',encoding='utf-8')
        _,paths,_=lab.prior_samples(self.folder,(self.now+timedelta(days=1)).date())
        self.assertEqual(paths,[])

    def test_no_padding_and_group_specific_equal_stakes(self):
        prices={'1-2-3':110,'2-1-3':90};p={'1-2-3':.8,'2-1-3':.2}
        self.assertIsNone(lab.choose(p,prices,'穴',2))
        picks=lab.choose(p,prices,'穴',1)
        self.assertEqual(picks[0]['stake_yen'],100)
        self.assertGreaterEqual(picks[0]['odds'],100)

    def test_record_tampering_and_resealed_risk_changes_fail(self):
        self.capture()
        rows=list(lab.checked(self.folder/lab.LEDGER))
        rows[0]['ceo_integration']=True
        (self.folder/lab.LEDGER).write_text(json.dumps(seal(rows[0]))+'\n',encoding='utf-8')
        with self.assertRaises(ValueError):lab.validated_rows(self.folder)

    def test_expired_calculation_cannot_save_a_prediction(self):
        frozen=lab.prepare_models(self.out,self.now)
        with patch('equation_lab.prepare_models',return_value=frozen), patch('equation_lab.time.monotonic',side_effect=[0,901]):
            self.assertEqual(self.capture(),'expired_during_equation_calculation')
        self.assertFalse((self.folder/lab.LEDGER).exists())

    def test_learned_models_capture_and_settle_only_future_race(self):
        append(self.folder/lab.OUTCOMES,{'race_id':'fixture','outcome':{'winning_buys':['1-2-3'],
            'payouts':{'1-2-3':1000},'payout_complete':True},'observed_at':(self.now+timedelta(hours=1)).isoformat()})
        next_day=self.now+timedelta(days=1)
        f=self.fixture
        f.now=next_day
        for frame in (f.race,*f.inputs.values()):
            frame['race_id']='future';frame['date']='2026-10-09'
            frame['close_at']=(next_day+timedelta(minutes=20)).timestamp()
        f.market['race_id']='future';f.market['odds_captured_at_jst']=next_day.isoformat()
        f.knowledge['window_end_exclusive']='2026-10-09'
        source=f.record();append(self.folder/lab.INPUTS,source)
        with patch.dict(models.CONFIG,minimum_training_races=1,minimum_training_days=1,iterations=2):
            reason=lab.capture(source['input_sha256'],source['snapshot_at'],self.out,next_day)
            self.assertEqual(reason,'captured_shadow')
            rows,bundles=lab.validated_rows(self.folder)
            self.assertEqual(rows[0]['race_id'],'future')
            model=next(iter(bundles.values()))
            self.assertEqual([r['race_id'] for r in model['training_manifest']],['fixture'])
            for method in ('market_residual','pairwise_order'):
                self.assertAlmostEqual(sum(rows[0]['distributions'][method].values()),1)
            report=lab.build_report(self.out,next_day)
            self.assertTrue(all(p['challenger']['races']==0 for p in report['comparisons']))

    def test_fixed_deadline_does_not_extend_until_it_wins(self):
        report=lab.evaluate([],models.METHODS[0],'本線',self.now,self.now+timedelta(days=57))
        self.assertEqual(report['status'],'insufficient_at_fixed_deadline')
        self.assertFalse(report['automatic_promotion']);self.assertFalse(report['ceo_integration'])

    def test_multiple_winners_settle_without_reclassifying_using_final_odds(self):
        tickets=[{'buy':'1-2-3','stake_yen':100,'group':'穴','odds':110},
                 {'buy':'1-3-2','stake_yen':100,'group':'穴','odds':120}]
        self.assertEqual(ticket_return(tickets,{'winning_buys':['1-2-3','1-3-2'],
            'payouts':{'1-2-3':9000,'1-3-2':8000},'payout_complete':True}),(True,17000))


if __name__=='__main__':unittest.main()
