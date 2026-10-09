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

import fusion_equations as eq
import fusion_lab as lab
import test_department_v2 as fixtures
from department_experiment_v2 import append, seal, frame_records


def small_packet(day='2026-10-01', rid='small'):
    riders=[{'car_no':c,'p_win':p,'p_second':.25,'p_third':.25,
        'department_score_second':5-c,'department_score_third':c,
        'score':100-c*3,'back_count':8-c,'place2_rate':.2,'place3_rate':.4,
        'recent_avg_finish':c,'line_id':(c+1)//2,'line_position':1 if c%2 else 2,
        'line_verification_status':'verified'} for c,p in enumerate((.5,.3,.15,.05),1)]
    prices={'-'.join(map(str,t)):50.+i*10 for i,t in enumerate(permutations(range(1,5),3))}
    return {'date':day,'actual':'1-2-3','race_id':rid,'source':{'race_id':rid,
        'quote_quality':{'complete_single_snapshot':True},'arms':{},'evidence':{
        'inputs':{d:copy.deepcopy(riders) for d in eq.DEPARTMENTS[:4]},'usable_quotes':prices}},
        'auxiliary':{'riders':copy.deepcopy(riders),'departments':{
            d:{'top3_cars':[1,2,3]} for d in eq.DEPARTMENTS}}}


class FusionModelTests(unittest.TestCase):
    def test_department_second_third_and_all_seven_scenarios_reach_features(self):
        packet=small_packet();before=eq.joint_features(packet,(1,2,3))
        packet['source']['evidence']['inputs']['line_department'][1]['department_score_second']=1000
        self.assertNotEqual(before,eq.joint_features(packet,(1,2,3)))
        for d in eq.DEPARTMENTS:
            changed=small_packet();changed['auxiliary']['departments'][d]['top3_cars']=[3,2,1]
            self.assertNotEqual(before,eq.joint_features(changed,(1,2,3)),d)

    def test_line_order_and_regrouping_change_conditional_features(self):
        packet=small_packet();original=eq.joint_features(packet,(1,2,3))
        packet['source']['evidence']['inputs']['risk_department'][0]['line_position']=2
        packet['source']['evidence']['inputs']['risk_department'][1]['line_position']=1
        self.assertNotEqual(original,eq.joint_features(packet,(1,2,3)))
        packet=small_packet();packet['source']['evidence']['inputs']['risk_department'][2]['line_id']=1
        packet['source']['evidence']['inputs']['risk_department'][2]['line_position']=3
        self.assertNotEqual(original,eq.joint_features(packet,(1,2,3)))

    def test_missing_measurement_is_not_zero_and_unknown_line_is_not_verified(self):
        packet=small_packet();r=packet['source']['evidence']['inputs']['risk_department'][0]
        r['back_count']=0;zero=eq.joint_features(packet,(1,2,3))
        r.pop('back_count');self.assertNotEqual(zero,eq.joint_features(packet,(1,2,3)))
        known=eq.joint_features(packet,(1,2,3));r['line_verification_status']='missing'
        self.assertNotEqual(known,eq.joint_features(packet,(1,2,3)))

    def test_result_columns_never_become_features(self):
        packet=small_packet();before=eq.joint_features(packet,(1,2,3))
        for r in packet['source']['evidence']['inputs']['risk_department']:
            r.update(finish_pos=9,actual_trifecta='4-3-2',official_finish_pos=8)
        self.assertEqual(before,eq.joint_features(packet,(1,2,3)))

    def test_joint_prediction_is_invariant_to_car_renaming(self):
        p=small_packet();q=copy.deepcopy(p);mapping={1:4,2:1,3:2,4:3}
        for rows in [q['auxiliary']['riders'],*q['source']['evidence']['inputs'].values()]:
            for r in rows:r['car_no']=mapping[r['car_no']]
        for d in q['auxiliary']['departments'].values():d['top3_cars']=[mapping[c] for c in d['top3_cars']]
        rename=lambda t:'-'.join(str(mapping[int(c)]) for c in t.split('-'))
        q['source']['evidence']['usable_quotes']={rename(k):v for k,v in p['source']['evidence']['usable_quotes'].items()}
        for t in eq.triples(p):
            self.assertEqual(eq.joint_features(p,t),eq.joint_features(q,tuple(mapping[c] for c in t)))

    def test_duplicate_expert_even_different_family_cannot_increase_influence(self):
        p={'1-2-3':.8,'3-2-1':.2};q={'1-2-3':.3,'3-2-1':.7}
        records=[('1-2-3',{'z:p':p,'m:q':q}),('3-2-1',{'z:p':p,'m:q':q})]
        duplicates=[(y,{**e,'a:copy':e['z:p'],'b:copy':e['z:p']}) for y,e in records]
        original=eq.fit_pool(records);duplicated=eq.fit_pool(duplicates)
        self.assertEqual(len(original['weights']),len(duplicated['weights']))
        a=eq.mix(original,records[0][1]);b=eq.mix(duplicated,duplicates[0][1])
        for k,v in a.items():self.assertAlmostEqual(v,b[k])
        self.assertAlmostEqual(sum(duplicated['weights'].values()),1)

    def test_probability_components_survive_missing_market_without_invented_market(self):
        packet=small_packet();packet['source']['quote_quality']['complete_single_snapshot']=False
        packet['source']['evidence']['usable_quotes'].pop('1-2-3')
        components=eq.static_experts(packet)
        self.assertNotIn('market:only',components)
        self.assertIn('original:model_positions',components)
        self.assertIn('positions:line_department',components)

    def test_three_chronological_phases_fit_and_temperature_replay(self):
        samples=[small_packet(f'2026-10-0{i+1}',str(i)) for i in range(6)]
        with patch.dict(eq.CONFIG,minimum_phase_races=2,minimum_phase_days=2,iterations=4,pool_iterations=5):
            trained=eq.train(samples)
        self.assertIsNotNone(trained['model'])
        phases=trained['phase_counts']
        self.assertLess(phases[0]['last'],phases[1]['first'])
        self.assertLess(phases[1]['last'],phases[2]['first'])
        forecast=eq.infer(trained['model'],samples[0])
        self.assertAlmostEqual(sum(forecast['distribution'].values()),1)
        self.assertTrue(all(v>0 for v in forecast['distribution'].values()))
        self.assertEqual(forecast,eq.infer(json.loads(json.dumps(trained['model'])),samples[0]))
        self.assertIn('paths',forecast['missing_families'])

    def test_insufficient_separate_periods_never_invents_fusion(self):
        trained=eq.train([small_packet()])
        self.assertIsNone(trained['model'])
        forecast=eq.infer(None,small_packet())
        self.assertIsNone(forecast['distribution'])
        self.assertFalse(forecast['effective_weights'])

    def test_conditional_process_responds_to_current_ability_and_line_context(self):
        p=small_packet();q=copy.deepcopy(p)
        q['source']['evidence']['inputs']['risk_department'][0]['score']=10
        dimension=len(eq.joint_features(p,(1,2,3)));beta=[0.]*dimension;beta[2]=4.
        base={'process_groups':{'4':{'beta':[beta,*[beta+[.1]*9 for _ in range(3)]]}}}
        a=eq.predict_process(base,p);b=eq.predict_process(base,q)
        self.assertAlmostEqual(sum(a.values()),1)
        self.assertNotEqual(a,b)
        self.assertIsNone(eq.predict_process({'process_groups':{}},p))

    def test_probability_errors_decompose_exactly(self):
        scores=eq.probability_scores(eq.market(small_packet()),'1-2-3')
        self.assertAlmostEqual(scores['nll'],sum(scores[k] for k in ('first_nll','second_given_first_nll','third_given_pair_nll')))

    def test_process_fits_only_real_annotations_in_base_period(self):
        samples=[small_packet(f'2026-10-0{i+1}',str(i)) for i in range(3)]
        paths=[{'race_id':str(i),'orders':{'start':[3,2,1],'bell':[2,3,1],
                'back':[2,1,3],'finish':[1,2,3]}} for i in range(3)]
        with patch.dict(eq.CONFIG,minimum_phase_races=1,minimum_phase_days=1,iterations=2,pool_iterations=2):
            trained=eq.train(samples,paths)
        group=trained['model']['base']['process_groups']['4']
        self.assertEqual(group['observed_races'],1)
        self.assertAlmostEqual(sum(eq.infer(trained['model'],samples[0])['components']['paths:conditional_process'].values()),1)


class FusionLedgerTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.DepartmentV2Tests();self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.out=self.fixture.out;self.folder=self.out/'company';self.now=self.fixture.now
        self.source=self.fixture.record();append(self.folder/lab.SOURCES,self.source)
        self.packet={'source':self.source,'auxiliary':{'riders':frame_records(self.fixture.race),
            'departments':{d:{'top3_cars':[1,2,3],'snapshot_at':self.now.isoformat(),
                'availability':'observed_current_cycle','source':'fixture'} for d in eq.DEPARTMENTS}}}

    def capture(self):return lab.capture_packet(self.packet,self.out,self.now)

    def outcome(self,payout=1000,order='1-2-3',hours=1):
        append(self.folder/lab.OUTCOMES,{'race_id':'fixture','observed_at':(self.now+timedelta(hours=hours)).isoformat(),
            'outcome':{'winning_buys':[order],'payouts':{order:payout},'payout_complete':True}})

    def test_capture_keeps_departments_risk_and_purchase_gates_unchanged(self):
        before=(self.folder/lab.SOURCES).read_bytes()
        self.assertEqual(self.capture(),'captured_components_waiting_for_training')
        rows,_=lab.validated_rows(self.folder);self.assertEqual(len(rows),1)
        self.assertEqual(before,(self.folder/lab.SOURCES).read_bytes())
        self.assertEqual(set(rows[0]['packet']['auxiliary']['departments']),set(eq.DEPARTMENTS))
        for group in lab.GROUPS:
            risk=[t for t in self.source['risk_tickets'] if t['group']==group]
            self.assertEqual(rows[0]['views'][group]['risk'],risk or None)
        self.assertFalse(rows[0]['ceo_integration']);self.assertFalse(rows[0]['purchase_authorized'])

    def test_report_never_backfills_past_inputs_or_forecasts(self):
        report=lab.build_report(self.out,self.now+timedelta(days=1))
        self.assertEqual(report['input_snapshots'],0)
        self.assertEqual(report['sources_without_fusion_input'],1)
        self.assertFalse((self.folder/lab.INPUTS).exists())
        self.assertFalse((self.folder/lab.LEDGER).exists())
        self.assertEqual(report['confirmation_state'],'not_registered_exploratory_only')

    def test_prior_training_excludes_same_day_and_conflicting_payouts(self):
        self.capture();self.outcome()
        self.assertEqual(lab.prior_packets(self.out,self.now)[0],[])
        self.assertEqual(len(lab.prior_packets(self.out,self.now+timedelta(days=1))[0]),1)
        self.outcome(payout=2000,hours=2)
        samples,_,info=lab.prior_packets(self.out,self.now+timedelta(days=1))
        self.assertEqual(samples,[]);self.assertEqual(info['conflicting_results'],['fixture'])

    def test_rounding_noise_in_same_official_payout_is_not_a_conflict(self):
        self.capture();self.outcome(payout=1739.9999999999998)
        self.outcome(payout=1740,hours=2)
        samples,_,info=lab.prior_packets(self.out,self.now+timedelta(days=1))
        self.assertEqual(len(samples),1)
        self.assertEqual(info['conflicting_results'],[])

    def test_daily_model_frozen_and_code_revision_separates(self):
        a=lab.prepare_model(self.out,self.now)
        with patch('fusion_lab.eq.train',side_effect=AssertionError('must not refit')):
            self.assertEqual(a,lab.prepare_model(self.out,self.now+timedelta(minutes=1)))
        with patch('fusion_lab.code_id',return_value='new-procedure'):
            b=lab.prepare_model(self.out,self.now+timedelta(minutes=1))
        self.assertNotEqual(a['record_sha256'],b['record_sha256'])

    def test_expiry_retains_input_but_does_not_backdate_forecast(self):
        frozen=lab.prepare_model(self.out,self.now)
        with patch('fusion_lab.prepare_model',return_value=frozen),patch('fusion_lab.time.monotonic',side_effect=[0,901]):
            self.assertEqual(self.capture(),'expired_during_fusion_calculation')
        self.assertEqual(len(lab.packets(self.folder)),1)
        self.assertFalse((self.folder/lab.LEDGER).exists())

    def test_future_or_stale_opinion_and_result_input_rejected(self):
        self.capture();rows=list(lab.checked(self.folder/lab.INPUTS))
        original=copy.deepcopy(rows[0])
        for change in ('future','result'):
            row=copy.deepcopy(original)
            if change=='future':row['auxiliary']['departments']['data_department']['snapshot_at']=(self.now+timedelta(seconds=1)).isoformat()
            else:row['auxiliary']['riders'][0]['finish_pos']=1
            (self.folder/lab.INPUTS).write_text(json.dumps(seal(row))+'\n',encoding='utf-8')
            with self.assertRaises(ValueError):lab.packets(self.folder)

    def test_resealed_fusion_or_risk_tampering_rejected(self):
        self.capture();row=list(lab.checked(self.folder/lab.LEDGER))[0]
        row['forecast']['distribution']={'1-2-3':1.}
        (self.folder/lab.LEDGER).write_text(json.dumps(seal(row))+'\n',encoding='utf-8')
        with self.assertRaises(ValueError):lab.validated_rows(self.folder)

    def test_probability_comparison_includes_risk_abstentions(self):
        self.fixture.market['odds_used']=1.
        self.source=self.fixture.record()
        self.assertEqual(self.source['risk_tickets'],[])
        (self.folder/lab.SOURCES).write_text(json.dumps(self.source)+'\n',encoding='utf-8')
        self.packet['source']=self.source;self.capture();self.outcome()
        report=lab.build_report(self.out,self.now+timedelta(days=1))
        self.assertFalse(report['comparisons'])
        self.assertTrue(report['probability_comparisons'])
        self.assertEqual(report['probability_comparisons'][0]['challenger']['races'],1)

    def test_candidate_shortage_no_padding_equal_group_prices_and_stakes(self):
        source={'evidence':{'usable_quotes':{'1-2-3':100.,'2-1-3':99.}}}
        p={'1-2-3':.9,'2-1-3':.1}
        self.assertIsNone(lab.choose(p,source,'穴',2)[0])
        self.assertIsNone(lab.choose(p,source,'穴',13)[0])
        self.assertEqual(lab.choose(p,source,'穴',1)[0][0]['stake_yen'],100)
        self.assertEqual(lab.choose(p,source,'本線',1)[0][0]['buy'],'2-1-3')
        self.assertIsNone(lab.choose({'1-2-3':.0001},source,'穴',1)[0])

    def test_latest_snapshot_never_falls_back_to_older_complete_market(self):
        self.capture()
        later=self.now+timedelta(seconds=60)
        self.fixture.now=later
        self.fixture.market=self.fixture.market.iloc[1:].copy()
        self.fixture.market['odds_captured_at_jst']=later.isoformat()
        from department_experiment_v2 import make_record
        source,reason=make_record(self.fixture.race,self.fixture.inputs,self.fixture.market,later,
            self.fixture.knowledge,self.fixture.provenance)
        self.assertEqual(reason,'partial_market_recorded');append(self.folder/lab.SOURCES,source)
        packet=copy.deepcopy(self.packet);packet['source']=source
        for d in packet['auxiliary']['departments'].values():d['snapshot_at']=later.isoformat()
        lab.capture_packet(packet,self.out,later);self.outcome()
        report=lab.build_report(self.out,self.now+timedelta(days=1))
        self.assertEqual(report['forecast_snapshots'],2)
        self.assertEqual(report['latest_race_cohort_forecasts'],1)
        self.assertFalse(any(r['method']=='market' or r['baseline']=='market' for r in report['probability_comparisons']))

    def test_latest_failed_capture_excludes_old_forecast_for_every_method(self):
        self.capture();self.outcome()
        self.fixture.now=self.now+timedelta(seconds=60)
        self.fixture.market['odds_captured_at_jst']=self.fixture.now.isoformat()
        append(self.folder/lab.SOURCES,self.fixture.record())
        report=lab.build_report(self.out,self.now+timedelta(days=1))
        self.assertEqual(report['forecast_snapshots'],1)
        self.assertEqual(report['excluded_older_forecasts_after_missing_latest_capture'],1)
        self.assertFalse(report['comparisons']);self.assertFalse(report['probability_comparisons'])

    def test_all_fusion_records_survive_concurrent_reset_without_duplicates(self):
        from preserve_validation import save,restore
        from reconcile_position_experiment import LEDGERS
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'repo';backup=Path(directory)/'backup'
            for name in lab.FILES:
                self.assertIn(name,LEDGERS)
                append(root/'outputs/company'/name,{'record':'local'})
            save(backup,root)
            for name in lab.FILES:
                (root/'outputs/company'/name).write_text('')
                append(root/'outputs/company'/name,{'record':'remote'})
            restore(backup,root);restore(backup,root)
            for name in lab.FILES:
                self.assertEqual({r['record'] for r in lab.checked(root/'outputs/company'/name)},{'local','remote'})

    def test_daily_training_capture_and_manifest_replay_end_to_end(self):
        with patch.dict(eq.CONFIG,minimum_phase_races=1,minimum_phase_days=1,iterations=2,pool_iterations=2):
            for day in range(4):
                now=self.now+timedelta(days=day);rid=f'prospective-{day}'
                f=self.fixture;f.now=now
                for frame in (f.race,*f.inputs.values()):
                    frame['race_id']=rid;frame['date']=now.date().isoformat()
                    frame['close_at']=(now+timedelta(minutes=20)).timestamp()
                f.market['race_id']=rid;f.market['odds_captured_at_jst']=now.isoformat()
                f.knowledge['window_end_exclusive']=now.date().isoformat()
                source=f.record();append(self.folder/lab.SOURCES,source)
                packet=copy.deepcopy(self.packet);packet['source']=source
                packet['auxiliary']['riders']=frame_records(f.race)
                for d in packet['auxiliary']['departments'].values():d['snapshot_at']=now.isoformat()
                reason=lab.capture_packet(packet,self.out,now)
                append(self.folder/lab.OUTCOMES,{'race_id':rid,'observed_at':(now+timedelta(hours=1)).isoformat(),
                    'outcome':{'winning_buys':['1-2-3'],'payouts':{'1-2-3':1000},'payout_complete':True}})
            self.assertEqual(reason,'captured_fusion')
            rows,bundles=lab.validated_rows(self.folder)
            final=next(r for r in rows if r['race_id']=='prospective-3')
            self.assertIsNotNone(final['forecast']['distribution'])
            trained=bundles[final['model_record']]
            self.assertEqual([r['race_id'] for r in trained['training_manifest']],['prospective-0','prospective-1','prospective-2'])
            report=lab.build_report(self.out,now+timedelta(days=1))
            self.assertTrue(any(r['method']=='fusion' for r in report['probability_comparisons']))
            records=list(lab.checked(self.folder/lab.MODELS))
            records[-1]['training_manifest'][0]['date']='2099-01-01'
            (self.folder/lab.MODELS).write_text(''.join(json.dumps(seal(r))+'\n' for r in records),encoding='utf-8')
            with self.assertRaises(ValueError):lab.validated_rows(self.folder)

    def test_cycle_uses_clock_at_each_race_and_rejects_old_opinions(self):
        report={'predictions':[{'race_id':'fixture','department':d,'forecast_available':True,
            'snapshot_at':(self.now-timedelta(seconds=60)).isoformat(),'top3_cars':[1,2,3]} for d in eq.DEPARTMENTS]}
        with patch('fusion_lab.clock',return_value=self.now),patch('fusion_lab.capture_packet',return_value='test') as capture:
            lab.capture_cycle(self.fixture.race,report,self.now,self.out)
        self.assertTrue(capture.called)
        aux=capture.call_args.args[0]['auxiliary']
        self.assertTrue(all(not v['top3_cars'] for v in aux['departments'].values()))

    def test_confirmation_requires_fit_future_midnight_and_positive_target(self):
        start=(self.now+timedelta(days=1)).replace(hour=0)
        with self.assertRaises(ValueError):lab.register_confirmation(self.out,start,.02,self.now)
        append(self.folder/lab.MODELS,{'code':lab.code_id(),'training':{'model':{'fixture':True}}})
        for bad_start,target in ((self.now,.02),(start,0),(start+timedelta(hours=1),.02)):
            with self.assertRaises(ValueError):lab.register_confirmation(self.out,bad_start,target,self.now)
        trial=lab.register_confirmation(self.out,start,.02,self.now)
        self.assertEqual(trial['family_alpha'],.025)
        self.assertFalse(trial['automatic_promotion'])
        with self.assertRaises(ValueError):lab.register_confirmation(self.out,start,.02,self.now)
        result=lab.confirmation([],trial,'穴',start+timedelta(days=57))
        self.assertEqual(result['status'],'insufficient_at_fixed_deadline')


if __name__=='__main__':unittest.main()
