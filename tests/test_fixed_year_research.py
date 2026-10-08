import gzip
import json
import tempfile
import unittest
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

import fixed_year_features as f
import fixed_year_models as m
import fixed_year_study as study
import fusion_equations as fusion
import equation_models as legacy
from fixed_year_research import boundaries,split_history,inventory
from betting_logic import score_riders
from department_coverage import position_scenario,strategist_scenario


def fixture(days=1,start='2024-01-01',n=7):
    rows=[]
    for d,date in enumerate(pd.date_range(start,periods=days)):
        for car in range(1,n+1):
            rows.append({'race_id':date.strftime('%Y%m%d')+'01','date':str(date.date()),
                'player_id':str(car),'car_no':car,'venue':'fixture','race_no':1,'entries_number':n,
                'score':90+car,'recent_avg_finish':car,'race_class':'fixture','style':'追',
                'finish_pos':(car+d-1)%n+1,'line_id':(car-1)//3,'line_position':(car-1)%3+1,
                'line_verification_status':'verified','line_size':3,'number_of_lines':3,
                'p_win':car/sum(range(1,n+1)),'p_second':(n+1-car)/sum(range(1,n+1)),
                'p_third':1/n,'place2_rate':car/50,'place3_rate':car/40})
    return pd.DataFrame(rows)


class FixedYearTests(unittest.TestCase):
    def test_exact_year_and_leap_boundary(self):
        a,b=boundaries('2026-10-09')
        self.assertEqual(str(a.date()),'2025-10-09')
        data=pd.DataFrame({'date':['2025-10-08','2025-10-09','2026-10-08','2026-10-09',None]})
        old,target=split_history(data,'2026-10-09')
        self.assertEqual(list(old.index),[0]);self.assertEqual(list(target.index),[1,2])
        self.assertEqual(str(boundaries('2024-02-29')[0].date()),'2023-02-28')

    def test_four_whole_day_phases_partition_only_older_data(self):
        data=fixture(70,n=3);parts=study.phase_split(data)
        self.assertEqual(sum(len(x) for x in parts),len(data))
        for a,b in zip(parts,parts[1:]):self.assertLess(max(a.date),min(b.date))

    def test_forbidden_outcomes_removed_and_dnf_not_top_three(self):
        data=fixture(3,n=3);data['official_finish_pos']=data.finish_pos;data['result_event_back']=True
        data.loc[0,'finish_pos']=0
        enriched,ref=f.daily_priors(data)
        self.assertTrue(enriched.finish_pos.isna().all())
        self.assertNotIn('official_finish_pos',enriched);self.assertNotIn('result_event_back',enriched)
        self.assertEqual(ref['1']['n'],2)
        first_day=enriched[enriched.date.eq('2024-01-01')]
        self.assertTrue(first_day.player_prior_races.eq(0).all())

    def test_frozen_reference_ignores_all_target_answers(self):
        old=fixture(5,n=3);target=fixture(4,start='2026-01-01',n=3)
        _,reference=f.daily_priors(old)
        expected=f.frozen_priors(target,reference)
        target['finish_pos']=1;target['official_finish_pos']=9;target['result_available']=True
        target['result_event_back']=False;target['payout']=999999
        pd.testing.assert_frame_equal(expected,f.frozen_priors(target,reference))
        self.assertTrue(expected.player_prior_races.eq(5).all())

    def test_inference_imputation_and_line_missing_flag_are_frozen(self):
        old=fixture(4,n=3);old,_=f.daily_priors(old);X,config=f.matrix(old)
        future=fixture(2,start='2026-01-01',n=3);future['score']=np.nan;future['line_verification_status']='missing'
        future,_=f.daily_priors(future);a,_=f.matrix(future,config)
        self.assertTrue(a.line_information_missing.eq(1).all())
        future['finish_pos']=1;future['result_event_back']=True
        b,_=f.matrix(future,config);pd.testing.assert_frame_equal(a,b)
        self.assertEqual(config['strength_medians']['score'],92)

    def test_current_risk_score_arithmetic_matches_production(self):
        rng=np.random.default_rng(9021)
        for n in (3,7,9):
            race=fixture(n=n)
            for name in ('player_prior_win_rate','player_prior_place2_rate','player_prior_place3_rate',
                         'track_place2_rate','weather_place3_rate'):
                race[name]=rng.random(n);race.loc[0,name]=np.nan
            expected=score_riders(race)
            actual=m.legacy_position_weights(race.where(pd.notna(race),None).to_dict('records'))
            for name in ('first','second','third'):
                np.testing.assert_allclose([r[name] for r in actual],expected['score_'+name],rtol=1e-13)

    def test_seven_reconstructed_scenarios_match_existing_rules(self):
        for n in (3,7,9):
            race=f.mask_outcomes(fixture(n=n));packet=m.make_packet(race,{})
            scenarios=packet['auxiliary']['departments']
            self.assertEqual(len(scenarios),7)
            self.assertEqual(scenarios['prediction_department']['top3_cars'],position_scenario(race))
            self.assertEqual(scenarios['high_payout_department']['top3_cars'],position_scenario(race,True))
            opinions=[{'forecast_available':True,'top3_cars':scenarios[d]['top3_cars']} for d in fusion.DEPARTMENTS[:4]]
            self.assertEqual(scenarios['strategist_department']['top3_cars'],strategist_scenario(race,opinions)[0])
        with self.assertRaises(ValueError):m.make_packet(fixture(),{})

    def test_joint_vectorized_gradient_is_same_equation(self):
        rng=np.random.default_rng(891)
        designs=[rng.normal(size=(n,7)) for n in (6,24,60)];targets=[0,3,40]
        expected=fusion.fit_linear(designs,targets)
        actual=m.fit_joint_matrix(np.concatenate(designs),[len(x) for x in designs],targets)
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)

    def test_price_independent_pairwise_matches_old_equation(self):
        rows=fixture().to_dict('records');beta=[.1,-.2,.3,.4,.1,-.1,.2,.0]
        quotes={'-'.join(map(str,t)):100. for t in permutations(range(1,8),3)}
        expected=legacy.predict('pairwise_order',{'beta':beta},rows,quotes)
        actual=m.pairwise_distribution(beta,rows)
        self.assertEqual(set(actual),set(expected))
        np.testing.assert_allclose(list(actual.values()),[expected[k] for k in actual],rtol=1e-12)

    def test_exclusions_do_not_use_target_labels(self):
        data=fixture(n=3);data['finish_pos']=1
        self.assertIsNone(study.input_reason(data));self.assertIsNone(study.outcome(data))
        data.loc[0,'car_no']=2;self.assertIsNotNone(study.input_reason(data))

    def test_training_rejects_holdout_rows_before_any_fit(self):
        with self.assertRaisesRegex(ValueError,'held-out'):
            study.train(fixture(start='2026-01-01'),Path('.'),'2025-10-09',fast=True)

    def test_complete_separate_train_predict_score_flow_and_label_invariance(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);old=fixture(50,n=3);target=fixture(3,start='2026-01-01',n=3)
            raw=folder/'fixture.csv';all_rows=pd.concat([old,target],ignore_index=True);all_rows.to_csv(raw,index=False)
            study.write_json(folder/'inventory.json',inventory(all_rows,raw,'2026-10-09'))
            study.train(old,folder,'2025-10-09',fast=True)
            study.predict(target,folder)
            path=folder/'holdout_predictions.jsonl.gz'
            with gzip.open(path,'rt',encoding='utf-8') as h:first=h.read()
            changed=target.copy();changed['finish_pos']=4-changed.finish_pos;changed['result_event_back']=True
            study.predict(changed,folder)
            with gzip.open(path,'rt',encoding='utf-8') as h:self.assertEqual(first,h.read())
            result=study.assess(target,folder)
            self.assertEqual(result['evaluated_races'],3)
            self.assertIsNone(result['roi']);self.assertFalse(result['ceo_integration'])
            self.assertTrue((folder/'fixed_year_report.html').is_file())
            for row in map(json.loads,first.splitlines()):
                self.assertNotIn('actual',row);self.assertNotIn('finish_pos',row)
            model=folder/'frozen_model.joblib'
            with model.open('ab') as handle:handle.write(b'changed')
            with self.assertRaisesRegex(ValueError,'modified'):study.predict(target,folder)


if __name__=='__main__':unittest.main()
