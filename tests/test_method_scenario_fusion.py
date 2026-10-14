import math
import tempfile
import unittest
from datetime import date,timedelta
from itertools import permutations
from pathlib import Path

from method_scenario_fusion import MethodScenarioFusion, _verified_line, _archived_structured_line


def observed():
    rows=[]
    for i in range(180):
        d=date(2024,1,1)+timedelta(days=i)
        rows.append({'race_id':f'r{i}','date':str(d),'method_status':'confirmed_by_podium_ids',
             'winner_method':('逃','捲','差')[i%3],
             'second_method':('マ','差','逃','捲')[i%4],
             'winner_player_id':str(i%7+1), 'second_player_id':str((i+2)%7+1),
             'field':'7','division':'A_regular','venue':'奈良'})
    return rows


def entrants():
    return [dict(car_no=c,player_id=str(c),line_id=1 if c<=3 else 2 if c<=5 else 3,
        line_position=c if c<=3 else c-3 if c<=5 else c-5,
        line_verification_status='verified') for c in range(1,8)]


def baseline():
    keys=['-'.join(map(str,t)) for t in permutations(range(1,8),3)]
    return {k:1/len(keys) for k in keys}


class MethodScenarioTests(unittest.TestCase):
    def setUp(self):
        self.records=observed();self.model=MethodScenarioFusion(self.records,'2025-01-01')
        self.race={'venue':'奈良','race_class':'A級','race_type':'予選','is_grade_race':False,'date':'2025-02-01'}

    def test_normalized_complete_and_distinct_rerank(self):
        p,d=self.model.rerank(baseline(),entrants(),self.race,line_weight=.3)
        self.assertEqual(set(p),set(baseline()));self.assertAlmostEqual(sum(p.values()),1.)
        self.assertTrue(all(0<v<1 for v in p.values()))
        self.assertEqual(d['first_riders_with_history'],7)
        self.assertTrue(d['line_verified'])
        self.assertTrue(any(abs(p[k]-baseline()[k])>1e-8 for k in p))

    def test_no_future_observations_in_fit(self):
        changed=[dict(r) for r in self.records]
        changed.append({**changed[0],'race_id':'future','date':'2025-06-01',
                        'winner_method':'捲','winner_player_id':'9'})
        original=self.model.rerank(baseline(),entrants(),self.race)[0]
        later=MethodScenarioFusion(changed,'2025-01-01').rerank(baseline(),entrants(),self.race)[0]
        self.assertEqual(original,later)
        self.assertEqual(self.model.training_races,180)

    def test_untrusted_outcome_fields_never_used_at_inference(self):
        a=self.model.rerank(baseline(),entrants(),self.race,line_weight=.3)[0]
        unsafe=[dict(x,finish_pos=1,result_factor='逃',winner_method='逃',actual_trifecta='1-2-3') for x in entrants()]
        b=self.model.rerank(baseline(),unsafe,{**self.race,'winner_method':'差'},line_weight=.3)[0]
        self.assertEqual(a,b)

    def test_incomplete_or_unverified_lines_disable_line_signal(self):
        e=entrants();e[0]['line_verification_status']='missing'
        self.assertIsNone(_verified_line(e))
        no_line=self.model.rerank(baseline(),e,self.race,line_weight=.5)
        rider_only=self.model.rerank(baseline(),e,self.race,line_weight=0)
        self.assertFalse(no_line[1]['line_verified'])
        self.assertEqual(no_line[0],rider_only[0])
        e=entrants();e[1]['line_position']=1
        self.assertIsNone(_verified_line(e))

    def test_unknown_riders_and_no_lines_exact_fallback(self):
        rows=[dict(r,player_id=f'new{r["car_no"]}',line_verification_status='unknown') for r in entrants()]
        p,_=self.model.rerank(baseline(),rows,self.race,line_weight=.5)
        for k,value in baseline().items():self.assertAlmostEqual(p[k],value,places=12)

    def test_reject_bad_prediction_provenance_and_cutoff(self):
        with self.assertRaisesRegex(ValueError,'cutoff'):
            self.model.rerank(baseline(),entrants(),{**self.race,'date':'2024-04-01'})
        with self.assertRaisesRegex(ValueError,'baseline'):
            self.model.rerank({'1-2-3':1.},entrants(),self.race)
        with self.assertRaisesRegex(ValueError,'positive'):
            self.model.rerank({**baseline(),'1-2-3':0.},entrants(),self.race)
        with self.assertRaisesRegex(ValueError,'strength'):
            self.model.rerank(baseline(),entrants(),self.race,strength=2.)
        with self.assertRaisesRegex(ValueError,'duplicated'):
            e=entrants();e[1]['player_id']=e[0]['player_id']
            self.model.rerank(baseline(),e,self.race)

    def test_archive_geometry_is_separate_from_provider_verified(self):
        entries=[dict(r, source_url='https://www.winticket.jp/keirin/nara/racecard/2025010153/1/1',
                      line_verification_status=None) for r in entrants()]
        self.assertIsNone(_verified_line(entries))
        self.assertIsNotNone(_archived_structured_line(entries))
        ref,meta=self.model.rerank(baseline(),entries,self.race,line_weight=.5)
        research,diag=self.model.rerank(baseline(),entries,self.race,unverified_line_weight=.2)
        self.assertEqual(ref,self.model.rerank(baseline(),entries,self.race)[0])
        self.assertFalse(diag['line_verified'])
        self.assertTrue(diag['line_structured_unverified'])
        self.assertNotEqual(ref,research)
        for r in entries: r['line_verification_status']='incomplete'
        self.assertIsNone(_archived_structured_line(entries))
        blocked=self.model.rerank(baseline(),entries,self.race,unverified_line_weight=.2)[0]
        self.assertEqual(blocked,ref)

    def test_unverified_geometry_requires_exact_provider_url(self):
        entries=[dict(r, line_verification_status=None, source_url='https://example.com/race') for r in entrants()]
        self.assertIsNone(_archived_structured_line(entries))
        no=self.model.rerank(baseline(),entries,self.race,unverified_line_weight=.5)[0]
        assert no==self.model.rerank(baseline(),entries,self.race)[0]
        entries[1]['source_url']='https://www.winticket.jp/keirin/nara/racecard/2025010153/1/1'
        self.assertIsNone(_archived_structured_line(entries))

    def test_no_train_with_unobserved_or_insufficient_history(self):
        bad=[dict(r,method_status='style_inference') for r in self.records]
        with self.assertRaisesRegex(ValueError,'insufficient'):
            MethodScenarioFusion(bad,'2025-01-01')

    def test_class_and_field_priors_distinct(self):
        a=self.model._prior(dict(venue='奈良',race_class='A級',race_type='予選',field=7))
        b=self.model._prior(dict(venue='岐阜',race_class='S級',race_type='予選',field=9))
        self.assertNotEqual(a,b)
        self.assertAlmostEqual(sum(a.values()),1.)

if __name__=='__main__':unittest.main()
