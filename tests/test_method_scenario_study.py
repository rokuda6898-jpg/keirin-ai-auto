import csv
import gzip
import hashlib
import json
import tempfile
import unittest
from datetime import date, timedelta
from itertools import permutations
from pathlib import Path
from types import SimpleNamespace

from method_scenario_study import assess, predict, read


class ScenarioStudyReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)
        self.history=root/'history.csv';self.methods=root/'methods.csv'
        self.predictions=root/'original.jsonl.gz';self.manifest=root/'original_manifest.json'
        self.out=root/'study'
        rows=[]
        for i in range(160):
            rows.append({'race_id':f'h{i}','date':str(date(2024,1,1)+timedelta(days=i)),
              'method_status':'confirmed_by_podium_ids','winner_method':('逃','捲','差')[i%3],
              'second_method':('逃','差','マ','捲')[i%4],
              'winner_player_id':str(100+i%7),'second_player_id':str(100+(i+2)%7),
              'division':'A_regular','venue':'奈良','field':7})
        with self.methods.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        keys=['-'.join(map(str,t)) for t in permutations(range(1,8),3)]
        races=[];entries=[]
        for i in range(105):
            rid=f'test{i}';d=str(date(2025,10,9)+timedelta(days=i%80))
            races.append({'race_id':rid,'date':d,'field':7,'race_class':'A級','keys':keys,
                          'distributions':{'fusion':[1/len(keys)]*len(keys)}})
            for car in range(1,8):
                entries.append({'race_id':rid,'date':d,'venue':'奈良',
                    'race_class':'A級','race_type':'予選','is_grade_race':0,
                    'car_no':car,'player_id':str(100+car),'entries_number':7,
                    'line_id':1 if car<4 else 2 if car<6 else 3,
                    'line_position':car if car<4 else car-3 if car<6 else car-5,
                    'line_verification_status':'verified','finish_pos':((car+i)%7)+1})
        with self.history.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(entries[0]));w.writeheader();w.writerows(entries)
        with gzip.open(self.predictions,'wt',encoding='utf-8') as f:
            for r in races:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        digest=hashlib.sha256(self.predictions.read_bytes()).hexdigest()
        self.manifest.write_text(json.dumps({'predictions_sha256':digest,'first':'2025-10-09',
                            'outcomes_accessed':False,'retrospective_only':True}))
        self.args=SimpleNamespace(archive=str(self.methods),history=str(self.history),
               predictions=str(self.predictions),baseline_manifest=str(self.manifest),
               out_dir=str(self.out),cutoff='2025-10-09')

    def test_frozen_phase_does_not_load_outcomes(self):
        predict(self.args)
        manifest=read(self.out/'scenario_prediction_manifest.json')
        self.assertEqual(manifest['predicted_races'],105)
        self.assertFalse(manifest['results_loaded_during_prediction'])
        with gzip.open(self.out/'scenario_predictions.jsonl.gz','rt',encoding='utf-8') as f:
            sample=json.loads(next(f))
        self.assertNotIn('finish_pos',sample)
        self.assertNotIn('actual_trifecta',sample)
        self.assertAlmostEqual(sum(sample['probabilities']['rider_line_035']),1.)
        assess(self.args)
        output=read(self.out/'scenario_results.json')
        self.assertEqual(output['assessment_races'],105)
        self.assertIn('top8_hit_rate',output['methods']['rider_line_035'])
        self.assertIsNone(output['return_rate'])

    def test_assessment_rejects_altered_predictions(self):
        predict(self.args)
        p=self.out/'scenario_predictions.jsonl.gz'
        with p.open('ab') as f:f.write(b'tamper')
        with self.assertRaisesRegex(ValueError,'changed'):
            assess(self.args)

    def test_baseline_provenance_is_mandatory(self):
        data=read(self.manifest);data['outcomes_accessed']=True
        self.manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'outcome-free'):
            predict(self.args)

if __name__=='__main__':unittest.main()
