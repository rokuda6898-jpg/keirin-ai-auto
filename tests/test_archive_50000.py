import unittest
import tempfile
from datetime import date
from pathlib import Path
from itertools import permutations

import pandas as pd
from archive_50000 import (VERSION, FEATURE_NAMES, probabilities, body_digest,
                           validated_model, rider_vectors)
from archive_50000_train import eligible_races, chronological_split, design


def riders(winner=1):
    out=[]
    for c in range(1,6):
        out.append({'race_id':'fixture','date':'2026-01-01','car_no':c,
            'player_id':str(c),'score':101-c,'win_rate':.2,
            'place2_rate':.4,'place3_rate':.5,'back_count':10-c,
            'recent_avg_finish':c,'entries_number':5,
            'finish_pos':winner if False else {winner:1, (winner%5)+1:2,
                ((winner+1)%5)+1:3}.get(c,float('nan'))})
    return out


class ArchiveFiftyThousandTests(unittest.TestCase):
    def test_methods_are_independent_and_valid(self):
        rows=riders()
        base={'position_betas':[[0,.5,1,.2,0,0,0,0,0],
                                [0,.1,.5,1,0,0,0,0,0],
                                [1,.1,.2,.5,0,0,0,0,0]],
              'position_intercepts':[0,0,0],
              'pairwise_beta':[1,0,1,0,0,0,0,0,0]}
        a=probabilities('archive_positions',base,rows)
        b=probabilities('archive_pairwise',base,rows)
        self.assertEqual(len(a),5*4*3)
        self.assertAlmostEqual(sum(a.values()),1.0)
        self.assertAlmostEqual(sum(b.values()),1.0)
        self.assertNotEqual(a,b)
        for k in a:self.assertEqual(len(set(k.split('-'))),3)

    def test_training_design_does_not_use_unplaced_order(self):
        first=riders(winner=2)
        second=riders(winner=4)
        for row in second:
            row['race_id']='next';row['date']='2026-01-02'
        f=pd.DataFrame(first+second)
        rows,valid=eligible_races(f,'2026-02-01')
        self.assertEqual(len(valid),2)
        x,y,w,px,py,pw=design(rows,['fixture','next'])
        self.assertEqual(len(x),10)
        self.assertEqual([sum(a) for a in y],[2,2,2])
        self.assertEqual(len(px),2*9)
        self.assertEqual(len(set(py)),2)

    def test_no_future_day_or_missing_top_three(self):
        f=pd.DataFrame(riders()+[{**r,'race_id':'next','date':'2026-02-01'} for r in riders()])
        rows,valid=eligible_races(f,'2026-02-01')
        self.assertEqual(len(valid),1)
        self.assertEqual(len(rows),5)
        f.loc[f.race_id.eq('fixture') & f.car_no.eq(2),'finish_pos']=float('nan')
        _,valid=eligible_races(f,'2026-02-01')
        self.assertEqual(len(valid),0)

    def test_chronological_is_strictly_later(self):
        count=51010
        v=pd.DataFrame({'race_id':[str(i) for i in range(count)],
            'day':['2026-01-01' if i<50020 else '2026-01-02' for i in range(count)]})
        with self.assertRaisesRegex(ValueError,'not enough later'):
            chronological_split(v)
        v['day']=['2026-01-01' if i<50000 else '2026-01-02' for i in range(count)]
        train,heldout,train_end,holdout_start=chronological_split(v)
        self.assertEqual(len(train),50000)
        self.assertEqual(len(heldout),1010)
        self.assertLess(train_end,holdout_start)

    def test_frozen_model_requires_fifty_thousand_and_past_cutoff(self):
        model={'version':VERSION,'trained_races':50000,'feature_names':list(FEATURE_NAMES),
          'training_end':'2026-10-08','train_holdout_overlap':False,
          'pairwise_beta':[0.1]*len(FEATURE_NAMES),
          'position_betas':[[0.2]*len(FEATURE_NAMES) for _ in range(3)],
          'position_intercepts':[0.,0.,0.]}
        model['model_sha256']=body_digest(model)
        self.assertTrue(validated_model(model,'2026-10-09'))
        self.assertFalse(validated_model(model,'2026-10-08'))
        model['trained_races']=49999
        self.assertFalse(validated_model(model,'2026-10-09'))


if __name__=='__main__':
    unittest.main()
