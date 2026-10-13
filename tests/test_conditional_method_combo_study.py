import unittest
from itertools import permutations
from conditional_method_combo_study import score_records
from conditional_method_combo import VARIANTS

class ReplayScoringTests(unittest.TestCase):
    def test_baseline_reproduction_is_mandatory(self):
        keys=['-'.join(map(str,t)) for t in permutations((1,2,3,4),3)]
        p=[1/len(keys)]*len(keys)
        row={'race_id':'x','date':'2026-05-01','race_class':'A級','field':4,'keys':keys,
             'line_verified':False,'line_structured_unverified':False,
             'probabilities':{v:p for v in VARIANTS}}
        with self.assertRaisesRegex(ValueError,'BASELINE NOT REPRODUCED'):
            score_records([row],{'x':keys[0]},bootstrap=20)

if __name__=='__main__':unittest.main()
