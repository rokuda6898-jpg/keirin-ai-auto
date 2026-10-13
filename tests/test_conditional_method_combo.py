import unittest
from itertools import permutations
from collections import defaultdict
import numpy as np
from conditional_method_combo import validated,pair_adjust,rerank_all,rank,select_value_candidates,VARIANTS

class ConditionalComboTests(unittest.TestCase):
    def setUp(self):
        self.keys=['-'.join(map(str,x)) for x in permutations(range(1,8),3)]
        self.triples=list(permutations(range(1,8),3))
        self.p=np.arange(1,len(self.keys)+1,dtype=float)
        self.p/=self.p.sum()
        self.pure={(a,b):1.+(a+2*b)/100 for a,b in permutations(range(1,8),2)}
        self.verified={(a,b):v*1.01 for (a,b),v in self.pure.items()}
        self.arch10={(a,b):v*1.02 for (a,b),v in self.pure.items()}
        self.arch30={(a,b):v*1.04 for (a,b),v in self.pure.items()}

    def test_all_variants_complete_finite(self):
        d,_=rerank_all(self.p,self.keys,self.pure,self.verified,self.arch10,self.arch30)
        self.assertEqual(tuple(d),VARIANTS)
        self.assertTrue(all(len(p)==len(self.p) and np.all(p>0) and np.isclose(p.sum(),1) for p in d.values()))
        self.assertTrue(np.array_equal(d['conditional'],self.p))

    def test_preserves_first_marginal_and_third_conditional(self):
        base=self.p.copy()
        adjusted=pair_adjust(base,self.triples,self.pure,.70,preserve_first=True)
        for car in range(1,8):
            inds=[i for i,x in enumerate(self.triples) if x[0]==car]
            self.assertAlmostEqual(base[inds].sum(),adjusted[inds].sum())
        for a,b in permutations(range(1,8),2):
            inds=[i for i,t in enumerate(self.triples) if t[:2]==(a,b)]
            self.assertTrue(np.allclose(base[inds]/base[inds].sum(),adjusted[inds]/adjusted[inds].sum()))

    def test_negative_corrupt_baseline_fails(self):
        p=self.p.copy();p[1]=-.2
        with self.assertRaises(ValueError):validated(p,self.keys)
        with self.assertRaises(ValueError):validated(self.p[:-1],self.keys)

    def test_rank_measures_3_6_8_10_12_18(self):
        for k in (3,6,8,10,12,18):self.assertEqual(len(rank(self.keys,self.p,k)),k)

    def test_odds_hole_not_cut_before_filter(self):
        odds={k:12. for k in self.keys}
        special=self.keys[19]
        odds[special]=120.
        picks=select_value_candidates(self.keys,self.p,odds,min_odds=50,limit=12)
        self.assertEqual(picks,[special])
        odds.pop(special)
        with self.assertRaises(ValueError):select_value_candidates(self.keys,self.p,odds,min_odds=50,limit=12)

if __name__=='__main__':unittest.main()
