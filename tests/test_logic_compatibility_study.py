import unittest
import numpy as np

from logic_compatibility_study import combine,OPTIONS


class CompatibilityStudyTests(unittest.TestCase):
    def setUp(self):
        self.keys=sorted(f'{a}-{b}-{c}' for a in (1,2,3,4) for b in (1,2,3,4) for c in (1,2,3,4) if a!=b and a!=c and b!=c)
        x=np.arange(1,len(self.keys)+1,dtype=float)
        self.anchor=x/x.sum()
        y=x[::-1].copy();self.base=y/y.sum()
        self.alts={name:self.base.copy() for name in OPTIONS}
        factors={pair:0.8+0.05*i for i,pair in enumerate(sorted({key.rsplit('-',1)[0] for key in self.keys}))}
        self.alts['archive_geometry_030']=np.array([self.base[i]*factors[key.rsplit('-',1)[0]] for i,key in enumerate(self.keys)])
        self.alts['archive_geometry_030']/=self.alts['archive_geometry_030'].sum()

    def test_identity_ratio_matches_anchor(self):
        p=combine(self.keys,self.anchor,self.base,self.alts)
        for name in ('rider_035','archive_geometry_010'):
            np.testing.assert_allclose(p['anchor_x_'+name],self.anchor)
            np.testing.assert_allclose(p['first_fixed_x_'+name],self.anchor)

    def test_first_marginals_are_held_constant(self):
        p=combine(self.keys,self.anchor,self.base,self.alts)['first_fixed_x_archive_geometry_030']
        for car in (1,2,3,4):
            idx=[i for i,k in enumerate(self.keys) if k.startswith(f'{car}-')]
            self.assertAlmostEqual(p[idx].sum(),self.anchor[idx].sum(),places=12)

    def test_third_conditional_is_held_constant(self):
        p=combine(self.keys,self.anchor,self.base,self.alts)['first_fixed_x_archive_geometry_030']
        for pair in (f'{a}-{b}' for a in (1,2,3,4) for b in (1,2,3,4) if a!=b):
            idx=[i for i,k in enumerate(self.keys) if k.startswith(f'{pair}-')]
            np.testing.assert_allclose(p[idx]/p[idx].sum(),self.anchor[idx]/self.anchor[idx].sum())

    def test_distributions_strictly_positive_and_normalized(self):
        p=combine(self.keys,self.anchor,self.base,self.alts)
        for value in p.values():
            self.assertTrue((value>0).all());self.assertAlmostEqual(value.sum(),1.,places=12)

    def test_missing_candidate_is_not_tolerated(self):
        with self.assertRaises((ValueError,KeyError)):
            combine(self.keys,self.anchor,self.base,{'rider_035':self.base})

    def test_duplicate_key_rejected(self):
        keys=self.keys.copy();keys[2]=keys[1]
        with self.assertRaises(ValueError):combine(keys,self.anchor,self.base,self.alts)

    def test_zero_proba_rejected(self):
        q=self.base.copy();q[0]=0.;q/=q.sum()
        with self.assertRaises(ValueError):combine(self.keys,self.anchor,q,self.alts)

    def test_unordered_keys_rejected(self):
        keys=self.keys[::-1]
        with self.assertRaises(ValueError):combine(keys,self.anchor,self.base,self.alts)

    def test_third_specific_scenario_rejected(self):
        altered={name:value.copy() for name,value in self.alts.items()}
        altered['rider_035']=self.base*np.linspace(0.8,1.2,len(self.keys))
        altered['rider_035']/=altered['rider_035'].sum()
        with self.assertRaises(ValueError):combine(self.keys,self.anchor,self.base,altered)

    def test_mismatched_field_rejected(self):
        keys=self.keys[:18]
        with self.assertRaises(ValueError):combine(keys,self.anchor[:18],self.base[:18],self.alts)


if __name__=='__main__':unittest.main()
