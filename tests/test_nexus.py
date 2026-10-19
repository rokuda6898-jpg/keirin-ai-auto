import sys,unittest,itertools,json,hashlib
from pathlib import Path
from fractions import Fraction
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from nexus_model.budget import allocate,settle
from nexus_model.source import outcomes
from nexus_model import optimizer,pair,inputs
import numpy as np

class Contract(unittest.TestCase):
    def test_allocation_and_low_odds_order(self):
        for n in [6,12,15,24]:
            tickets=['-'.join(map(str,z)) for z in list(itertools.permutations(range(1,8),3))[:n]];odds={t:Fraction(21+37*j,10) for j,t in enumerate(tickets)}
            for b in range(5000,10001,100):
                a=allocate(tickets,odds,b)
                if not a['available']:
                    self.assertEqual(a['reason_code'],'no_break_even');self.assertLess(a['best_possible_return'],b);self.assertNotIn('stakes',a);continue
                v=list(a['stakes'].values());self.assertEqual(sum(v),b);self.assertTrue(all(x>=100 and x%100==0 for x in v));self.assertEqual(v,sorted(v,reverse=True));self.assertTrue(a['quote_break_even']);self.assertGreaterEqual(a['minimum_return'],b)
    def test_no_missing_odds_fabrication(self):
        a=allocate(['1-2-3'],{},5000);self.assertFalse(a['available']);self.assertEqual(a['reason_code'],'missing_odds');self.assertNotIn('stakes',a)
    def test_tie_and_refund_accounting(self):
        x=settle({'1-2-3':200,'2-1-3':100,'4-5-6':100},{'1-2-3':400,'2-1-3':500},['4-5-6']);self.assertTrue(x['hit']);self.assertEqual(x['returned'],1400)
        self.assertTrue(settle({'1-2-3':100},{},refund_all=True)['void'])
    def test_full_formations_no_fixed_pair_requirement(self):
        cars=list(range(1,8));triples=list(itertools.permutations(range(7),3));q=np.array([1. if b==(a+1)%7 and c==(a+2)%7 else .001 for a,b,c in triples]);q/=q.sum()
        for cap in [6,15,24]:
            p=optimizer.select(cars,q,cap)
            if cap==15:p=pair.select(cars,q,cap,p)
            actual={t for b in p['blocks'] for t in itertools.product(b['first'],b['second'],b['third']) if len(set(t))==3}
            self.assertEqual(len(actual),cap);self.assertLessEqual(len(p['blocks']),4);self.assertEqual(actual,set(map(tuple,p['tickets'])));self.assertGreater(len({x[0] for x in actual}),1)
    def test_input_and_model_portability_receipts(self):
        self.assertTrue(json.loads((ROOT/'models/nexus53/portability_check.json').read_text())['passed'])
        self.assertTrue(all(not r['differences'] for r in json.loads((ROOT/'models/nexus53/input_adapter_check.json').read_text())))
        folder=ROOT/'models/nexus53'
        for name,record in json.loads((folder/'manifest.json').read_text(encoding='utf-8'))['assets'].items():
            self.assertEqual(hashlib.sha256((folder/name).read_bytes()).hexdigest(),record['sha'],name)
    def test_today_history_rejected(self):
        h=inputs.History();h.data['through']='2026-10-19'
        with self.assertRaisesRegex(ValueError,'before race day'):inputs.build(h,{},dict(date='2026-10-19'))
    def test_unpublished_result_not_scored(self):
        self.assertFalse(outcomes({'race':{'cancel':False,'decidedAt':0},'trifecta':[]})[3])
        self.assertTrue(outcomes({'race':{'cancel':False,'decidedAt':123},'trifectaPayoffStatus':3,'trifecta':[]})[2])
        self.assertFalse(outcomes({'race':{'cancel':False,'decidedAt':0},'trifectaPayoffStatus':3,'trifecta':[]})[2])

if __name__=='__main__':unittest.main()

