import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nexus_model.publication import allocation_snapshot, is_open_prediction, settle_prediction


class PublicationContract(unittest.TestCase):
    def test_preclose_auto_budget_and_settlement(self):
        p={'tickets':['1-2-3','2-3-1','3-1-2']}
        p.update(allocation_snapshot(p, {t:3 for t in p['tickets']},
            '2026-10-19T12:00:00+09:00', '2026-10-19T12:01:00+09:00',
            '2026-10-19T12:05:00+09:00'))
        self.assertEqual(p['auto_budget'], 5100)
        self.assertEqual(p['evaluated_budgets'], 51)
        self.assertEqual(p['feasible_budgets'], list(range(5100, 10000, 300)))
        self.assertLessEqual(len(p['allocations']), 7)
        self.assertFalse(p['allocations']['5000']['available'])
        result=settle_prediction(p, {'1-2-3':300})
        self.assertEqual(result['auto']['stake'], 5100)
        self.assertEqual(result['auto']['returned'], 5100)
        self.assertNotIn('5000', result)

    def test_postclose_cannot_create_a_prior_allocation(self):
        for odds_at,saved_at in [('12:05','12:01'),('12:00','12:05')]:
            with self.assertRaises(ValueError):
                allocation_snapshot({'tickets':['1-2-3']}, {'1-2-3':3},
                    f'2026-10-19T{odds_at}:00+09:00',f'2026-10-19T{saved_at}:00+09:00',
                    '2026-10-19T12:05:00+09:00')

    def test_both_positions_must_be_open(self):
        self.assertTrue(is_open_prediction({'tickets':['1-2-3','2-1-3']}))
        self.assertTrue(is_open_prediction({'tickets':['1-2-3','1-3-2']}))
        self.assertFalse(is_open_prediction({'tickets':['1-2-3','3-2-1']}))

if __name__ == '__main__':unittest.main()
