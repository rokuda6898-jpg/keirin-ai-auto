import unittest
import pandas as pd
from tactics_feature_audit import truth, line_geometry, analyze


def entrants(date, suffix, *, verified=True, valid_line=True):
    r=[]
    for i in range(1, 4):
        r.append({'race_id':suffix, 'date':date, 'car_no':i, 'player_id':f'p{i}',
                  'line_id':1 if valid_line else None, 'line_position':i if valid_line else None,
                  'line_verification_status':'verified' if verified else 'missing',
                  'result_event_spurtSucceeded':True if i==1 else False,
                  'result_event_thrustSucceeded':None})
    return r

class HistoryAuditTest(unittest.TestCase):
    def test_boolean_denominator(self):
        self.assertIsNone(truth(None))
        self.assertIsNone(truth('nan'))
        self.assertIs(truth(0.0),False)
        self.assertIs(truth('true'),True)
    def test_bad_line_not_promoted(self):
        valid=pd.DataFrame(entrants('2024-05-01','r1'))
        self.assertTrue(line_geometry(valid))
        missing=pd.DataFrame(entrants('2024-05-01','r1',valid_line=False))
        self.assertFalse(line_geometry(missing))
    def test_partition_and_missing_event_are_distinct(self):
        records=entrants('2024-05-01','r1',verified=False)+entrants('2026-01-01','r2')
        output=analyze(pd.DataFrame(records),'2025-10-09')
        self.assertEqual(output['partitions']['training']['line']['usable_verified_geometry'],0)
        self.assertEqual(output['partitions']['training']['line']['geometry_without_verified_status'],1)
        self.assertEqual(output['partitions']['holdout']['line']['usable_verified_geometry'],1)
        e=output['partitions']['training']['events']['result_event_spurtSucceeded']
        self.assertEqual((e['true'],e['false'],e['observed']),(1,2,3))
        self.assertFalse(output['partitions']['training']['events']['result_event_thrustSucceeded']['explicit_denominator_available'])
        self.assertFalse(output['data_point_in_time_verified'])

if __name__=='__main__': unittest.main()
