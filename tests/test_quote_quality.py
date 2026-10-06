import unittest
import pandas as pd
from quote_quality import validate_quotes


class QuoteQualityTests(unittest.TestCase):
    def test_stale_future_invalid_and_excluded_quotes_cannot_generate_ev(self):
        frame=pd.DataFrame({'bet_type':['trifecta']*6,'odds_used':[20]*6,
            'odds_captured_at_jst':['1970-01-01T00:15:00Z','1970-01-01T00:10:00Z','1970-01-01T00:20:00Z','bad',None,'1970-01-01T00:15:00Z'],
            'odds_verification_status':['verified']*5+['excluded']})
        result=validate_quotes(frame,1000,1100)
        self.assertEqual(result.odds_used.notna().tolist(),[True,False,False,False,True,False])
        self.assertTrue(frame.odds_used.notna().all())

    def test_nontrifecta_and_legacy_unknown_are_preserved(self):
        frame=pd.DataFrame({'bet_type':['win'],'odds_used':[2.0],'odds_captured_at_jst':['bad']})
        self.assertEqual(validate_quotes(frame,1000).odds_used.iloc[0],2)
        frame=frame.drop(columns='odds_captured_at_jst')
        self.assertEqual(validate_quotes(frame,1000).odds_used.iloc[0],2)
