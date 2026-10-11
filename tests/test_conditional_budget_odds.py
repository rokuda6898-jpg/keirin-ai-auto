from datetime import datetime, timedelta
import unittest
from unittest.mock import patch
import conditional_budget_odds as market
from test_conditional_budget import TICKETS

NOW = datetime.fromisoformat('2026-10-11T09:00:00+09:00')

def fixture():
    return {'race_id':'012820261011', 'date':'2026-10-11', 'venue':'立川', 'race_no':1,
            'close_at':(NOW+timedelta(hours=1)).timestamp(), 'top12':TICKETS,
            'source_url':'https://www.winticket.jp/keirin/tachikawa/racecard/2026101028/2/1'}

class ConditionalBudgetOddsTests(unittest.TestCase):
    def test_first_complete_source_is_used_without_mixing(self):
        with patch.object(market, '_winticket', return_value=dict.fromkeys(TICKETS, 20)), patch.object(market, '_oddspark') as second:
            q = market.capture(fixture(), lambda:NOW, get_html=lambda url:'fixture')
            self.assertEqual(q['provider'], 'winticket')
            self.assertEqual(set(q['odds']), set(TICKETS))
            second.assert_not_called()

    def test_missing_primary_quote_uses_complete_fallback(self):
        with patch.object(market, '_winticket', return_value={}), patch.object(market, '_oddspark', return_value=dict.fromkeys(TICKETS, 30)):
            q = market.capture(fixture(), lambda:NOW, get_html=lambda url:'fixture')
            self.assertEqual(q['provider'], 'oddspark')
            self.assertEqual(q['odds'][TICKETS[0]], '30')

    def test_no_complete_source_does_not_merge_partial_maps(self):
        with patch.object(market, '_winticket', return_value=dict.fromkeys(TICKETS[:6], 20)), patch.object(market, '_oddspark', return_value=dict.fromkeys(TICKETS[6:], 30)):
            with self.assertRaises(ValueError):
                market.capture(fixture(), lambda:NOW, get_html=lambda url:'fixture')

    def test_invalid_race_or_url_never_fetches(self):
        for change in ({'race_id':'012820261010'}, {'race_no':2}, {'source_url':'http://127.0.0.1/private'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                market.capture(fixture()|change, lambda:NOW, get_html=lambda u:self.fail('request'))

    def test_closed_during_fetch_is_rejected(self):
        times = iter([NOW, NOW+timedelta(hours=2), NOW+timedelta(hours=2)])
        with patch.object(market, '_winticket', return_value=dict.fromkeys(TICKETS, 20)):
            with self.assertRaises(ValueError):
                market.capture(fixture(), lambda:next(times), get_html=lambda url:'fixture')

    def test_withdrawal_does_not_fall_back_to_stale_source(self):
        with patch.object(market, '_winticket', side_effect=ValueError('withdrawn rider')), patch.object(market, '_oddspark') as fallback:
            with self.assertRaises(ValueError):
                market.capture(fixture(), lambda:NOW, get_html=lambda url:'fixture')
            fallback.assert_not_called()

if __name__ == '__main__':
    unittest.main()
