import json
import unittest
from market_odds_sources import normalize,parse_oddspark,parse_netkeirin,reconcile,parse_kdreams

class MarketOddsTests(unittest.TestCase):
    def test_wrong_pool_and_race_rejected(self):
        with self.assertRaises(ValueError):
            parse_netkeirin({"status":"OK","data":{"nkrace_odds::202610064402":{}}},"20261006","44",1)
        with self.assertRaises(ValueError):
            parse_oddspark('<title>2026年10月6日 大垣競輪 2R</title>',"20261006",1,"大垣")
    def test_ordered_netkeirin_pool_only(self):
        data={"status":"OK","data":{"nkrace_odds::202610064401":{"list_8":[["010206","3"]],"list_9":[["020106","5.5"]]}}}
        quotes,_=parse_netkeirin(data,"20261006","44",1)
        self.assertEqual(quotes,{"2-1-6":"5.5"})
    def test_invalid_and_ceiling_values_excluded(self):
        self.assertEqual(normalize({"2-1-6":5.5,"2-2-6":8,"2-1-9":9,"6-7-3":9999.9,"1-3-4":float("inf")},{1,2,3,4,5,6,7}),{"2-1-6":5.5})
    def test_disagreement_has_no_ev_quote(self):
        chosen,conflicts,evidence=reconcile([{"provider":"a","odds":{"2-1-6":5.5}},{"provider":"b","odds":{"2-1-6":999}}])
        self.assertEqual(chosen,{})
        self.assertIn("2-1-6",conflicts)
    def test_conservative_consensus_and_fallback(self):
        chosen,_,evidence=reconcile([{"provider":"a","odds":{"2-1-6":5.5,"1-2-6":7.5}},{"provider":"b","odds":{"2-1-6":5.6}}])
        self.assertEqual(chosen,{"2-1-6":5.5,"1-2-6":7.5})
        self.assertEqual(evidence["2-1-6"],["a","b"])
