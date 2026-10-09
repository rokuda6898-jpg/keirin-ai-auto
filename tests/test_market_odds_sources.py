import json
import unittest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo
import market_odds_sources as market
from market_odds_sources import normalize,parse_oddspark,parse_netkeirin,reconcile,parse_kdreams

class MarketOddsTests(unittest.TestCase):
    def test_independent_evidence_does_not_modify_production_snapshot(self):
        now=datetime(2026,10,9,12,tzinfo=ZoneInfo('Asia/Tokyo'))
        rows=[{'race_id':'034820261009','date':'2026-10-09','race_no':3,'venue':'四日市',
               'close_at':(now+timedelta(minutes=20)).timestamp(),'car_no':n,
               'source_url':'https://www.winticket.jp/keirin/yokkaichi/racecard/2026100748/3/3'} for n in (1,2,3)]
        odds=[{'bet_type':'trifecta','buy':'1-2-3','odds_used':30.}]
        with tempfile.TemporaryDirectory() as temp:
            production=Path(temp)/'production'; independent=Path(temp)/'independent'
            production.mkdir(); path=production/'market_odds_034820261009.json';path.write_text('original')
            with patch.object(market,'OUTPUT_DIR',production), patch.object(market,'datetime') as clock, patch.object(market.requests,'get',side_effect=RuntimeError('offline')), patch.object(market.requests,'post',side_effect=RuntimeError('offline')):
                clock.now.return_value=now
                result=market.verify_market(rows,odds,report_dir=independent)
                self.assertEqual(path.read_text(),'original')
                self.assertTrue((independent/path.name).exists())
                self.assertEqual(result[0]['odds_used'],30.)
                market.verify_market(rows,odds)
                self.assertEqual(json.loads(path.read_text())['race_id'],'034820261009')
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
