import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from export_historical_odds import quote,export,COLUMNS


def sample():return {'race_id':'012120240101','date':'2024-01-01','buy':'1-2-3','trifecta_odds':'51.2','source_url':'https://www.winticket.jp/keirin/test/racecard/2024010121/1/1'}


class ExportTests(unittest.TestCase):
    def test_outcome_labels_cannot_affect_export(self):
        a=sample();a.update({'actual_trifecta':'1-2-3','is_actual':True,'finish_pos':1,'payout':5120})
        b={**a,'actual_trifecta':'3-2-1','is_actual':False,'finish_pos':7,'payout':0}
        self.assertEqual(quote(a),quote(b));self.assertEqual(set(quote(a)),set(COLUMNS))

    def test_invalid_quote_is_rejected(self):
        for changes in [{'trifecta_odds':'NaN'},{'buy':'1-1-3'},{'date':'tomorrow'},{'source_url':'https://example.com'}]:
            with self.assertRaises(ValueError):quote({**sample(),**changes})

    def test_cache_fallback_is_explicitly_not_prerace(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'raw';(root/'race_cache').mkdir(parents=True)
            (root/'race_cache/a.json').write_text(json.dumps({'odds_rows':[sample()]}),encoding='utf-8')
            out=Path(d)/'out';result=export(root,out,'fixture-cache')
            self.assertEqual(result['races'],1);self.assertFalse(result['usable_for_same_race_forecast'])
            self.assertFalse(result['preclose_timestamp_certified'])
            with gzip.open(out/'historical_final_trifecta.csv.gz','rt',encoding='utf-8') as src:
                self.assertEqual(len(list(csv.DictReader(src))),1)

    def test_empty_cache_is_not_successful_data_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            result=export(Path(d)/'missing',Path(d)/'out')
            self.assertFalse(result['data_available']);self.assertEqual(result['races'],0)


if __name__=='__main__':unittest.main()
