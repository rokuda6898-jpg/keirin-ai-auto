import copy
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from bs4 import BeautifulSoup
import three_year_equations_ui as ui
from site_ui import add_equation_entry


class EquationNavigationTests(unittest.TestCase):
    def epoch(self, hour, minute=0):
        return datetime(2026,10,9,hour,minute,tzinfo=ui.JST).timestamp()

    def test_start_and_close_are_distinct_and_time_bands_use_start(self):
        row={'race_id':'r','date':'2026-10-09','venue':'平塚','close_at':self.epoch(16,55)}
        schedule={'r':{'date':row['date'],'venue':'平塚','start_at':self.epoch(17)}}
        self.assertEqual(ui.race_times(row,schedule),{'start_label':'17:00','close_label':'16:55','band':'夜'})
        schedule['r']['date']='2026-10-08'
        self.assertEqual(ui.race_times(row,schedule)['start_label'],'未確認')
        self.assertEqual(ui.race_times(row,schedule)['band'],'時刻未確認')
        for hour,band in ((11,'朝'),(12,'昼'),(16,'昼'),(17,'夜'),(20,'夜'),(21,'深夜')):
            self.assertEqual(ui.race_times(dict(row,start_at=self.epoch(hour)),{})['band'],band)

    def test_view_preserves_forecasts_and_escapes_embedded_data(self):
        records=[{'race_id':'r','date':'2026-10-09','venue':'</script><script>bad</script>',
                  'race_no':1,'close_at':self.epoch(18),'snapshot_at_jst':'2026-10-09T10:00:00+09:00',
                  'methods':{'first_anchor':{'tickets':[{'buy':'1-2-3','probability':.1,'stake_yen':100}]}}}]
        before=copy.deepcopy(records)
        value={'training':{},'coverage':{'races':[]},'updated_at_jst':'2026-10-09','statistics':{},'common_comparison_statistics':{}}
        with tempfile.TemporaryDirectory() as temp:
            document=ui.document(records,[],{},value,{'first_anchor':'式'},lambda m:[],Path(temp)/'missing.csv')
        soup=BeautifulSoup(document,'html.parser')
        data=json.loads(soup.select_one('#race-data').string)
        self.assertEqual(data['races'][0]['methods'],records[0]['methods'])
        self.assertEqual(records,before)
        self.assertNotIn('</script><script>bad',document)
        self.assertEqual(len(soup.select('#times button')),6)
        self.assertTrue(soup.select_one('#venues'))
        self.assertTrue(soup.select_one('#day'))

    def test_home_entry_is_idempotent_and_keeps_existing_race_content(self):
        source='<html><body><main><div class="hero">紹介</div><article id="race-1">既存の買い目 1-2-3</article></main></body></html>'
        once=add_equation_entry(source);twice=add_equation_entry(once)
        soup=BeautifulSoup(twice,'html.parser')
        self.assertEqual(len(soup.select('#individual-equation-entry')),0)
        self.assertEqual(len(soup.select('#shadow-original-picks')),1)
        self.assertEqual(soup.select_one('#race-1').get_text(),'既存の買い目 1-2-3')
        self.assertEqual(once,twice)
