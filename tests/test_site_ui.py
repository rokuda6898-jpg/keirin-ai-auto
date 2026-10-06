import json
import unittest
import pandas as pd
from bs4 import BeautifulSoup
from site_ui import history_document,enhance_today

class SiteUITests(unittest.TestCase):
    def test_history_race_groups_and_pending_returns(self):
        rows=[{"date":"2026-10-06","venue":"大垣","race_no":1,"race_id":"r1","buy":"2-1-6","is_prospective":True,"is_decided":True,"is_hit":True,"stake_yen":100,"actual_return_yen":550,"actual_profit_yen":450}, {"date":"2026-10-06","venue":"大垣","race_no":1,"race_id":"r1","buy":"2-1-4","is_prospective":True,"is_decided":False,"is_hit":False,"stake_yen":100}]
        soup=BeautifulSoup(history_document(pd.DataFrame(rows)),"html.parser")
        races=json.loads(soup.select_one('#historyData').string)
        self.assertEqual(len(races),1)
        self.assertFalse(races[0]['decided'])
        self.assertIsNone(races[0]['return'])
        self.assertIn('200円',races[0]['html'])
        self.assertIn('一部結果待ち',races[0]['html'])
    def test_html_escape_and_script_payload(self):
        row={"date":"2026-10-06","venue":"</script><script>bad</script>","race_no":1,"race_id":"r1","buy":"1-2-3","is_prospective":True,"is_decided":False}
        document=history_document(pd.DataFrame([row]))
        self.assertNotIn('</script><script>bad',document)
        soup=BeautifulSoup(document,"html.parser")
        self.assertEqual(json.loads(soup.select_one('#historyData').string)[0]['venue'],row['venue'])
    def test_today_colored_ticket_and_company_navigation(self):
        document='<html><head></head><body><main><nav></nav><div class="bet"><b>穴 3連単</b><strong>2-1-6</strong><span>200円</span><small>5.5倍</small></div></main></body></html>'
        soup=BeautifulSoup(enhance_today(document),"html.parser")
        self.assertEqual(len(soup.select('.car-box')),3)
        self.assertEqual(soup.select_one('.ticket-amount').get_text(),'200円')
        self.assertTrue(soup.select_one('a[href="company/annual_department_report.html"]'))

    def test_history_uses_latest_preclose_snapshot(self):
        base={"date":"2026-10-06","venue":"大垣","race_no":1,"race_id":"r1","bet_type":"trifecta","is_prospective":True,"is_decided":True,"is_hit":False,"stake_yen":100,"actual_return_yen":0}
        rows=[dict(base,buy="1-2-3",prediction_created_at_jst="2026-10-06T10:00:00+09:00"),dict(base,buy="2-1-6",prediction_created_at_jst="2026-10-06T10:05:00+09:00")]
        soup=BeautifulSoup(history_document(pd.DataFrame(rows)),"html.parser")
        race=json.loads(soup.select_one('#historyData').string)[0]
        self.assertEqual(race['stake'],100)
        self.assertIn('1点',race['html'])
