from pathlib import Path
import json
import unittest
import pandas as pd
from bs4 import BeautifulSoup
from site_ui import history_document,enhance_today

class SiteUITests(unittest.TestCase):
    def test_quote_provenance_is_explicit_and_escaped(self):
        from site_ui import odds_provenance
        self.assertIn('未保存',odds_provenance({}))
        text=odds_provenance({'odds_sources':'winticket | <script>','odds_captured_at_jst':'2026-10-06T10:00:00+09:00'})
        self.assertIn('10:00:00',text)
        self.assertIn('WINTICKET',text)
        self.assertNotIn('<script>',text)
        self.assertIn('確認できません',odds_provenance({'odds_sources':'winticket','odds_captured_at_jst':'bad'}))
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

    def test_saved_picks_reject_postclose_and_keep_latest_preclose(self):
        import tempfile
        from unittest import mock
        from site_ui import saved_race_tickets
        with tempfile.TemporaryDirectory() as directory:
            base={"race_id":"014420261006","bet_type":"trifecta","close_at":1000,"prediction_created_at_jst":"1970-01-01T00:15:00Z"}
            pd.DataFrame([dict(base,buy="2-1-6")]).to_csv(Path(directory)/"shadow_bets_20261006_001500.csv",index=False)
            pd.DataFrame([dict(base,buy="1-2-3",prediction_created_at_jst="1970-01-01T00:18:00Z")]).to_csv(Path(directory)/"shadow_bets_20261006_001800.csv",index=False)
            with mock.patch("site_ui.OUTPUT_DIR",Path(directory)):
                self.assertEqual(saved_race_tickets("014420261006",1100)[0]['buy'],'2-1-6')
                self.assertEqual(saved_race_tickets("014420261006",950),[])


    def test_reference_holes_exclude_low_odds_and_main_overlap(self):
        import tempfile
        from site_ui import reference_candidates
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"candidates.csv"
            base={"race_id":"race1","bet_type":"trifecta","prob":0.02,"main_formation":False,"hole_formation":True}
            rows=[dict(base,buy="1-2-3",main_formation=True,odds_used=150),dict(base,buy="2-1-3",odds_used=5.5),dict(base,buy="3-2-1",odds_used=150),dict(base,buy="3-1-2",odds_used=None)]
            pd.DataFrame(rows).to_csv(path,index=False)
            result=reference_candidates("race1",path)
            holes=[row['buy'] for row in result if row['reference_group']=='穴狙い参考']
            self.assertEqual(set(holes),{'3-2-1','3-1-2'})
            self.assertEqual(len({row['buy'] for row in result}),len(result))
