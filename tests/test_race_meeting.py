import csv
import json
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from race_meeting import build_race_meetings
from company_operations import build_company_operations


class RaceMeetingTests(unittest.TestCase):
    def test_equal_voice_four_topics_missing_data_and_stale_quotes(self):
        now=datetime(2026,10,7,19,tzinfo=ZoneInfo('Asia/Tokyo'))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            plan={'race_id':'a','venue':'四日市','race_no':7,'strategy_version':'test','main_limit':8,
                  'formation':{'first':[1,2]},'chaos_index':45,'fixed_policy_status':'calibration_unverified'}
            (root/'latest_race_strategy.json').write_text(json.dumps([plan]),encoding='utf-8')
            columns=['race_id','strategy_version','bet_type','is_selected','ticket_group','odds_captured_at_jst','odds_verification_status']
            with (root/'latest_bet_candidates.csv').open('w',newline='',encoding='utf-8') as handle:
                writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
                for version,group in [('test','本線'),('test','穴'),('old','本線')]:
                    writer.writerow(dict(zip(columns,['a',version,'trifecta','True',group,(now-timedelta(minutes=6)).isoformat(),'verified'])))
            original=(root/'latest_race_strategy.json').read_bytes()
            meeting=build_race_meetings(root,now)[0]
            self.assertEqual(meeting['main_count'],1)
            self.assertEqual(meeting['hole_count'],1)
            self.assertEqual(len(meeting['topics']),4)
            self.assertEqual(len(meeting['opinions']),5)
            self.assertEqual({v['speaking_weight'] for v in meeting['opinions']},{1})
            self.assertTrue(any('2点' in v for v in meeting['topics']['information_gaps']))
            self.assertTrue(any('並び・欠場' in v for v in meeting['topics']['information_gaps']))
            self.assertFalse(meeting['automatic_promotion'])
            report=build_company_operations(root)
            self.assertTrue(report['employee_workflow']['equal_speaking_rights'])
            self.assertIn('外れる展開',(root/'company/operations.html').read_text(encoding='utf-8'))
            self.assertEqual((root/'latest_race_strategy.json').read_bytes(),original)

    def test_missing_sources_do_not_invent_employees_or_forecasts(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(build_race_meetings(Path(directory),datetime.now(ZoneInfo('Asia/Tokyo'))),[])


if __name__=='__main__':
    unittest.main()
