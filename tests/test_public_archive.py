import csv
import json
import tempfile
import unittest
from pathlib import Path
from public_archive import build
from archive_history import export
from build_public_site import build as build_site


class PublicArchiveTests(unittest.TestCase):
    def test_original_site_preserves_tickets_and_relative_links(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'site').mkdir();(root/'outputs/company').mkdir(parents=True)
            (root/'site/index.html').write_text('home')
            original='<body><a href="../history.html">履歴</a><p>本線 1-2-3 100円</p></body>'
            (root/'outputs/company/operations.html').write_text(original,encoding='utf-8')
            (root/'outputs/history.html').write_text('<body>history</body>')
            build_site(root)
            copied=(root/'public-build/original/company/operations.html').read_text(encoding='utf-8')
            self.assertIn('本線 1-2-3 100円',copied)
            self.assertIn('href="../../archive.html"',copied)
            self.assertIn('href="../history.html"',copied)

    def test_history_has_no_invented_predictions_and_keeps_snapshots(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);out=root/'outputs/company';out.mkdir(parents=True)
            (out/'historical_race_archive.csv').write_text('race_id,date,venue,race_no,winner_car_no,second_car_no,third_car_no\na,2025-01-01,場,1,1,2,3\n',encoding='utf-8')
            rows=[dict(race_id='b',date='2025-01-02',venue='場',race_no=2,snapshot_at=t,tickets=[dict(buy='2-3-1',stake_yen=300)]) for t in ['08:00','08:10']]
            (out/'ticket_return_settled.json').write_text(json.dumps(rows))
            records={r['id']:r for r in build(root)['races']}
            self.assertEqual(records['a']['actual'],'1-2-3')
            self.assertEqual(records['a']['forecasts'],[])
            self.assertEqual(len(records['b']['forecasts']),2)

    def test_export_excludes_incomplete_and_tied_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'data/raw').mkdir(parents=True);(root/'outputs/company').mkdir(parents=True)
            with (root/'data/raw/history.csv').open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['race_id','date','venue','race_no','finish_pos','car_no'])
                for rid,positions in [('ok',[1,2,3]),('missing',[1,2]),('tie',[1,1,2,3])]:
                    for car,pos in enumerate(positions,1):writer.writerow([rid,'2020-01-01','track',1,pos,car])
            self.assertEqual(export(root),1)
            # A later partial source must never erase the stored race.
            (root/'data/raw/history.csv').write_text('race_id,date,venue,race_no,finish_pos,car_no\n')
            self.assertEqual(export(root),1)
