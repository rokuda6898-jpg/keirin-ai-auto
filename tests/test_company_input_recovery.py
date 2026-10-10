import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from company_input_recovery import recover, read_rows

NOW = datetime.fromisoformat('2026-10-11T07:00:00+09:00')


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fixture(root):
    (root/'data/raw').mkdir(parents=True)
    (root/'outputs').mkdir()
    close = (NOW+timedelta(hours=2)).timestamp()
    write_csv(root/'outputs/latest_race_schedule.csv', [dict(race_id='r1', date='2026-10-11', close_at=close)])
    rows = [dict(race_id='r1', date='2026-10-11', close_at=close, car_no=i,
                 player_id=str(i+100), entries_number=3, declared_entries_number=3,
                 cancelled_car_numbers='', finish_pos='', result_available='0',
                 source_url='https://www.winticket.jp/keirin/test/racecard/2026101100/1/1') for i in (1,2,3)]
    write_csv(root/'data/raw/conditional_today_entries.csv', rows)
    feed = dict(races=[dict(id='r1', date='2026-10-11', riders=[dict(car=str(i)) for i in (1,2,3)],
                shadow=dict(snapshot_at=(NOW-timedelta(minutes=5)).isoformat(), tickets=['3-2-1']))])
    (root/'outputs/conditional_live.json').write_text(json.dumps(feed), encoding='utf-8')
    (root/'data/raw/today_entries.csv').write_text('original', encoding='utf-8')
    return rows, feed


class CompanyRecoveryTests(unittest.TestCase):
    def test_complete_inputs_recovered_without_copying_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); fixture(root)
            self.assertTrue(recover(root, NOW))
            rows=read_rows(root/'data/raw/today_entries.csv')
            self.assertEqual(len(rows),3)
            self.assertNotIn('tickets', rows[0])
            self.assertNotIn('p_win', rows[0])
            self.assertEqual((root/'data/raw/upcoming_races_count.txt').read_text(), '1')
            self.assertFalse((root/'outputs/company/company_decision_ledger.jsonl').exists())

    def test_incomplete_duplicate_outcome_and_other_date_rejected(self):
        for kind in ('missing', 'duplicate', 'outcome', 'date', 'deadline'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root=Path(temp); rows,_=fixture(root)
                if kind=='missing': rows=rows[:-1]
                if kind=='duplicate': rows[1]['car_no']=1
                if kind=='outcome': rows[0]['finish_pos']='1'
                if kind=='date': rows[0]['date']='2026-10-10'
                if kind=='deadline': rows[0]['close_at']=float('nan')
                write_csv(root/'data/raw/conditional_today_entries.csv', rows)
                self.assertFalse(recover(root,NOW))
                self.assertEqual((root/'data/raw/today_entries.csv').read_text(),'original')

    def test_stale_capture_and_future_capture_rejected(self):
        for delta in (-31,1):
            with self.subTest(minutes=delta), tempfile.TemporaryDirectory() as temp:
                root=Path(temp); _,feed=fixture(root)
                feed['races'][0]['shadow']['snapshot_at']=(NOW+timedelta(minutes=delta)).isoformat()
                (root/'outputs/conditional_live.json').write_text(json.dumps(feed),encoding='utf-8')
                self.assertFalse(recover(root,NOW))
                self.assertEqual((root/'data/raw/today_entries.csv').read_text(),'original')

    def test_missing_scheduled_race_does_not_shrink_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); fixture(root)
            schedule=read_rows(root/'outputs/latest_race_schedule.csv')
            schedule.append({**schedule[0], 'race_id':'r2'})
            write_csv(root/'outputs/latest_race_schedule.csv', schedule)
            self.assertFalse(recover(root,NOW))
            self.assertEqual((root/'data/raw/today_entries.csv').read_text(),'original')


if __name__=='__main__': unittest.main()
