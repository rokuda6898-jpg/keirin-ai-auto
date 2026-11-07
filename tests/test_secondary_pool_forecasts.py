"""Morning publication must survive late upstream jobs without backdating bets."""
import csv
import json
import tempfile
import unittest
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

from secondary_pool_forecasts import (
    INPUT_PATH, JST, SNAPSHOT_PATH, check_coverage, refresh_inputs, rows_digest,
    run, valid_receipt,
)

DAY = '2026-11-08'
NOW = datetime(2026, 11, 8, 7, tzinfo=JST)


def entries(number=1, hour=8, minute=50):
    start = datetime(2026, 11, 8, hour, minute, tzinfo=JST).timestamp()
    return [dict(race_id=f'{number:02}1220261108', date=DAY, venue='青森', race_no=str(number),
                 car_no=str(car), player_id=str(10000+car), player_name=f'選手{car}',
                 score=str(82-car), win_rate=str(.30-car*.025), place2_rate='.4',
                 place3_rate='.5', recent_avg_finish='3', close_at=str(int(start-300)),
                 start_at=str(int(start)), entries_number='7', declared_entries_number='7',
                 source_url=f'https://www.winticket.jp/keirin/aomori/racecard/2026110812/1/{number}')
            for car in range(1, 8)]


def receipt(rows, captured='2026-11-08T06:00:00+09:00'):
    return dict(race_id=rows[0]['race_id'], date=DAY, source_url=rows[0]['source_url'],
                captured_at=captured, source_sha256='a'*64,
                rows_sha256=rows_digest(rows), rows=deepcopy(rows))


class SecondaryPoolForecasts(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def write_json(self, relative, value):
        path = self.root/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')

    def save_legacy(self, rows, *, date=DAY, scheduled=None):
        path = self.root/'data/raw/today_entries.csv'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        one = {r['race_id']:r for r in (scheduled or rows)}
        predictions = [dict(race_id=rid, date=date, snapshot_at_jst=date+'T06:00:00+09:00',
                            top12=['1-2-3','1-3-4']) for rid in one]
        self.write_json('outputs/company/company_decision_predictions.json',
                        dict(coverage=dict(date=date, schedule_verified=True,
                                           races=[{k:r[k] for k in ['race_id','date','venue','race_no','start_at','close_at']}
                                                  for r in one.values()]), predictions=predictions))

    def save_inputs(self, records, **extra):
        self.write_json(INPUT_PATH, dict(date=DAY, index_verified=True,
                        races={r['race_id']:r for r in records},
                        expected_urls=[r['source_url'] for r in records], **extra))

    def feed(self):
        return json.loads((self.root/'outputs/trio_live.json').read_text())

    def test_stale_company_does_not_block_independent_same_day_input(self):
        rows = entries()
        self.save_legacy(rows, date='2026-11-07')
        self.save_inputs([receipt(rows)])
        self.assertTrue(run(self.root, NOW))
        row = self.feed()['races'][0]
        self.assertIsNotNone(row['forecast'])
        self.assertEqual(row['input_source'], 'independent_public_racecard')
        self.assertEqual(row['forecast']['captured_at'], '2026-11-08T06:00:00+09:00')
        self.assertEqual(row['forecast']['saved_at'], NOW.isoformat())
        self.assertTrue(check_coverage(self.root, NOW))

    def test_missing_receipt_does_not_relabel_stale_company_as_current(self):
        self.save_legacy(entries(), date='2026-11-07')
        self.assertFalse(run(self.root, NOW))
        self.assertIsNone(self.feed()['races'][0]['forecast'])
        self.assertEqual(self.feed()['health']['stage'], 'warning')

    def test_future_naive_and_modified_receipts_cannot_authorize_forecasts(self):
        rows = entries()
        self.save_legacy(rows, date='2026-11-07')
        for captured in ['2026-11-08T07:01:00+09:00','2026-11-08T06:00:00']:
            r = receipt(rows, captured)
            self.assertFalse(valid_receipt(r, NOW))
            self.save_inputs([r])
            self.assertFalse(run(self.root, NOW))
            self.assertIsNone(self.feed()['races'][0]['forecast'])
        r = receipt(rows)
        r['rows'][0]['score'] = '999'
        self.assertFalse(valid_receipt(r, NOW))

    def test_generation_clock_and_five_minute_guard_prevent_late_creation(self):
        rows = entries(hour=7, minute=10)
        self.save_legacy(rows, date='2026-11-07')
        self.save_inputs([receipt(rows)])
        self.assertFalse(run(self.root, NOW))  # Exactly five minutes before close.
        with patch('secondary_pool_forecasts.now_jst', return_value=NOW.replace(hour=8)):
            self.assertFalse(run(self.root))  # Input collection did not freeze an early clock.
        self.assertIsNone(self.feed()['races'][0]['forecast'])

    def test_published_forecasts_and_owner_exacta_are_immutable(self):
        rows = entries()
        self.save_legacy(rows)
        self.write_json('outputs/exacta_live.json', dict(publisher='owner_exacta_hit_v1', sentinel='keep'))
        exacta = (self.root/'outputs/exacta_live.json').read_bytes()
        run(self.root, NOW)
        frozen = deepcopy(self.feed()['races'][0])
        rows[0]['score'] = '10'
        self.save_legacy(rows)
        self.assertTrue(run(self.root, NOW.replace(hour=10)))
        row = self.feed()['races'][0]
        self.assertEqual(row['forecast'], frozen['forecast'])
        self.assertEqual(row['first_published_at'], frozen['first_published_at'])
        self.assertEqual((self.root/'outputs/exacta_live.json').read_bytes(), exacta)

    def test_missing_entire_morning_race_is_counted_from_company_schedule(self):
        first, last = entries(), entries(7,11,30)
        self.save_legacy(last, scheduled=first+last)
        self.assertFalse(run(self.root, NOW))
        c = self.feed()['coverage']
        self.assertEqual((c['expected_races'],c['forecasted_races']), (2,1))
        self.assertEqual(c['missing_race_ids'], [first[0]['race_id']])
        self.assertEqual((c['morning_expected'],c['morning_forecasted']), (2,1))
        self.assertEqual(c['morning_missing_race_ids'], [first[0]['race_id']])
        before = (self.root/'outputs/trio_live.json').read_bytes()
        with self.assertRaisesRegex(ValueError, 'incomplete daily trio'):
            check_coverage(self.root, NOW)
        self.assertEqual((self.root/'outputs/trio_live.json').read_bytes(), before)

    def test_missing_scheduled_url_is_not_double_counted(self):
        first, last = entries(), entries(7,11,30)
        self.save_legacy(last)
        self.write_json(SNAPSHOT_PATH/DAY/'schedule.json', dict(date=DAY, captured_at=NOW.isoformat(),
                        races=[first[0],last[0]]))
        self.write_json(INPUT_PATH, dict(date=DAY, index_verified=True, races={},
                        expected_urls=[first[0]['source_url'],last[0]['source_url']]))
        self.assertFalse(run(self.root, NOW))
        self.assertEqual(self.feed()['coverage']['expected_races'],2)
        self.assertEqual(self.feed()['coverage']['unresolved_source_urls'],[])

    def test_unverified_schedule_never_reports_ready(self):
        rows=entries()
        self.save_inputs([receipt(rows)])
        data=json.loads((self.root/INPUT_PATH).read_text())
        data['index_verified']=False
        self.write_json(INPUT_PATH,data)
        self.assertFalse(run(self.root,NOW))
        self.assertEqual(self.feed()['coverage']['forecasted_races'],1)
        self.assertFalse(self.feed()['coverage']['schedule_verified'])

    def test_authoritative_cancellation_is_not_a_missing_prediction(self):
        rows=entries()
        r=receipt(rows)
        r.update(cancelled=True, rows=[], rows_sha256=rows_digest([]),
                 meta=dict(race_id=r['race_id'],date=DAY,source_url=r['source_url'],cancel=True,
                           venue='青森',venue_id='12',cup_id='2026110812',race_number=1,
                           close_at=int(rows[0]['close_at']),start_at=int(rows[0]['start_at'])))
        self.save_inputs([r])
        self.assertTrue(run(self.root,NOW))
        self.assertEqual(self.feed()['coverage']['cancelled_races'],1)
        self.assertEqual(self.feed()['coverage']['missing_race_ids'],[])
        self.assertEqual(self.feed()['races'][0]['status'],'cancelled')
        self.assertTrue(check_coverage(self.root,NOW))

    def test_verified_saved_index_and_valid_inputs_need_no_network(self):
        rows=entries()
        self.save_legacy(rows)
        self.write_json(SNAPSHOT_PATH/DAY/'index.json',dict(date=DAY,captured_at=NOW.isoformat(),
                        urls=[rows[0]['source_url']]))
        index=Mock(side_effect=AssertionError('unnecessary index fetch'))
        fetch=Mock(side_effect=AssertionError('unnecessary racecard fetch'))
        result=refresh_inputs(self.root,clock=lambda:NOW,index_loader=index,fetcher=fetch)
        index.assert_not_called()
        fetch.assert_not_called()
        self.assertTrue(result['index_verified'])

    def test_source_denial_stops_all_further_source_requests(self):
        rows=entries()
        self.save_legacy(rows,date='2026-11-07')
        denied=RuntimeError('403 Forbidden')
        denied.response=type('Response',(),{'status_code':403})()
        fetch=Mock()
        result=refresh_inputs(self.root,clock=lambda:NOW,index_loader=Mock(side_effect=denied),fetcher=fetch)
        fetch.assert_not_called()
        self.assertFalse(result['index_verified'])
        self.assertEqual(len(result['errors']),1)


if __name__=='__main__':
    unittest.main()
