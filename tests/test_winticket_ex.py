import copy
import csv
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from winticket_ex import (METRICS, collect, feature_rows, line_pairs, metric,
                         posterior_rate, save_snapshot, snapshot, source_metrics)

NOW = datetime(2026, 10, 11, 0, 0, tzinfo=timezone.utc)
URL = 'https://www.winticket.jp/keirin/takeo/racecard/2026100984/3/7'
RID = '078420261011'


def card(n=7):
    return {'race': {'id': RID, 'entriesNumber': n,
                     'closeAt': (NOW + timedelta(hours=2)).timestamp(),
                     'startAt': (NOW + timedelta(hours=2, minutes=5)).timestamp()},
            'entries': [{'number': i, 'playerId': f'{i:06d}', 'absent': False} for i in range(1, n+1)],
            'records': [{'playerId': f'{i:06d}', **{key: {'total':10,'succeeded':2,'percentage':20} for key in METRICS.values()}} for i in range(1,n+1)],
            'linePrediction': {'lines': [{'entries':[{'numbers':[i]} for i in range(1,n+1)]}]}}


class MetricTests(unittest.TestCase):
    def test_correct_semantic_mapping(self):
        result = source_metrics({'exLeftBehind': {'total':24,'succeeded':6,'percentage':25},
                                 'exSplitLine': {'total':10,'succeeded':1,'percentage':10}})
        self.assertEqual(result['chigiri']['rate'], .25)
        self.assertEqual(result['chigirare']['rate'], .1)

    def test_display_truncation_and_exact_rate(self):
        self.assertAlmostEqual(metric({'total':3,'succeeded':2,'percentage':66})['rate'], 2/3)

    def test_zero_history_is_not_failure(self):
        m = metric({'total':0,'succeeded':0,'percentage':0})
        self.assertEqual(m['status'],'no_history')
        self.assertIsNone(m['rate'])

    def test_zero_success_with_trials(self):
        m = metric({'total':12,'succeeded':0,'percentage':0})
        self.assertEqual(m['status'],'observed')
        self.assertEqual(m['rate'],0)

    def test_missing_is_separate(self):
        self.assertEqual(metric(None)['status'],'missing')
        self.assertEqual(metric({})['status'],'invalid')

    def test_invalid_counters(self):
        for n,k,p in [(2,3,100),(-1,0,0),(1.5,1,66),(True,1,100),(10,2,90),(0,0,50)]:
            with self.subTest(n=n,k=k,p=p):
                self.assertEqual(metric({'total':n,'succeeded':k,'percentage':p})['status'],'invalid')

    def test_shrinkage_distinguishes_sample_size(self):
        a=metric({'total':1,'succeeded':1,'percentage':100})
        b=metric({'total':20,'succeeded':20,'percentage':100})
        self.assertLess(posterior_rate(a,.5,5),posterior_rate(b,.5,5))
        self.assertIsNone(posterior_rate(metric(None),.5,5))


class SnapshotTests(unittest.TestCase):
    def test_seven_and_nine(self):
        for n in [7,9]:
            s=snapshot(card(n),URL,NOW,RID)
            self.assertEqual(s['active_entries'],n)
            self.assertEqual(s['available_metrics'],6*n)
            self.assertEqual(len(feature_rows(s,NOW)),n)

    def test_withdrawal_preserved(self):
        d=card();d['entries'][2]['absent']=True
        s=snapshot(d,URL,NOW)
        self.assertEqual(s['active_entries'],6)
        self.assertEqual(s['cancelled_cars'],[3])
        self.assertEqual(line_pairs(s),[])

    def test_missing_five_of_seven_rejected(self):
        d=card();d['entries']=d['entries'][:5]
        with self.assertRaises(ValueError):snapshot(d,URL,NOW)

    def test_missing_and_duplicate_records_rejected(self):
        d=card();d['records'].pop()
        with self.assertRaises(ValueError):snapshot(d,URL,NOW)
        d=card();d['records'].append(d['records'][0])
        with self.assertRaises(ValueError):snapshot(d,URL,NOW)

    def test_cross_race_rejected(self):
        with self.assertRaises(ValueError):snapshot(card(),URL,NOW,'068420261011')

    def test_changed_deadline_rejected(self):
        with self.assertRaises(ValueError):snapshot(card(),URL,NOW,scheduled_close=NOW.timestamp()+3600)

    def test_only_allowlisted_source(self):
        with self.assertRaises(ValueError):snapshot(card(),'https://example.com/keirin/x',NOW)

    def test_naive_clock_rejected(self):
        with self.assertRaises(ValueError):snapshot(card(),URL,NOW.replace(tzinfo=None))

    def test_partial_metrics_do_not_become_zero(self):
        d=card();del d['records'][0]['exSpurt']
        s=snapshot(d,URL,NOW)
        self.assertEqual(s['coverage'],'partial')
        self.assertIsNone(feature_rows(s,NOW)[0]['ex_kamashi_rate'])

    def test_historical_fetch_is_not_prospective(self):
        s=snapshot(card(),URL,NOW+timedelta(days=1))
        self.assertFalse(s['preclose_eligible'])
        with self.assertRaises(ValueError):feature_rows(s,NOW)
        with tempfile.TemporaryDirectory() as p:
            with self.assertRaises(ValueError):save_snapshot(s,p)

    def test_after_forecast_capture_cannot_join(self):
        s=snapshot(card(),URL,NOW+timedelta(minutes=1))
        with self.assertRaises(ValueError):feature_rows(s,NOW)

    def test_deadline_buffer_and_final_odds(self):
        self.assertFalse(snapshot(card(),URL,NOW+timedelta(hours=2,minutes=-5))['preclose_eligible'])
        d=card();d['finalOdds']=True
        self.assertFalse(snapshot(d,URL,NOW)['preclose_eligible'])

    def test_current_results_do_not_change_features(self):
        d=card();base=feature_rows(snapshot(d,URL,NOW),NOW)
        d['results']=[{'playerId':'000001','order':1,'splitLine':True}]
        self.assertEqual(feature_rows(snapshot(d,URL,NOW),NOW),base)

    def test_row_order_invariance(self):
        d=card();base=snapshot(d,URL,NOW)
        d['entries'].reverse();d['records'].reverse()
        self.assertEqual(snapshot(d,URL,NOW),base)

    def test_pair_features_only_for_true_leader_and_second(self):
        s=snapshot(card(),URL,NOW)
        self.assertEqual(line_pairs(s),[(1,2)])
        f=feature_rows(s,NOW)
        self.assertAlmostEqual(f[0]['ex_pair_split_interaction'],.04)
        self.assertIsNone(f[2]['ex_pair_split_interaction'])
        s['line_prediction']['lines'][0]['entries'][1]['numbers']=[2,3]
        self.assertEqual(line_pairs(s),[])

    def test_snapshot_immutable(self):
        s=snapshot(card(),URL,NOW)
        with tempfile.TemporaryDirectory() as p:
            first=save_snapshot(s,p);raw=first.read_bytes()
            self.assertEqual(save_snapshot(s,p),first)
            s['captured_at']=(NOW+timedelta(minutes=1)).isoformat()
            second=save_snapshot(s,p)
            self.assertNotEqual(first,second)
            self.assertEqual(first.read_bytes(),raw)


class CollectorTests(unittest.TestCase):
    def setup_schedule(self,p):
        path=Path(p)/'outputs/latest_race_schedule.csv';path.parent.mkdir(parents=True)
        with path.open('w') as f:
            w=csv.DictWriter(f,fieldnames=['date','race_id','close_at','source_url'])
            w.writeheader();w.writerow({'date':'2026-10-11','race_id':RID,
                                       'close_at':card()['race']['closeAt'],'source_url':URL})

    @patch('winticket_ex.time.sleep')
    def test_capture_cache_and_near_close_refresh(self,_):
        with tempfile.TemporaryDirectory() as p:
            self.setup_schedule(p)
            get=lambda url:(card(),NOW)
            r=collect(p,get,lambda:NOW);self.assertEqual(r['fetched'],1)
            r=collect(p,get,lambda:NOW);self.assertEqual(r['cached'],1)
            near=NOW+timedelta(minutes=100)
            r=collect(p,lambda url:(card(),near),lambda:near)
            self.assertEqual(r['fetched'],1)
            self.assertEqual(len(list((Path(p)/'outputs/ex_data/snapshots').rglob('*.json'))),2)

    @patch('winticket_ex.time.sleep')
    def test_failure_visible_and_no_fake_success(self,_):
        with tempfile.TemporaryDirectory() as p:
            self.setup_schedule(p)
            def bad(url):raise ValueError('source down')
            r=collect(p,bad,lambda:NOW)
            self.assertEqual(r['fetched'],0)
            self.assertEqual(len(r['failures']),1)

    def test_stale_schedule_fails(self):
        with tempfile.TemporaryDirectory() as p:
            self.setup_schedule(p)
            with self.assertRaises(ValueError):collect(p,clock=lambda:NOW+timedelta(days=1))


if __name__=='__main__':unittest.main()
