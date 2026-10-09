import copy
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup
from company_decision import decide, build_company_decisions, add_company_decisions
from department_coverage import DEPARTMENTS
from forecast_coverage import reconcile_coverage


class CompanyDecisionTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 10, 10, 8, tzinfo=ZoneInfo('Asia/Tokyo'))
        self.close = (self.now+timedelta(hours=1)).timestamp()
        self.rows = [dict(department=d, top3_cars=[1,2,3], tickets=[], forecast_available=True,
                          snapshot_at=self.now.isoformat(), close_at=self.close, race_id='r') for d in DEPARTMENTS]

    def test_all_departments_affect_decision_and_inputs_unchanged(self):
        before = copy.deepcopy(self.rows)
        a = decide(self.rows, {'1','2','3','4'}, self.now, self.close)
        self.assertEqual(a['top12'], ['1-2-3'])
        self.assertEqual(self.rows, before)
        for row in self.rows[:4]:
            row['top3_cars'] = [4,3,2]
        b = decide(self.rows, {'1','2','3','4'}, self.now, self.close)
        self.assertEqual(b['top12'][0], '4-3-2')
        self.assertEqual(len(b['department_submissions']), 7)
        self.assertLessEqual(len(b['top12']), 12)

    def test_missing_or_late_department_cannot_be_faked(self):
        with self.assertRaises(ValueError):
            decide(self.rows[:-1], {'1','2','3'}, self.now, self.close)
        with self.assertRaises(ValueError):
            decide(self.rows, {'1','2','3'}, self.now+timedelta(hours=2), self.close)
        self.rows[0]['snapshot_at'] = (self.now+timedelta(minutes=1)).isoformat()
        with self.assertRaises(ValueError):
            decide(self.rows, {'1','2','3'}, self.now, self.close)

    def test_shared_fallback_votes_do_not_multiply_and_strategist_does_not_revote(self):
        for row in self.rows[:4]:
            row['opinion_origin']='shared_model_fallback'
        before=copy.deepcopy(self.rows)
        result=decide(self.rows,{'1','2','3'},self.now,self.close)
        weights=result['strategist']['department_vote_weights']
        self.assertAlmostEqual(sum(weights[d] for d in DEPARTMENTS[:4]),1)
        self.assertEqual(weights['strategist_department'],0)
        self.assertEqual(self.rows,before)
        self.assertEqual(result['top12'],['1-2-3'])

    def test_independent_agreement_keeps_its_weight(self):
        result=decide(self.rows,{'1','2','3'},self.now,self.close)
        weights=result['strategist']['department_vote_weights']
        self.assertEqual(weights['data_department'],1)
        self.assertEqual(weights['line_department'],1)

    def test_schedule_only_race_is_counted_as_missing(self):
        schedule = [dict(race_id='missing', date='2026-10-10', venue='場', race_no=2, close_at=self.close)]
        report = reconcile_coverage([], schedule, set(), self.now)
        self.assertEqual(report['input_races'], 1)
        self.assertEqual(report['upcoming_missing_race_ids'], ['missing'])

    def test_freeze_and_render_without_changing_baseline(self):
        pred = pd.DataFrame([dict(race_id='r', date='2026-10-10', venue='場', race_no=1, close_at=self.close, car_no=c) for c in [1,2,3]])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            first = build_company_decisions(pred, {'predictions':self.rows}, self.now, folder)
            for row in self.rows:
                row['top3_cars'] = [3,2,1]
            second = build_company_decisions(pred, {'predictions':self.rows}, self.now+timedelta(minutes=5), folder)
            self.assertEqual(first['predictions'], second['predictions'])
            soup = BeautifulSoup('<article class="race" id="race-r"><div class="picks-panel"><h3>買い目</h3><div class="bet">旧買い目 2-3-1</div></div></article>', 'html.parser')
            add_company_decisions(soup, folder)
            add_company_decisions(soup, folder)
            self.assertEqual(len(soup.select('.company-final-decision')), 1)
            self.assertEqual(soup.select_one('.bet').text, '旧買い目 2-3-1')
            self.assertIn('1-2-3', soup.select_one('.company-final-decision').text)
