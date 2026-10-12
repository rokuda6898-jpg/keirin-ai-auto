import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from venue_time_learning import (EXPECTED_VENUES, TacticModel, assess, classify,
                                 jst_hour, run, time_band, to_races, today_shadow,
                                 venue_name, winner_method)


def fixture(days=36):
    rows=[]
    for day, d in enumerate(pd.date_range('2025-01-01', periods=days)):
        venue='奈良' if day%2 else '岸和田'
        for car in range(1,8):
            rows.append({'race_id':f'{day:05d}','date':d.strftime('%Y-%m-%d'),
                'venue':venue, 'car_no':car,'player_id':f'{day:05d}-{car}',
                'entries_number':7,'race_class':'A級' if day%3 else 'S級',
                'race_type':'チャレンジ予選' if day%3==1 else '一般',
                'start_at':pd.Timestamp(d.strftime('%Y-%m-%d')+' 21:15:00',tz='Asia/Tokyo').timestamp(),
                'finish_pos':car,'official_finish_pos':car,
                'result_factor':('逃' if day%3 else '捲') if car==1 else '',
                'line_verification_status':'verified','line_id': (car-1)//3 + 1})
    return pd.DataFrame(rows)


class VenueTimeLearningTests(unittest.TestCase):
    def test_all_official_venues_including_unavailable_track(self):
        self.assertEqual(len(EXPECTED_VENUES),43)
        self.assertEqual(venue_name('伊東温泉'), '伊東')
        self.assertIn('千葉', EXPECTED_VENUES)

    def test_grade_car_and_time_logic(self):
        self.assertEqual(classify({'race_class':'Ａ級','race_type':'チャレンジ準決勝'}),('A','A_challenge'))
        self.assertEqual(classify({'race_class':'S級','is_grade_race':True}),('S','S_graded'))
        self.assertEqual(classify({'race_class':'L級'}),('L','L'))
        self.assertEqual(time_band(9),'morning')
        self.assertEqual(time_band(15),'day')
        self.assertEqual(time_band(20),'evening')
        self.assertEqual(time_band(21),'late')
        self.assertEqual(jst_hour(1791892200), 20)
        self.assertEqual(winner_method('捲り'),'捲')
        self.assertIsNone(winner_method('1'))

    def test_invalid_races_and_future_excluded(self):
        rows=fixture(4)
        rows.loc[rows.race_id.eq('00001') & rows.car_no.eq(2),'finish_pos']=1
        rows.loc[rows.race_id.eq('00002') & rows.car_no.eq(7),'car_no']=6
        clean=to_races(rows,'2025-01-04')
        self.assertEqual(clean.race_id.tolist(),['00000'])
        self.assertEqual(clean.iloc[0].method,'捲')
        self.assertEqual(clean.iloc[0].band,'late')
        self.assertTrue(clean.iloc[0].forecast_line_top2)

    def test_method_rates_and_smoothed_time_predictions(self):
        race=to_races(fixture(36),'2025-03-01')
        model=TacticModel(race)
        result=model.predict(race.iloc[0])
        for method in ['national','venue','venue_band','venue_hour']:
            self.assertAlmostEqual(sum(result[method].values()),1)
            self.assertTrue(all(0<p<1 for p in result[method].values()))
        self.assertEqual(model.known_methods,36)

    def test_training_uses_past_only_and_future_labels_cannot_affect(self):
        data=to_races(fixture(36),'2025-03-01')
        test=data.iloc[-1].copy()
        old=TacticModel(data[data.date.lt(test.date)])
        p1=old.predict(test)
        altered=data.copy()
        altered.loc[altered.race_id.eq(test.race_id),'method']='逃'
        new=TacticModel(altered[altered.date.lt(test.date)])
        self.assertEqual(p1,new.predict(test))
        record=altered.iloc[-1].copy()
        orig=old.predict(record)
        record['method']='差'; record['raw_factor']='差';record['forecast_line_top2']=False
        self.assertEqual(orig,old.predict(record))

    def test_full_research_writes_outputs_without_touching_production(self):
        with tempfile.TemporaryDirectory() as temp:
            tmp=Path(temp); history=tmp/'history.csv';fixture(36).to_csv(history,index=False)
            today=tmp/'today.csv';fixture(1).drop(columns=['finish_pos','official_finish_pos','result_factor']).to_csv(today,index=False)
            report=run(history,tmp/'research','2025-04-01',today_path=today,min_races=20)
            self.assertEqual(report['history_races_clean'],36)
            self.assertEqual(report['venues_expected'],43)
            self.assertIn('千葉',report['venues_without_races'])
            self.assertFalse(report['production_modified'])
            self.assertEqual(len(pd.read_csv(tmp/'research/all_venues.csv')),43)
            self.assertEqual(len(pd.read_csv(tmp/'research/today_shadow.csv')),1)
            self.assertTrue((tmp/'research/shadow_tactics_model.json').exists())
            self.assertEqual(json.loads((tmp/'research/summary.json').read_text())['status'],'insufficient_data_do_not_apply')


if __name__ == '__main__':
    unittest.main()
