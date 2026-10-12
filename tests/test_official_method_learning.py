import tempfile
import unittest
from pathlib import Path

import pandas as pd

from official_method_learning import (Distribution, load_records, train_validate,
                                       infer_today)
from official_method_archive import FIRST, SECOND, OUTPUT_COLUMNS


def observed(n=760):
    records=[]
    for i in range(n):
        d=pd.Timestamp("2025-03-01")+pd.Timedelta(days=i)
        records.append({
            "race_id":f"race{i:05d}","date":d,"venue":"奈良" if i%2 else "岐阜",
            "tier":"A" if i%3 else "S","division":"A_regular" if i%3 else "S_regular",
            "field":7,"hour_jst":9 if i%4==0 else 21,"time_band":"morning" if i%4==0 else "late",
            "winner_method":FIRST[i%3],"second_method":SECOND[i%4],
            "method_status":"confirmed_by_podium_ids",
        })
    return pd.DataFrame(records)


class ObservedMethodLearningTests(unittest.TestCase):
    def test_distinct_first_and_second_distributions(self):
        df=observed(600)
        first=Distribution(df,"winner_method",FIRST)
        second=Distribution(df,"second_method",SECOND)
        self.assertEqual(first.n,600)
        self.assertEqual(second.n,600)
        for m in (first,second):
            result=m.predict(df.iloc[0])
            for stage in ("national","venue","venue_band","venue_hour"):
                self.assertAlmostEqual(sum(result[stage].values()),1,places=12)
                self.assertTrue(all(0 < p < 1 for p in result[stage].values()))
        self.assertNotIn("マ",first.predict(df.iloc[0])["venue_hour"])
        self.assertIn("マ",second.predict(df.iloc[0])["venue_hour"])

    def test_target_labels_do_not_affect_prediction(self):
        df=observed(600)
        model=Distribution(df.iloc[:400],"winner_method",FIRST)
        row=df.iloc[500].copy()
        orig=model.predict(row)
        row["winner_method"]="差"
        row["second_method"]="捲"
        self.assertEqual(orig,model.predict(row))

    def test_exact_chronological_test_partition(self):
        df=observed(760)
        outcome=train_validate(df,"2027-04-01")
        self.assertEqual(outcome["status"],"evaluated_shadow_only")
        self.assertGreater(outcome["train_races"],400)
        self.assertGreater(outcome["test_races"],100)
        for col in ("first","second"):
            self.assertIn("venue_hour",outcome["metrics"][col])

    def test_current_race_needs_no_result_label(self):
        df=observed(60)
        fm=Distribution(df,"winner_method",FIRST)
        sm=Distribution(df,"second_method",SECOND)
        with tempfile.TemporaryDirectory() as tmp:
            dest=Path(tmp)/"today.csv"
            today=pd.DataFrame([{"race_id":"today1","date":"2026-10-13",
                "car_no":i,"venue":"奈良","race_class":"A級",
                "race_type":"一般","start_at":1791892200} for i in range(1,8)])
            today.to_csv(dest,index=False)
            preview=infer_today(dest,fm,sm)
            self.assertEqual(len(preview),1)
            self.assertAlmostEqual(sum(float(preview.iloc[0][f"p_first_{m}_venue_hour"]) for m in FIRST),1,places=5)
            self.assertAlmostEqual(sum(float(preview.iloc[0][f"p_second_{m}_venue_hour"]) for m in SECOND),1,places=5)
            self.assertFalse(bool(preview.iloc[0]["purchase_authorized"]))


if __name__=="__main__":
    unittest.main()
