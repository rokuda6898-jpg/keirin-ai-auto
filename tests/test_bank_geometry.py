import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from bank_geometry import read_bank_geometry,add_bank_geometry,angle_to_degrees,canonical_name
from bank_geometry_study import read_verified,make_features,make_model,train_and_validate,tabulate


class BankGeometryTests(unittest.TestCase):
    def test_exactly_43_tracks_and_no_source_index_typo(self):
        frame=read_bank_geometry()
        self.assertEqual(len(frame),43)
        self.assertEqual(frame.lap_m.value_counts().to_dict(),{400:32,333:7,500:3,250:1})
        indexed=frame.set_index("venue")
        self.assertEqual(indexed.loc["富山","lap_m"],333)
        self.assertEqual(indexed.loc["平塚","lap_m"],400)
        self.assertAlmostEqual(indexed.loc["立川","cant_deg"],31+13/60+6/3600)
        self.assertEqual(indexed.loc["熊本","configuration_from"],pd.Timestamp("2024-07-01"))
        self.assertEqual(indexed.loc["千葉","race_system"],"PIST6")
        self.assertTrue(pd.isna(indexed.loc["千葉","straight_m"]))
        self.assertEqual(canonical_name("松坂"),"松阪")
        self.assertAlmostEqual(angle_to_degrees("34:01:48"),34.03,places=5)
        with self.assertRaises(ValueError):
            angle_to_degrees("31:13:60")

    def test_excludes_obsolete_kumamoto_and_nonstandard_pist6(self):
        source=pd.DataFrame([
            {"venue":"熊本","date":"2023-03-02"},
            {"venue":"熊本","date":"2024-08-03"},
            {"venue":"富山","date":"2026-02-04"},
            {"venue":"千葉","date":"2026-02-04"},
            {"venue":"松坂","date":"2026-02-04"},
            {"venue":"平塚","date":"2026-02-04"},
        ])
        result=add_bank_geometry(source)
        self.assertEqual(len(result),len(source))
        self.assertTrue(result.iloc[0].bank_historically_obsolete)
        self.assertFalse(result.iloc[0].bank_geometry_usable)
        self.assertTrue(pd.isna(result.iloc[0].bank_lap_m))
        self.assertTrue(result.iloc[1].bank_geometry_usable)
        self.assertEqual(result.iloc[1].bank_lap_m,400)
        self.assertEqual(result.iloc[2].bank_lap_m,333)
        self.assertFalse(result.iloc[3].bank_geometry_usable)
        self.assertEqual(result.iloc[4].bank_lap_m,400)
        self.assertEqual(result.iloc[5].bank_lap_m,400)
        with self.assertRaises(ValueError):
            add_bank_geometry(result)

    @staticmethod
    def fixtures(n=60):
        venues=("奈良","富山","平塚","大宮")
        techniques=("逃","捲","差")
        seconds=("マ","差","逃","捲")
        result=[]
        for i in range(n):
            date=pd.Timestamp("2024-01-01")+pd.Timedelta(days=8*i)
            result.append({
                "race_id":f"official{i:05}","date":date.strftime("%Y-%m-%d"),
                "venue":venues[i%4],"winner_method":techniques[i%3],
                "second_method":seconds[i%4],
                "method_status":"confirmed_by_podium_ids",
                "field":7,"hour_jst":9 if i%2 else 20,
                "time_band":"morning" if i%2 else "evening",
                "tier":"A" if i%2 else "S",
                "division":"A_regular" if i%2 else "S_regular",
                "meeting_grade":"other",
            })
        return pd.DataFrame(result)

    def test_only_real_observed_labels_and_shape_stats(self):
        with tempfile.TemporaryDirectory() as root:
            file=Path(root)/"real_methods.csv"
            self.fixtures().to_csv(file,index=False)
            data=read_verified(file,"2026-10-13")
            self.assertEqual(len(data),60)
            self.assertTrue(data.bank_geometry_usable.all())
            self.assertFalse({"winner_method","second_method","first_second_method_pair"}.intersection(
                make_features(data,True).columns))
            shape=make_features(data,True)
            self.assertIn("bank_straight_m",shape)
            self.assertIn("bank_cant_deg",shape)
            self.assertNotIn("bank_straight_m",make_features(data,False))
            cells=tabulate(data,Path(root)/"reports")
            self.assertGreater(cells["bank_shape_method_rates.csv"],0)
            table=pd.read_csv(Path(root)/"reports"/"bank_class_time_methods.csv")
            self.assertIn("pair_逃マ_count",table.columns)
            self.assertEqual(table.races.sum(),60)

    def test_frozen_split_with_geometry_evaluates_real_outcomes_only(self):
        with tempfile.TemporaryDirectory() as root:
            file=Path(root)/"real_methods.csv"
            self.fixtures(120).to_csv(file,index=False)
            data=read_verified(file,"2026-10-13")
            result=train_and_validate(data,"2026-10-13",root,min_train=10,min_holdout=10)
            self.assertEqual(result["status"],"evaluated_shadow_only")
            for method in ("first","second","pair"):
                self.assertEqual(result["methods"][method]["train"]+result["methods"][method]["test"],120)
                self.assertIn("bank_geometry",result["methods"][method]["scenarios"])
            # An outcome change in a future race cannot change its pre-race bank features.
            row=data.iloc[[0]].copy()
            other=row.copy()
            other["winner_method"]="差" if row.iloc[0].winner_method!="差" else "逃"
            other["second_method"]="マ"
            pd.testing.assert_frame_equal(make_features(row,True),make_features(other,True))


if __name__=="__main__":
    unittest.main()
