import tempfile
import unittest
from pathlib import Path

import pandas as pd

from official_method_archive import (SECOND, decode_cup_results, distribute_across_venues,
                                    method_stats, second_method, targets)


def sample_rows(venue="岐阜", cup_id="2026101043", date="2026-10-10", day=1, prefix="014320261010"):
    slug={"岐阜":"gifu","奈良":"nara"}[venue]
    url=f"https://www.winticket.jp/keirin/{slug}/racecard/{cup_id}/{day}/1"
    return pd.DataFrame([{"race_id":prefix,"date":date,"venue":venue,
        "source_url":url,"player_id":f"{n:06d}","finish_pos":n} for n in range(1,8)])


def fake_state(rid="014320261010",idx=1,wrong_winner=False):
    cup={
        "venue":{"name":"岐阜"},
        "races":[{"id":rid,"scheduleId":"2026101043","number":1,"class":"A級",
            "raceType":"チャレンジ予選","startAt":1791616380,
            "entriesNumber":7,"isGradeRace":False}],
        "schedules":[{"id":"2026101043","date":"20261010","index":idx}],
    }
    results={"raceResults":[{"raceId":rid,"results":[
        {"raceId":rid,"order":1,"playerId":"111111" if wrong_winner else "000001","factor":"逃"},
        {"raceId":rid,"order":2,"playerId":"000002","factor":"マ"},
        {"raceId":rid,"order":3,"playerId":"000003","factor":""},
    ]}]}
    return {"tanStackQuery":{"queries":[
        {"queryKey":["keirin/cup","FETCH_KEIRIN_CUP",{"id":"2026101043"}],"state":{"data":cup}},
        {"queryKey":["keirin/cup","FETCH_KEIRIN_CUP_RACE_RESULT",
                     {"cupId":"2026101043","index":idx}],"state":{"data":results}},
    ]}}


class MethodArchiveTests(unittest.TestCase):
    def test_source_targets_extract_day_and_validate_1st_2nd(self):
        day=targets(sample_rows(),asof="2026-10-13",min_date="2026-01-01")
        self.assertEqual(len(day),1)
        (cup,idx), refs=next(iter(day.items()))
        self.assertEqual(idx,1)
        self.assertTrue(cup.endswith("/racecard/2026101043"))
        self.assertEqual(refs[0]["winner_id"],"000001")
        self.assertEqual(refs[0]["second_id"],"000002")

    def test_real_method_feed_extracts_first_and_second_with_podium_verification(self):
        expected=next(iter(targets(sample_rows()).values()))
        actual, errors=decode_cup_results(fake_state(),1,expected)
        self.assertEqual(len(actual),1)
        self.assertEqual(actual[0]["winner_method"],"逃")
        self.assertEqual(actual[0]["second_method"],"マ")
        self.assertEqual(actual[0]["division"],"A_challenge")
        self.assertEqual(actual[0]["time_band"],"evening")
        self.assertFalse(errors)

    def test_wrong_player_or_day_cannot_be_labeled(self):
        expected=next(iter(targets(sample_rows()).values()))
        actual, errors=decode_cup_results(fake_state(wrong_winner=True),1,expected)
        self.assertEqual(len(actual),0)
        self.assertEqual(errors["archive_result_mismatch"],1)
        with self.assertRaises(ValueError):
            decode_cup_results(fake_state(idx=2),1,expected)
        self.assertIsNone(second_method(""))
        self.assertEqual(second_method("マーク"),"マ")

    def test_rotate_venues_before_repeats(self):
        days=targets(pd.concat([sample_rows(),sample_rows("奈良","2026101053","2026-10-11",1,"015320261011"),
            sample_rows("岐阜","2026101243","2026-10-12",1,"014320261012")]))
        selected=distribute_across_venues(days,set(),2)
        self.assertEqual(len(selected),2)
        self.assertEqual(len(set(rows[0]["venue"] for _,rows in selected)),2)

    def test_rotate_historical_years_before_one_year_dominates(self):
        chunks=[
            sample_rows("岐阜","2026090143","2026-09-01",1,"014320260901"),
            sample_rows("岐阜","2026080143","2026-08-01",1,"014320260801"),
            sample_rows("岐阜","2025090143","2025-09-01",1,"014320250901"),
            sample_rows("岐阜","2024090143","2024-09-01",1,"014320240901"),
        ]
        days=targets(pd.concat(chunks),asof="2026-10-13")
        selected=distribute_across_venues(days,set(),3)
        years=[items[0]["date"][:4] for _,items in selected]
        self.assertEqual(years,["2026","2025","2024"])

    def test_full_venue_output_and_mark_column(self):
        expected=next(iter(targets(sample_rows()).values()))
        records,_=decode_cup_results(fake_state(),1,expected)
        with tempfile.TemporaryDirectory() as tmp:
            stats=method_stats(pd.DataFrame(records),tmp)
            self.assertEqual(stats["complete_first"],1)
            self.assertEqual(stats["complete_second"],1)
            overview=pd.read_csv(Path(tmp)/"venue_methods.csv")
            self.assertEqual(len(overview),43)
            actual=overview.set_index("venue").loc["岐阜"]
            self.assertEqual(actual["first_逃_n"],1)
            self.assertEqual(actual["second_マ_n"],1)
            self.assertEqual(actual["pair_逃マ_n"],1)
            self.assertAlmostEqual(actual["pair_逃マ_pct"],100.0)


if __name__=="__main__":
    unittest.main()
