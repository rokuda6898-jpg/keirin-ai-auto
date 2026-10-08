"""Regression checks for mandatory, prospective seven-department prediction coverage."""
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from department_coverage import DEPARTMENTS, build_all_department_coverage, build_department_scoreboard, position_scenario
from annual_knowledge import forecast_departments
from site_manager import RiderButtonParser, audit_entries, audit_site_output, normalized_player_id

JST = ZoneInfo("Asia/Tokyo")


def entries(now, minutes_to_close=15):
    return pd.DataFrame([
        {
            "race_id": "017420261008", "date": "2026-10-08",
            "venue": "高知", "race_no": 1, "player_id": str(1000 + car),
            "car_no": car, "close_at": (now + timedelta(minutes=minutes_to_close)).timestamp(),
            "p_win": float(8 - car), "p_second": float(car % 5 + 1),
            "p_third": float(car % 3 + 1),
        } for car in range(1, 8)
    ])


def specialist_rows(race, now):
    rows = []
    for d in DEPARTMENTS[:4]:
        rows.append({
            "department": d, "race_id": str(race.iloc[0]["race_id"]),
            "venue": "高知", "race_no": 1,
            "close_at": float(race.iloc[0]["close_at"]),
            "snapshot_at": now.isoformat(timespec="seconds"),
            "forecast_available": True, "winner_car": 1, "top3_cars": [1, 2, 3],
            "tickets": [], "purchase_authorized": False,
            "display_status": "independent_annual_reference",
        })
    return rows


class DepartmentCoverageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name)
        self.now = datetime(2026, 10, 8, 9, 0, tzinfo=JST)
        self.race = entries(self.now)
        self.plans = [{"race_id": "017420261008", "high_payout_department": {"tickets": []}}]

    def tearDown(self):
        self.tmp.cleanup()

    def test_all_seven_departments_have_three_distinct_cars(self):
        report = build_all_department_coverage(
            self.race, self.plans, specialist_rows(self.race, self.now), self.now, self.output
        )
        self.assertEqual(report["race_count"], 1)
        self.assertEqual(report["responses"], len(DEPARTMENTS))
        self.assertEqual(report["upcoming_missing_forecasts"], 0)
        self.assertEqual(report["coverage_status"], "complete")
        self.assertEqual(set(row["department"] for row in report["predictions"]), set(DEPARTMENTS))
        for row in report["predictions"]:
            self.assertTrue(row["forecast_available"])
            self.assertEqual(len(set(row["top3_cars"])), 3)
            self.assertFalse(row["purchase_authorized"])
        hole = next(row for row in report["predictions"] if row["department"] == "high_payout_department")
        self.assertEqual(hole["display_status"], "unpriced_upset_scenario_not_100plus_bet")
        self.assertTrue((self.output / "company" / "all_department_predictions.html").exists())

    def test_all_seven_are_frozen_and_scored_on_identical_official_results(self):
        report = build_all_department_coverage(
            self.race, self.plans, specialist_rows(self.race, self.now), self.now, self.output
        )
        ledger = (self.output / "company" / "all_department_prediction_ledger.jsonl")
        frozen = [json.loads(line) for line in ledger.read_text().splitlines() if line]
        self.assertEqual(len(frozen), 7)
        main = next(x for x in report["predictions"] if x["department"] == "prediction_department")
        actual = "-".join(map(str, main["top3_cars"]))
        (self.output / "latest_results.json").write_text(
            json.dumps([{"race_id": "017420261008", "actual_trifecta": actual,
                         "official_result_available": True}]), encoding="utf-8"
        )
        scores = build_department_scoreboard(self.output)
        self.assertEqual(scores["departments"]["prediction_department"]["exact_trifecta"], 1)
        self.assertEqual({scores["departments"][d]["races"] for d in DEPARTMENTS}, {1})
        self.assertEqual(build_department_scoreboard(self.output)["settled_predictions"], 7)

    def test_missing_specialist_is_failure_not_a_green_coverage_state(self):
        rows = specialist_rows(self.race, self.now)[:3]
        with self.assertRaisesRegex(RuntimeError, "Mandatory specialist forecasts missing"):
            build_all_department_coverage(self.race, self.plans, rows, self.now, self.output)
        report = json.loads((self.output / "company" / "all_department_predictions.json").read_text())
        self.assertEqual(report["coverage_status"], "incomplete")

    def test_closed_race_is_not_falsely_predicted(self):
        past = entries(self.now, minutes_to_close=-1)
        report = build_all_department_coverage(past, self.plans, [], self.now, self.output)
        self.assertEqual(report["responses"], len(DEPARTMENTS))
        self.assertTrue(all(not row["forecast_available"] for row in report["predictions"]))
        self.assertFalse((self.output / "company" / "all_department_prediction_ledger.jsonl").exists())

    def test_four_specialists_forecast_when_annual_records_are_missing(self):
        knowledge = {
            "window_end_exclusive": "2026-10-08", "asof_date": "2026-10-08",
            "annual_races": 0, "profiles": {},
        }
        empty_tickets = pd.DataFrame(columns=["is_selected", "buy", "ticket_group", "prob", "ev"])
        with patch("annual_knowledge.score_riders", side_effect=lambda frame, quotes: frame), \
             patch("annual_knowledge.select_race", return_value=(empty_tickets, {"main_count": 0, "hole_count": 0})), \
             patch("strategist_validation.build_strategist_validation", return_value={}):
            proposals = forecast_departments(
                self.race, pd.DataFrame(), knowledge, self.now, self.output
            )
        self.assertEqual(len(proposals), 4)
        for entry in proposals:
            self.assertTrue(entry["forecast_available"])
            self.assertEqual(entry["display_status"], "model_fallback_reference_missing")
            self.assertEqual(len(set(entry["top3_cars"])), 3)
            self.assertEqual(len(entry["tickets"]), 0)

    def test_zero_padded_rider_ids_match_rendered_site_name(self):
        parser = RiderButtonParser()
        parser.feed('<article><button class="rider" data-player-id="15667" '
                    'data-name="戸田瑞姫" type="button"></button></article>')
        self.assertEqual(normalized_player_id("015667"), "15667")
        self.assertEqual(parser.riders[normalized_player_id("015667")], "戸田瑞姫")
        self.assertNotIn(normalized_player_id("015668"), parser.riders)

    def test_site_audit_matches_real_name_with_zero_padded_source_player_id(self):
        source = self.output / "entries.csv"
        pd.DataFrame([{
            "race_id": "012220261008", "venue": "前橋", "race_no": 1,
            "car_no": 1, "player_id": "015667", "player_name": "戸田瑞姫",
            "entries_number": 1,
        }]).to_csv(source, index=False)
        site = self.output / "index.html"
        site.write_text(
            '<!doctype html><!-- nexus-render-schema nexus-departments-v2 -->'
            '<article id="race-012220261008"><button class="rider" '
            'data-player-id="15667" data-name="戸田瑞姫"></button></article>'
            + "<!--" + "padding" * 150 + "-->", encoding="utf-8",
        )
        with patch("site_manager.TODAY_CSV", source), \
             patch("site_manager.OUTPUT_DIR", self.output):
            self.assertEqual(audit_site_output(), [])

    def test_site_audit_merges_actual_name_mismatches_by_race(self):
        source = self.output / "entries.csv"
        pd.DataFrame([
            {"race_id": "012220261008", "venue": "前橋", "race_no": 1,
             "car_no": car, "player_id": pid, "player_name": name}
            for car, pid, name in ((1, "015667", "戸田瑞姫"), (2, "015149", "中野咲"))
        ]).to_csv(source, index=False)
        site = self.output / "index.html"
        site.write_text(
            '<!doctype html><!-- nexus-render-schema nexus-departments-v2 -->'
            '<article id="race-012220261008">'
            '<button class="rider" data-player-id="15667" data-name="別人"></button>'
            '<button class="rider" data-player-id="15149" data-name="別人"></button>'
            '</article>' + "<!--" + "padding" * 150 + "-->", encoding="utf-8",
        )
        with patch("site_manager.TODAY_CSV", source), \
             patch("site_manager.OUTPUT_DIR", self.output):
            problems = audit_site_output()
        self.assertEqual(len(problems), 1)
        self.assertEqual(problems[0]["type"], "site_player_name_mismatch")
        self.assertEqual(problems[0]["mismatch_count"], 2)

    def test_cancelled_car_zero_does_not_create_phantom_missing_rider(self):
        entries_path = self.output / "source_entries.csv"
        pd.DataFrame([
            {
                "race_id": "052520261008", "venue": "試験場",
                "race_no": 5, "car_no": car,
                "entries_number": 6, "declared_entries_number": 7,
                "cancelled_car_numbers": "0,2",
            }
            for car in (1, 3, 4, 5, 6, 7)
        ]).to_csv(entries_path, index=False)
        with patch("site_manager.TODAY_CSV", entries_path):
            problems, stats = audit_entries()
        self.assertEqual(problems, [])
        self.assertEqual(stats["052520261008"]["cancelled_cars"], [2])

    def test_four_specialists_never_backfill_postclose(self):
        knowledge = {
            "window_end_exclusive": "2026-10-08", "asof_date": "2026-10-08",
            "annual_races": 0, "profiles": {},
        }
        with patch("strategist_validation.build_strategist_validation", return_value={}):
            proposals = forecast_departments(
                entries(self.now, minutes_to_close=-1), pd.DataFrame(), knowledge,
                self.now, self.output,
            )
        self.assertEqual(len(proposals), 4)
        self.assertTrue(all(not row["forecast_available"] for row in proposals))
        self.assertFalse((self.output / "company" / "annual_department_prediction_ledger.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
