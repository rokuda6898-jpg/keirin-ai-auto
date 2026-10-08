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
from department_coverage import DEPARTMENTS, build_all_department_coverage, build_department_scoreboard, build_high_payout_axis_report, diagnose_position_misses, freeze_high_payout_axis_experiments, position_scenario
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

    def test_forecasts_are_frozen_once_and_preserved_after_close(self):
        first = build_all_department_coverage(
            self.race, self.plans, specialist_rows(self.race, self.now), self.now, self.output
        )
        ledger = self.output / "company/all_department_prediction_ledger.jsonl"
        frozen_bytes = ledger.read_bytes()
        changed = self.race.copy()
        changed["p_win"] = list(reversed(changed["p_win"].tolist()))
        updated = self.now + timedelta(minutes=2)
        build_all_department_coverage(
            changed, self.plans, specialist_rows(changed, updated), updated, self.output
        )
        self.assertEqual(ledger.read_bytes(), frozen_bytes)
        after_close = build_all_department_coverage(
            changed, self.plans, [], self.now + timedelta(minutes=20), self.output
        )
        original = {p["department"]: p["top3_cars"] for p in first["predictions"]}
        preserved = {p["department"]: p["top3_cars"] for p in after_close["predictions"]}
        self.assertEqual(preserved, original)
        self.assertEqual(len(ledger.read_text().splitlines()), len(DEPARTMENTS))

    def test_frozen_department_ledger_survives_git_reset_checkpoint(self):
        import shutil
        from preserve_validation import save, restore
        build_all_department_coverage(
            self.race, self.plans, specialist_rows(self.race, self.now), self.now, self.output
        )
        original = self.output / "company/all_department_prediction_ledger.jsonl"
        root = self.output / "working"
        target = root / "outputs/company/all_department_prediction_ledger.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        checkpoint = self.output / "checkpoint"
        save(checkpoint, root)
        target.unlink()
        restore(checkpoint, root)
        self.assertEqual(target.read_bytes(), original.read_bytes())
        restore(checkpoint, root)
        self.assertEqual(len(target.read_text().splitlines()), len(DEPARTMENTS))

    def test_miss_diagnostics_distinguish_third_omission_from_order_error(self):
        settled = [
            {"race_id": "r1", "department": "pace_department",
             "predicted": ["1", "2", "3"], "actual": ["1", "2", "4"]},
            {"race_id": "r2", "department": "pace_department",
             "predicted": ["3", "1", "2"], "actual": ["1", "2", "3"]},
            {"race_id": "r3", "department": "risk_department",
             "predicted": ["1", "2", "3"], "actual": ["4", "2", "3"]},
            {"race_id": "r4", "department": "high_payout_department",
             "predicted": ["1", "3", "2"], "actual": ["1", "2", "3"]},
        ]
        report = diagnose_position_misses(settled)
        pace = report["pace_department"]
        self.assertEqual(pace["evaluated_races"], 2)
        self.assertEqual(pace["pattern_counts"]["first_and_second_right_third_wrong"], 1)
        self.assertEqual(pace["pattern_counts"]["actual_third_not_in_top3"], 1)
        self.assertEqual(pace["pattern_counts"]["all_three_right_wrong_order"], 1)
        self.assertEqual(pace["pattern_counts"]["winner_selected_for_second_or_third"], 1)
        self.assertEqual(report["risk_department"]["pattern_counts"]["winner_not_in_top3"], 1)
        self.assertEqual(report["high_payout_department"]["pattern_counts"]["second_third_swapped"], 1)
        self.assertEqual(report["pace_department"]["evidence_status"], "exploratory_small_sample")
        self.assertEqual(report["line_department"]["evaluated_races"], 0)

    def test_longshot_first_axis_comparison_is_preclose_frozen_and_unpriced(self):
        views = specialist_rows(self.race, self.now)
        # Six reference offices back car 1, except risk which backs car 3.
        for item in views:
            if item["department"] == "risk_department":
                item["winner_car"] = 3
                item["top3_cars"] = [3, 1, 2]
        report = build_all_department_coverage(
            self.race, self.plans, views, self.now, self.output
        )
        ledger = self.output / "company/high_payout_axis_shadow_ledger.jsonl"
        frozen = [json.loads(x) for x in ledger.read_text().splitlines()]
        self.assertEqual(len(frozen), 1)
        row = frozen[0]
        self.assertEqual(row["version"], "high_payout_first_axis_shadow_v1")
        self.assertFalse(row["purchase_authorized"])
        self.assertEqual(row["variants"]["six_department_consensus"], 1)
        self.assertEqual(row["variants"]["risk_axis"], 3)
        self.assertEqual(row["variants"]["consensus_veto"], 1)
        self.assertGreater(float(row["close_at"]), self.now.timestamp() + 300)
        snapshot_bytes = ledger.read_bytes()
        # Updating the live ranking cannot rewrite the frozen A/B hypotheses.
        changed = self.race.copy()
        changed["p_win"] = list(reversed(changed["p_win"].tolist()))
        build_all_department_coverage(
            changed, self.plans, views, self.now + timedelta(minutes=2), self.output
        )
        self.assertEqual(ledger.read_bytes(), snapshot_bytes)
        self.assertEqual(report["coverage_status"], "complete")

    def test_longshot_axis_comparison_only_scores_official_postfreeze_outcomes(self):
        views = specialist_rows(self.race, self.now)
        for item in views:
            if item["department"] == "risk_department":
                item["winner_car"] = 3
                item["top3_cars"] = [3, 1, 2]
        build_all_department_coverage(
            self.race, self.plans, views, self.now, self.output
        )
        folder = self.output / "company"
        ledger = folder / "high_payout_axis_shadow_ledger.jsonl"
        row = json.loads(ledger.read_text().splitlines()[0])
        consensus = row["variants"]["six_department_consensus"]
        (self.output / "latest_results.json").write_text(json.dumps([{
            "race_id": "017420261008",
            "official_result_available": False,
            "actual_trifecta": f"{consensus}-2-3" if consensus != 2 else "2-3-4",
        }]), encoding="utf-8")
        self.assertEqual(build_high_payout_axis_report(self.output)["settled_races"], 0)
        actual = f"{consensus}-2-3" if consensus != 2 else "2-3-4"
        (self.output / "latest_results.json").write_text(json.dumps([{
            "race_id": "017420261008",
            "official_result_available": True,
            "actual_trifecta": actual,
        }]), encoding="utf-8")
        summary = build_high_payout_axis_report(self.output)
        self.assertEqual(summary["settled_races"], 1)
        self.assertEqual(summary["variants"]["six_department_consensus"]["winner_hits"], 1)
        self.assertFalse(summary["auto_promotion"])
        self.assertFalse(summary["ready_for_review"])
        self.assertTrue((folder / "high_payout_axis_shadow_report.html").exists())
        (self.output / "latest_results.json").write_text("[]", encoding="utf-8")
        self.assertEqual(build_high_payout_axis_report(self.output)["settled_races"], 1)

    def test_longshot_settled_evidence_survives_corrupted_results_file(self):
        views = specialist_rows(self.race, self.now)
        build_all_department_coverage(
            self.race, self.plans, views, self.now, self.output
        )
        folder = self.output / "company"
        frozen = json.loads((folder / "high_payout_axis_shadow_ledger.jsonl").read_text().splitlines()[0])
        winner = int(frozen["variants"]["current_hole"])
        second, third = [car for car in range(1, 8) if car != winner][:2]
        results_file = self.output / "latest_results.json"
        results_file.write_text(json.dumps([{
            "race_id": "017420261008",
            "official_result_available": True,
            "actual_trifecta": f"{winner}-{second}-{third}",
        }]), encoding="utf-8")
        first = build_high_payout_axis_report(self.output)
        self.assertEqual(first["settled_races"], 1)
        settled_path = folder / "high_payout_axis_shadow_settled.json"
        original = settled_path.read_bytes()
        # Result downloads can be temporarily truncated: do not destroy
        # immutable official settlements just because a source read failed.
        results_file.write_text("{broken-json", encoding="utf-8")
        again = build_high_payout_axis_report(self.output)
        self.assertEqual(again["settled_races"], 1)
        self.assertEqual(settled_path.read_bytes(), original)

        # If the saved settlement itself is corrupt, stop instead of
        # replacing existing evidence with an empty sample.
        settled_path.write_text("{broken-json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            build_high_payout_axis_report(self.output)
        self.assertEqual(settled_path.read_text(encoding="utf-8"), "{broken-json")

    def test_longshot_axis_has_no_hindsight_backfill(self):
        past = entries(self.now, minutes_to_close=-1)
        report = build_all_department_coverage(past, self.plans, [], self.now, self.output)
        self.assertEqual(report["upcoming_missing_forecasts"], 0)
        self.assertEqual(build_high_payout_axis_report(self.output)["frozen_races"], 0)
        self.assertFalse((self.output / "company/high_payout_axis_shadow_ledger.jsonl").exists())

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

    def test_legacy_winner_only_ledger_is_not_misread_as_three_place_forecast(self):
        race = entries(self.now, minutes_to_close=-10)
        folder = self.output / "company"
        folder.mkdir(parents=True, exist_ok=True)
        stored = {
            "department": "data_department",
            "race_id": "017420261008",
            "venue": "高知",
            "race_no": 1,
            "winner_car": 4,
            "snapshot_at": (self.now - timedelta(minutes=20)).isoformat(timespec="seconds"),
            "close_at": float(race.iloc[0]["close_at"]),
            "tickets": [],
        }
        path = folder / "annual_department_prediction_ledger.jsonl"
        path.write_text(json.dumps(stored, ensure_ascii=False) + chr(10), encoding="utf-8")
        knowledge = {
            "window_end_exclusive": "2026-10-08", "asof_date": "2026-10-08",
            "annual_races": 0, "profiles": {},
        }
        with patch("strategist_validation.build_strategist_validation", return_value={}):
            proposals = forecast_departments(race, pd.DataFrame(), knowledge, self.now, self.output)
        self.assertEqual(len(proposals), 4)
        first = next(item for item in proposals if item["department"] == "data_department")
        self.assertEqual(first["winner_car"], 4)
        self.assertFalse(first["forecast_available"])
        self.assertEqual(first["top3_cars"], [])
        self.assertEqual(first["display_status"], "legacy_preclose_winner_only")
        report = json.loads((folder / "annual_department_predictions.json").read_text())
        self.assertEqual(report["forecast_count"], 0)
        self.assertEqual(len(path.read_text().splitlines()), 1)

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
