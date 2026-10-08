import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock
from zoneinfo import ZoneInfo

import pandas as pd

import site_manager


JST = ZoneInfo("Asia/Tokyo")


class FrozenDateTime(datetime):
    current = None

    @classmethod
    def now(cls, tz=None):
        value = cls.current
        if value is None:
            return super().now(tz)
        if tz is not None:
            return value.astimezone(tz)
        return value.replace(tzinfo=None)


class SiteManagerTests(unittest.TestCase):
    def test_result_timing_distinguishes_normal_waiting_and_overdue(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            today_csv = root / "today_entries.csv"
            output_dir = root / "outputs"
            output_dir.mkdir()

            now = datetime(2026, 10, 5, 23, 30, tzinfo=JST)
            FrozenDateTime.current = now

            rows = [
                {
                    "race_id": "normal",
                    "start_at": int((now - timedelta(minutes=9, seconds=59)).timestamp()),
                    "close_at": int((now - timedelta(minutes=14, seconds=59)).timestamp()),
                },
                {
                    "race_id": "waiting",
                    "start_at": int((now - timedelta(minutes=10)).timestamp()),
                    "close_at": int((now - timedelta(minutes=15)).timestamp()),
                },
                {
                    "race_id": "overdue",
                    "start_at": int((now - timedelta(minutes=20)).timestamp()),
                    "close_at": int((now - timedelta(minutes=25)).timestamp()),
                },
            ]
            pd.DataFrame(rows).to_csv(today_csv, index=False)
            (output_dir / "latest_results.json").write_text(
                json.dumps([
                    {"race_id": "normal", "official_result_available": False},
                    {"race_id": "waiting", "official_result_available": False},
                    {"race_id": "overdue", "official_result_available": False},
                ]),
                encoding="utf-8",
            )

            with mock.patch.multiple(
                site_manager,
                TODAY_CSV=today_csv,
                OUTPUT_DIR=output_dir,
                datetime=FrozenDateTime,
            ):
                problems = site_manager.audit_results()

            by_type = {p["type"]: p for p in problems}
            self.assertNotIn("normal", {
                rid
                for problem in problems
                for rid in problem.get("race_ids", problem.get("overdue_races", []))
            })
            self.assertEqual(by_type["results_waiting"]["race_ids"], ["waiting"])
            self.assertEqual(by_type["results_overdue"]["race_ids"], ["overdue"])

    def test_race_coverage_uses_snapshot_date_across_midnight(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            today_csv = root / "today_entries.csv"
            schedule_csv = root / "latest_race_schedule.csv"

            pd.DataFrame([
                {"race_id": "r1", "date": "2026-10-05", "player_id": "p1"},
                {"race_id": "r2", "date": "2026-10-05", "player_id": "p2"},
            ]).to_csv(today_csv, index=False)
            pd.DataFrame([
                {"race_id": "r1", "date": "2026-10-05"},
                {"race_id": "r2", "date": "2026-10-05"},
                {"race_id": "next-day", "date": "2026-10-06"},
            ]).to_csv(schedule_csv, index=False)

            FrozenDateTime.current = datetime(2026, 10, 6, 0, 5, tzinfo=JST)
            with mock.patch.multiple(
                site_manager,
                TODAY_CSV=today_csv,
                RACE_SCHEDULE_PATH=schedule_csv,
                datetime=FrozenDateTime,
            ):
                problems = site_manager.audit_race_coverage()

            self.assertEqual(problems, [])

    def test_recurrence_streak_resets_after_thirty_minute_gap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "manager_incident_history.jsonl"
            now = datetime.now(JST)
            records = [
                now - timedelta(minutes=5),
                now - timedelta(minutes=20),
                now - timedelta(minutes=55),
                now - timedelta(minutes=130),
            ]
            with history_path.open("w", encoding="utf-8") as f:
                for at in records:
                    f.write(json.dumps({
                        "at_jst": at.isoformat(timespec="seconds"),
                        "problems": [{
                            "type": "results_overdue",
                            "race_ids": ["078420261005"],
                        }],
                    }, ensure_ascii=False) + "\n")

            problems = [{
                "type": "results_overdue",
                "race_ids": ["078420261005"],
            }]
            with mock.patch.object(site_manager, "INCIDENT_HISTORY_PATH", history_path):
                counts = site_manager.incident_recurrence_counts(problems)

            self.assertEqual(counts[("results_overdue", "078420261005")], 2)

    def test_site_player_name_audit_ignores_attribute_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "outputs"
            output_dir.mkdir()
            today_csv = root / "today_entries.csv"

            pd.DataFrame([
                {
                    "race_id": "race1",
                    "player_id": "p100",
                    "car_no": 1,
                    "player_name": "選手A",
                }
            ]).to_csv(today_csv, index=False)

            html = (
                '<html><head><meta name="nexus-render-schema" content="nexus-departments-v2"/></head><body>'
                '<article data-start="1" id="race-race1" class="race venue-visible">'
                '<button class="rider selected" data-name="選手A" '
                'data-car="1" data-extra="x" data-player-id="p100">'
                '<span>選手A</span></button>'
                '</article>'
                '</body></html>'
            )
            html += " " * 1500
            (output_dir / "index.html").write_text(html, encoding="utf-8")

            with mock.patch.multiple(
                site_manager,
                TODAY_CSV=today_csv,
                OUTPUT_DIR=output_dir,
            ):
                problems = site_manager.audit_site_output()

            self.assertEqual(problems, [])

    def test_site_identity_does_not_leak_from_another_race(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root=Path(temp_dir)
            pd.DataFrame([{"race_id":"race1","player_id":"p100","car_no":1,"player_name":"選手A"}]).to_csv(root/"today.csv",index=False)
            document='<meta name="nexus-render-schema" content="nexus-departments-v2"/><article id="race-race1" class="race"></article><article class="race" id="race-race2"><button class="rider" data-name="選手A" data-player-id="p100"></button></article>'+" "*1500
            (root/"index.html").write_text(document,encoding="utf-8")
            with mock.patch.multiple(site_manager,TODAY_CSV=root/"today.csv",OUTPUT_DIR=root):
                problems=site_manager.audit_site_output()
            self.assertEqual(len(problems),1)
            self.assertEqual(problems[0]["type"],"site_player_name_mismatch")

    def test_results_waiting_does_not_enter_repair_loop(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            status_path = Path(temp_dir) / "manager_status.json"
            stats = {
                "race": {
                    "expected_entries": 7,
                    "count": 7,
                }
            }
            waiting = [{"type": "results_waiting", "race_ids": ["race"]}]

            with mock.patch.multiple(
                site_manager,
                STATUS_PATH=status_path,
            ), mock.patch.object(
                site_manager, "audit_entries", return_value=([], stats)
            ), mock.patch.object(
                site_manager, "audit_race_coverage", return_value=[]
            ), mock.patch.object(
                site_manager, "audit_prediction_outputs", return_value=[]
            ), mock.patch.object(
                site_manager, "audit_identity_and_prediction_quality", return_value=[]
            ), mock.patch.object(
                site_manager, "audit_freshness", return_value=[]
            ), mock.patch.object(
                site_manager, "audit_budget", return_value=[]
            ), mock.patch.object(
                site_manager, "audit_live_bets", return_value=[]
            ), mock.patch.object(
                site_manager, "audit_site_output", return_value=[]
            ), mock.patch.object(
                site_manager, "audit_results", return_value=waiting
            ), mock.patch.object(
                site_manager, "repair"
            ) as repair, mock.patch.object(
                site_manager, "snapshot_last_good"
            ):
                site_manager.main()

            repair.assert_not_called()
            payload = json.loads(status_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "waiting_results")
            self.assertEqual(len(payload["attempts"]), 1)


if __name__ == "__main__":
    unittest.main()
