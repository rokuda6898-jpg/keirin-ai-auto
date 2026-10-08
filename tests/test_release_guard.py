"""Regression tests for fail-closed release validation."""
import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from release_guard import DEPARTMENTS, SOURCES, SCHEMA, validate_release, write_release_manifest


class ReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.now = datetime(2026, 10, 8, 9, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
        for name in SOURCES:
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("version 1", encoding="utf-8")
        entries = self.root / "data/raw/today_entries.csv"
        entries.parent.mkdir(parents=True)
        entries.write_text("race_id,car_no\n000001,1\n000001,2\n", encoding="utf-8")
        out = self.root / "outputs"
        (out / "company").mkdir(parents=True)
        (out / "latest_predictions.csv").write_text(
            "race_id,car_no\n000001,1\n000001,2\n", encoding="utf-8"
        )
        (out / "index.html").write_text(
            '<meta name="nexus-render-schema" content="' + SCHEMA + '">'
            '<article id="race-000001"></article>', encoding="utf-8"
        )
        payload = {
            "coverage_status": "complete",
            "predictions": [
                {"race_id": "000001", "department": dept,
                 "close_at": (self.now + timedelta(minutes=20)).timestamp(),
                 "forecast_available": True} for dept in DEPARTMENTS
            ],
        }
        (out / "company/all_department_predictions.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
        for name in ("all_department_predictions.html", "all_department_results.html"):
            (out / "company" / name).write_text("<html></html>", encoding="utf-8")
        write_release_manifest(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_healthy_release_is_accepted(self):
        self.assertEqual(validate_release(self.root, self.now), [])

    def test_code_change_requires_real_output_regeneration(self):
        (self.root / "src/predict.py").write_text("version 2", encoding="utf-8")
        self.assertTrue(any("code changed" in error for error in
                            validate_release(self.root, self.now)))

    def test_missing_specialist_cannot_be_published(self):
        path = self.root / "outputs/company/all_department_predictions.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["predictions"] = data["predictions"][:-1]
        path.write_text(json.dumps(data), encoding="utf-8")
        self.assertTrue(any("missing/duplicate department" in error for error in
                            validate_release(self.root, self.now)))

    def test_serialized_html_with_reversed_attributes_is_valid(self):
        (self.root / "outputs/index.html").write_text(
            '<!doctype html><html><head>'
            '<meta content="' + SCHEMA + '" name="nexus-render-schema"/>'
            '</head><body><article class="race" id="race-000001"></article>'
            '</body></html>', encoding="utf-8"
        )
        self.assertEqual(validate_release(self.root, self.now), [])

    def test_reordered_attributes_do_not_hide_a_missing_race(self):
        (self.root / "outputs/index.html").write_text(
            '<meta content="' + SCHEMA + '" name="nexus-render-schema">'
            '<article class="race" id="race-unrelated"></article>',
            encoding="utf-8"
        )
        self.assertTrue(any("race absent from site HTML" in error
                            for error in validate_release(self.root, self.now)))

    def test_stale_site_html_cannot_be_published(self):
        (self.root / "outputs/index.html").write_text(
            '<meta name="nexus-render-schema" content="provisional-picks-v1">',
            encoding="utf-8",
        )
        self.assertTrue(any("HTML schema" in error for error in
                            validate_release(self.root, self.now)))


if __name__ == "__main__":
    unittest.main()
