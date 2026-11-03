"""Last-night secondary model results must be recovered without inventing picks."""
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from secondary_results_backfill import backfill, pending_races, _safe_url
from secondary_results_sync import FEEDS, sync

DAY = "2026-11-03"
RID = "018320261103"
URL = "https://www.winticket.jp/keirin/kurume/racecard/2026110183/3/1"
NOW = datetime(2026, 11, 4, 1, 30, tzinfo=ZoneInfo("Asia/Tokyo"))


def outcome(rid=RID, available=True):
    return {
        "race_id": rid, "official_result_available": available,
        "actual_trifecta": "1-2-3", "actual_trifecta_buys": ["1-2-3"],
        "payouts_trifecta_json": '{"1-2-3":2700}',
        "actual_trio": "1-2-3", "payouts_trio_json": '{"1-2-3":1500}',
        "actual_exacta": "1-2", "payouts_exacta_json": '{"1-2":1100}',
        "result_observed_at_jst": "2026-11-03T20:35:00+09:00",
    }


class SecondaryResultsBackfillTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.out = self.root / "outputs"
        self.out.mkdir()
        for name, (pool, model) in FEEDS.items():
            tickets = ["1-2-3"] if pool != "exacta" else ["1-2"]
            row = {
                "race_id": RID, "date": DAY, "cup_id": "2026110183", "index": 3,
                "race_number": 1, "close_at": 1793689140, "source_url": URL,
                "performance_eligible": True, "forecast": {
                    "model_id": model, "unit_yen": 100, "count": 1,
                    "tickets": tickets, "total_yen": 100,
                    "saved_at": "2026-11-03T15:00:00+09:00",
                    "source_sha256": "immutable",
                },
            }
            payload = {
                "schema": 1, "model_id": model, "pool": pool,
                "today": DAY, "generated_at": "2026-11-03T15:00:00+09:00",
                "races": [row], "observed": {"model_id": model, "pending": 1},
            }
            (self.out / (name + "_live.json")).write_text(json.dumps(payload))
        # Midnight rollover: today's feed must remain unchanged.
        self.today = [{"race_id": "018620261104", "official_result_available": False}]
        (self.out / "latest_results.json").write_text(json.dumps(self.today))

    def test_retrieve_last_night_original_payouts_without_replacing_today(self):
        requested = []
        def parse(url):
            requested.append(url)
            return outcome()
        report = backfill(self.root, now=NOW, parse=parse)
        self.assertEqual(requested, [URL])
        self.assertEqual(report["saved_official"], 1)
        path = self.out / "official_results_archive" / (DAY + ".json")
        self.assertTrue(path.exists())
        self.assertEqual(json.loads(path.read_text())[0]["race_id"], RID)
        self.assertEqual(json.loads((self.out / "latest_results.json").read_text()), self.today)
        sync(self.out)
        for name, (pool, _) in FEEDS.items():
            feed = json.loads((self.out / (name + "_live.json")).read_text())
            self.assertEqual(feed["observed"]["pending"], 0)
            self.assertEqual(feed["races"][0]["settlement"]["status"], "hit")
            self.assertEqual(feed["races"][0]["forecast"]["source_sha256"], "immutable")
            self.assertEqual(feed["generated_at"], "2026-11-03T15:00:00+09:00")

    def test_no_backfill_for_untrusted_url(self):
        self.assertIsNone(_safe_url({"source_url": "https://evil.example/keirin/kurume/racecard/2026110183/3/1",
                                     "cup_id": "2026110183", "index": 3, "race_number": 1}))
        path = self.out / "trio_live.json"
        feed = json.loads(path.read_text())
        feed["races"][0]["source_url"] = "https://evil.example/another"
        path.write_text(json.dumps(feed))
        pending = pending_races(self.root, now=NOW)
        self.assertEqual(set(pending), {RID})  # other trusted feeds remain
        other = self.out / "exacta_live.json"
        third = self.out / "high_roi_live.json"
        for p in [other, third]:
            data = json.loads(p.read_text())
            data["races"][0]["source_url"] = "https://evil.example/another"
            p.write_text(json.dumps(data))
        self.assertFalse(pending_races(self.root, now=NOW))

    def test_invalid_fetched_identity_cannot_be_saved(self):
        report = backfill(self.root, now=NOW, parse=lambda _: outcome("028320261103"))
        self.assertEqual(report["saved_official"], 0)
        self.assertEqual(report["failures"][0]["reason"], "ValueError")
        self.assertFalse((self.out / "official_results_archive" / (DAY + ".json")).exists())

    def test_not_confirmed_result_remains_pending(self):
        report = backfill(self.root, now=NOW, parse=lambda _: outcome(available=False))
        self.assertEqual(report["saved_official"], 0)
        self.assertFalse((self.out / "official_results_archive" / (DAY + ".json")).exists())

    def test_no_repeated_fetch_when_archived_payout_is_complete(self):
        backfill(self.root, now=NOW, parse=lambda _: outcome())
        report = backfill(self.root, now=NOW, parse=lambda _: self.fail("must reuse archived official result"))
        self.assertEqual(report["attempted"], 0)


if __name__ == "__main__":
    unittest.main()
