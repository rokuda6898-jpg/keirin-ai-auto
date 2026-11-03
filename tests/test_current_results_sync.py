"""Regression tests for current home page settlement from official outcomes."""
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from sync_current_results import sync


class CurrentResultsSyncTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def write(self, path, data):
        (self.out / path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def load(self):
        return json.loads((self.out / "current_live.json").read_text(encoding="utf-8"))

    @staticmethod
    def race(rid, tickets, eligible=True):
        return {
            "race_id": rid, "date": "2026-11-03", "performance_eligible": eligible,
            "forecast": {"model_id": "MODEL", "tickets": tickets,
                         "saved_at": "2026-11-03T08:00:00+09:00"},
        }

    def test_official_updates_results_not_frozen_forecasts(self):
        rows = [self.race("1", ["1-2-3", "2-3-1"]),
                self.race("2", ["5-3-2"]),
                self.race("3", ["6-2-1"], eligible=False),
                self.race("4", ["4-2-1"])]
        self.write("current_live.json", {
            "model_id": "MODEL", "today": "2026-11-03",
            "generated_at": "2026-11-03T09:12:00+09:00",
            "observed": {}, "races": rows,
        })
        before = deepcopy(rows)
        self.write("latest_results.json", [
            {"race_id": "1", "official_result_available": True,
             "actual_trifecta": "1-2-3",
             "payouts_trifecta_json": '{"1-2-3":15000}',
             "result_observed_at_jst": "2026-11-03T10:00:00+09:00"},
            {"race_id": "4", "official_result_available": True,
             "actual_trifecta": "7-3-4",
             "payouts_trifecta_json": "{}",
             "result_observed_at_jst": "2026-11-03T10:00:00+09:00"},
        ])
        self.write("nexus_live.json", {"races": [
            {"race_id": "1", "result_complete": True, "winners": ["1-2-3"],
             "payouts": {"1-2-3": 15000},
             "results_updated_at": "2026-11-03T10:01:00+09:00"},
            {"race_id": "2", "result_complete": True, "winners": ["2-3-5"],
             "payouts": {"2-3-5": 1000},
             "results_updated_at": "2026-11-03T10:01:00+09:00"},
            {"race_id": "3", "result_complete": True, "winners": ["1-2-3"],
             "payouts": {"1-2-3": 1000},
             "results_updated_at": "2026-11-03T10:01:00+09:00"},
        ]})
        summary = sync(self.out)
        data = self.load()
        self.assertEqual(summary["newly_settled_today"], 3)
        self.assertEqual(data["races"][0]["settlement"]["returned"], 15000)
        self.assertEqual(data["races"][0]["settlement"]["status"], "hit")
        self.assertEqual(data["races"][1]["settlement"]["status"], "miss")
        self.assertEqual(data["observed"]["races"], 2)
        self.assertEqual(data["observed"]["hits"], 1)
        self.assertEqual(data["observed"]["HIGH_hits"], 1)
        self.assertEqual(data["observed"]["pending"], 1)
        self.assertEqual(data["observed"]["unverified_publication"], 1)
        self.assertEqual(data["races"][3].get("settlement"), None)
        self.assertEqual([r["forecast"] for r in data["races"]],
                         [r["forecast"] for r in before])
        self.assertEqual(data["generated_at"], "2026-11-03T09:12:00+09:00")
        self.assertFalse(sync(self.out)["changed"])

    def test_full_refund_and_newer_settlement_protected(self):
        latest = self.race("10", ["1-2-3", "1-3-2"])
        newer = self.race("11", ["7-3-1"])
        newer["settlement"] = {
            "status": "hit", "stake": 100, "returned": 1200,
            "winning_tickets": ["7-3-1"], "payoffs": {"7-3-1": 1200},
            "settled_at": "2026-11-03T12:00:00+09:00",
        }
        self.write("current_live.json", {
            "model_id": "MODEL", "today": "2026-11-03",
            "races": [latest, newer], "observed": {},
        })
        self.write("latest_results.json", [])
        self.write("nexus_live.json", {"races": [
            {"race_id": "10", "refund_all": True, "winners": [],
             "payouts": {}, "result_complete": True,
             "results_updated_at": "2026-11-03T11:00:00+09:00"},
            {"race_id": "11", "result_complete": True,
             "winners": ["1-2-3"], "payouts": {"1-2-3": 1000},
             "results_updated_at": "2026-11-03T11:00:00+09:00"},
        ]})
        result = sync(self.out)
        self.assertEqual(result["updated_races"], 1)
        data = self.load()
        self.assertEqual(data["races"][0]["settlement"]["status"], "void")
        self.assertEqual(data["races"][0]["settlement"]["refund_yen"], 200)
        self.assertEqual(data["races"][0]["settlement"]["returned"], 200)
        self.assertEqual(data["races"][1]["settlement"]["status"], "hit")
        self.assertEqual(data["observed"]["races"], 1)
        self.assertEqual(data["observed"]["void"], 1)


if __name__ == "__main__":
    unittest.main()
