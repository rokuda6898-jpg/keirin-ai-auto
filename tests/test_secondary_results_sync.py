"""Official result refresh must not regenerate any frozen prediction."""
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from secondary_results_sync import FEEDS, official_outcome, settlement, summary, sync


def forecast(name, pool):
    model = FEEDS[name][1]
    tickets = ["1-2-3", "2-1-3"] if pool == "trifecta" else (
        ["1-2-3", "1-3-4"] if pool == "trio" else ["1-2", "2-1"])
    return {"model_id": model, "pool": pool, "count": 2,
            "unit_yen": 100, "total_yen": 200, "tickets": tickets,
            "saved_at": "2026-11-03T15:00:00+09:00", "source_sha256": "frozen",
            "blocks": [{"untouched": True}]}


def official(*, pool_payouts=True):
    return {"race_id": "018320261103", "official_result_available": True,
            "result_observed_at_jst": "2026-11-03T16:18:00+09:00",
            "actual_trifecta": "1-2-3", "actual_trifecta_buys": ["1-2-3", "2-1-3"],
            "payouts_trifecta_json": json.dumps({"1-2-3": 5000, "2-1-3": 6000}),
            "actual_trio": "1-2-3",
            "actual_exacta": "1-2",
            "payouts_trio_json": json.dumps({"1-2-3": 3200} if pool_payouts else {}),
            "payouts_exacta_json": json.dumps({"1-2": 3500, "2-1": 4000} if pool_payouts else {})}


class SecondaryResultSync(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.out = Path(d.name)
        self.rows = {}
        self.frozen = {}
        for name, (pool, model) in FEEDS.items():
            row = {"race_id": "018320261103", "date": "2026-11-03",
                   "close_at": 1793700000, "performance_eligible": True,
                   "status": "forecast", "forecast": forecast(name, pool)}
            feed = {"schema": 1, "pool": pool, "model_id": model,
                    "generated_at": "2026-11-03T15:00:00+09:00",
                    "today": "2026-11-03", "races": [row],
                    "observed": {"model_id": model, "pending": 1}}
            (self.out / (name + "_live.json")).write_text(json.dumps(feed))
            self.rows[name] = row
            self.frozen[name] = deepcopy(row["forecast"])

    def save_official(self, row):
        (self.out / "latest_results.json").write_text(json.dumps([row]))

    def feed(self, name):
        return json.loads((self.out / (name + "_live.json")).read_text())

    def test_three_models_settle_from_original_official_payouts(self):
        self.save_official(official())
        result = sync(self.out)
        for name in FEEDS:
            feed = self.feed(name)
            row = feed["races"][0]
            self.assertEqual(row["forecast"], self.frozen[name])
            self.assertEqual(feed["generated_at"], "2026-11-03T15:00:00+09:00")
            self.assertEqual(feed["observed"]["races"], 1)
            self.assertEqual(feed["observed"]["hits"], 1)
            self.assertEqual(feed["observed"]["pending"], 0)
            self.assertEqual(result[name]["newly_settled"], 1)
        self.assertEqual(self.feed("high_roi")["races"][0]["settlement"]["returned"], 11000)
        self.assertEqual(self.feed("trio")["races"][0]["settlement"]["returned"], 3200)
        self.assertEqual(self.feed("exacta")["races"][0]["settlement"]["returned"], 7500)
        self.assertTrue(self.feed("high_roi")["races"][0]["settlement"]["HIGH_hit"] is False)

    def test_missing_pool_payout_remains_pending_not_a_fake_loss(self):
        self.save_official(official(pool_payouts=False))
        sync(self.out)
        self.assertEqual(self.feed("high_roi")["observed"]["races"], 1)
        for name in ("trio", "exacta"):
            self.assertNotIn("settlement", self.feed(name)["races"][0])
            self.assertEqual(self.feed(name)["observed"]["pending"], 1)

    def test_prior_official_settlement_not_downgraded_by_pending(self):
        self.save_official(official())
        sync(self.out)
        previous = {name: self.feed(name) for name in FEEDS}
        pending = official(pool_payouts=False)
        pending["official_result_available"] = False
        pending["actual_trifecta_buys"] = []
        self.save_official(pending)
        sync(self.out)
        for name in FEEDS:
            self.assertEqual(self.feed(name), previous[name])

    def test_void_is_excluded_from_roi_denominator(self):
        self.save_official(dict(official(), refund_all=True))
        sync(self.out)
        for name in FEEDS:
            feed = self.feed(name)
            self.assertEqual(feed["observed"]["void"], 1)
            self.assertEqual(feed["observed"]["races"], 0)
            self.assertEqual(feed["observed"]["stake"], 0)
            self.assertEqual(feed["races"][0]["settlement"]["refund_yen"], 200)

    def test_missing_or_invalid_official_winner_is_not_settled(self):
        source = official()
        source["payouts_trifecta_json"] = "{}"
        self.assertIsNone(official_outcome(source, "trifecta"))
        source["official_result_available"] = False
        for pool in ("trifecta", "trio", "exacta"):
            self.assertIsNone(official_outcome(source, pool))


if __name__ == "__main__":
    unittest.main()
