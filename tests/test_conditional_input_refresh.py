import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from conditional_input_refresh import INPUT, refresh, valid_race
from conditional_live import run

NOW = datetime.fromisoformat("2026-10-11T08:00:00+09:00")


def race_rows(rid, close, date="2026-10-11", num=7):
    return [
        {"race_id": rid, "date": date, "car_no": car, "player_id": str(100 + car),
         "entries_number": num, "close_at": close, "player_name": "選手" + str(car)}
        for car in range(1, num + 1)
    ]


def setup(root):
    (root / "outputs/company").mkdir(parents=True)
    (root / "data/raw").mkdir(parents=True)
    early = (NOW + timedelta(minutes=30)).timestamp()
    late = (NOW + timedelta(hours=2)).timestamp()
    closed = (NOW - timedelta(minutes=1)).timestamp()
    schedule = [
        {"race_id": "early", "date": "2026-10-11", "venue": "試験", "race_no": 1,
         "close_at": early, "start_at": early + 300,
         "source_url": "https://www.winticket.jp/keirin/test/racecard/2026101100/1/1"},
        {"race_id": "late", "date": "2026-10-11", "venue": "試験", "race_no": 2,
         "close_at": late, "start_at": late + 300,
         "source_url": "https://www.winticket.jp/keirin/test/racecard/2026101100/1/2"},
        {"race_id": "closed", "date": "2026-10-11", "venue": "試験", "race_no": 3,
         "close_at": closed, "start_at": closed + 300,
         "source_url": "https://www.winticket.jp/keirin/test/racecard/2026101100/1/3"},
    ]
    pd.DataFrame(schedule).to_csv(root / "outputs/latest_race_schedule.csv", index=False)
    return schedule


class ConditionalInputRefreshTests(unittest.TestCase):
    def test_retrieves_missing_upcoming_only_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            schedule = setup(root)
            attempted = []
            by_url = {item["source_url"]: item for item in schedule}
            def fetcher(url, completeness_attempts):
                self.assertEqual(completeness_attempts, 1)
                item = by_url[url]
                attempted.append(item["race_id"])
                return race_rows(item["race_id"], item["close_at"]), []
            first = refresh(root=root, clock=lambda: NOW, fetcher=fetcher)
            self.assertEqual(attempted, ["early", "late"])
            self.assertEqual(first["newly_fetched"], 2)
            self.assertEqual(first["valid_cached"], 2)
            self.assertTrue((root / INPUT).exists())
            repeat = refresh(root=root, clock=lambda: NOW, fetcher=lambda *_args, **_kw: self.fail("refetch"))
            self.assertEqual(repeat["attempted"], 0)
            self.assertEqual(repeat["valid_cached"], 2)
            self.assertEqual(set(pd.read_csv(root / INPUT).race_id), {"early", "late"})

    def test_rejects_mismatched_incomplete_or_closed_races(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            schedule = setup(root)
            def malicious(url, completeness_attempts):
                item = next(x for x in schedule if x["source_url"] == url)
                if item["race_id"] == "early":
                    return race_rows("wrong", item["close_at"]), []
                return race_rows("late", item["close_at"], num=2), []
            result = refresh(root=root, clock=lambda: NOW, fetcher=malicious)
            self.assertEqual(result["valid_cached"], 0)
            self.assertEqual(len(result["failed"]), 2)
            self.assertFalse((root / INPUT).exists())
            self.assertFalse(valid_race(pd.DataFrame(race_rows("early",
                schedule[0]["close_at"], date="2026-10-10")),
                "early", "2026-10-11", schedule[0]["close_at"]))

    def test_conditional_predictor_uses_verified_isolated_entries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            schedule = setup(root)
            pd.DataFrame(race_rows("early", schedule[0]["close_at"],
                                   date="2026-10-10")).to_csv(root/"data/raw/today_entries.csv", index=False)
            pd.DataFrame(race_rows("early", schedule[0]["close_at"])).to_csv(root/INPUT, index=False)
            def predictor(frame, bundle):
                self.assertEqual(set(frame.race_id), {"early"})
                keys = ["1-2-3", "1-3-2", "2-1-3", "2-3-1", "3-1-2",
                        "3-2-1", "1-2-4", "1-4-2", "2-1-4", "4-1-2", "2-4-1", "4-2-1"]
                probs = [1/len(keys)]*len(keys)
                yield {"race_id": "early", "keys": keys,
                       "independent": probs, "chain": probs}
            feed, missing = run(root, clock=lambda: NOW, predictor=predictor,
                model_loader=lambda _: ({"chain_weight": 1, "model_sha256": "test"}, {"cutoff": "2025-01-01"}))
            self.assertEqual(len(next(r for r in feed["races"] if r["id"] == "early")["shadow"]["tickets"]), 12)
            self.assertEqual(set(missing), {"late"})
            self.assertFalse((root / "outputs/company/company_decision_ledger.jsonl").exists())

    def test_same_day_bad_cached_field_is_not_used(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            schedule = setup(root)
            pd.DataFrame(race_rows("early", schedule[0]["close_at"])[:-1]).to_csv(root/INPUT, index=False)
            result = refresh(root=root, clock=lambda: NOW,
                             fetcher=lambda url, completeness_attempts: ([], []))
            self.assertEqual(result["valid_cached"], 0)
            self.assertFalse((root/INPUT).exists())


if __name__ == "__main__":
    unittest.main()
