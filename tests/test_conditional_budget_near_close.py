import copy
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import conditional_budget_near_close as near
from conditional_budget import allocate_range, check_saved_plan


class NearCloseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.now = datetime(2026, 10, 19, 20, 0, tzinfo=near.JST)
        self.close = self.now + timedelta(minutes=20)
        self.rid = "016220261019"
        self.tickets = ["1-2-3", "2-1-3", "3-2-1"]
        initial = self.now - timedelta(hours=1)
        budget = allocate_range(self.tickets, dict.fromkeys(self.tickets, 20))
        budget.update(odds_captured_at_jst=(initial-timedelta(seconds=3)).isoformat(),
                      allocation_frozen_at_jst=initial.isoformat())
        self.original = dict(race_id=self.rid, date="2026-10-19", venue="広島", race_no=1,
                             close_at=self.close.timestamp(), top12=self.tickets,
                             snapshot_at_jst=initial.isoformat(), budget=budget,
                             actual=[], payouts={}, cancelled_cars=[])
        self.source = dict(schema=1, source_kind="existing_public_page_response",
                           captured_at=(self.now-timedelta(seconds=30)).isoformat(),
                           sha256="a"*64, url="https://www.winticket.jp/keirin/hiroshima/racecard/2026101862/2/1",
                           meta=dict(race_id=self.rid, date="2026-10-19", venue_id="62", venue="広島",
                                     race_number=1, close_at=self.close.timestamp(), cancel=False),
                           data=dict(race=dict(id=self.rid, number=1, closeAt=self.close.timestamp(),
                                               entriesNumber=7, cancel=False, decidedAt=None),
                                     schedule=dict(date="20261019"), finalOdds=False, results=[],
                                     entries=[dict(number=i, raceId=self.rid, absent=False) for i in range(1, 8)],
                                     trifecta=[dict(key=list(map(int, t.split("-"))), raceId=self.rid,
                                                    unitPrice=100, absent=False, odds=o)
                                                for t, o in zip(self.tickets, (10, 25, 40))]))
        self.write_original()
        self.write_source()

    def write_original(self, rows=None):
        path = self.root / near.LEDGER
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r)+"\n" for r in rows or [self.original]))

    def write_source(self):
        path = self.root / near.SNAPSHOTS / "2026-10-19/preclose" / (self.rid+".json")
        near.save_json(path, self.source)

    def run_at(self, stamp=None):
        return near.run(self.root, clock=lambda: stamp or self.now)

    def write_result(self, complete=True):
        result = dict(race_id=self.rid, official_result_available=True,
                      actual_trifecta_buys=["1-2-3", "2-1-3"],
                      payouts_trifecta_json={"1-2-3": 1000, **({"2-1-3": 1600} if complete else {})},
                      result_observed_at_jst=(self.close+timedelta(minutes=5)).isoformat())
        near.save_json(self.root / "outputs/latest_results.json", [result])
        return result

    def test_same_tickets_budget_and_first_quote_frozen_without_network_or_ledger_edits(self):
        original_bytes = (self.root / near.LEDGER).read_bytes()
        source_bytes = (self.root / near.SNAPSHOTS / "2026-10-19/preclose" / (self.rid+".json")).read_bytes()
        with patch("socket.socket", side_effect=AssertionError("network forbidden")), \
                patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
            payload = self.run_at()
            row = payload["races"][0]
            self.assertEqual(row["status"], "allocated")
            self.assertEqual(row["tickets"], self.tickets)
            self.assertEqual(row["initial_budget"], self.original["budget"])
            self.assertEqual(row["budget"]["budget_yen"], self.original["budget"]["budget_yen"])
            self.assertEqual(row["budget"]["version"], self.original["budget"]["version"])
            self.assertTrue(check_saved_plan(row["budget"], self.tickets))
            self.assertEqual((self.root / near.SNAPSHOTS / "2026-10-19/preclose" / (self.rid+".json")).read_bytes(), source_bytes)
            frozen = copy.deepcopy(row["budget"])
            self.source["data"]["trifecta"][0]["odds"] = 90
            self.source["captured_at"] = (self.now+timedelta(minutes=1)).isoformat()
            self.write_source()
            later = self.run_at(self.now+timedelta(minutes=2))["races"][0]
            self.assertEqual(later["budget"], frozen)
        self.assertEqual((self.root / near.LEDGER).read_bytes(), original_bytes)

    def test_stale_future_initial_and_outside_window_quotes_wait_without_allocation(self):
        stamps = [self.now-timedelta(seconds=601), self.now+timedelta(seconds=1),
                  self.now-timedelta(hours=1), self.close-timedelta(minutes=41)]
        for stamp in stamps:
            with self.subTest(stamp=stamp):
                self.source["captured_at"] = stamp.isoformat()
                self.write_source()
                row = self.run_at()["races"][0]
                self.assertEqual(row["status"], "waiting")
                self.assertIsNone(row["budget"])
                self.assertTrue(row["reason"])
        row = self.run_at(self.close-timedelta(minutes=4))["races"][0]
        self.assertEqual(row["status"], "missed")

    def test_source_identity_absence_result_final_and_incomplete_odds_rejected(self):
        valid = copy.deepcopy(self.source)
        changes = [lambda s: s["meta"].update(date="2026-10-18"),
                   lambda s: s["data"]["race"].update(id="026220261019"),
                   lambda s: s["data"].update(finalOdds=True),
                   lambda s: s["data"]["race"].update(cancel=True),
                   lambda s: s["data"]["race"].update(decidedAt=self.now.timestamp()),
                   lambda s: s["data"]["entries"][0].update(absent=True),
                   lambda s: s["data"]["entries"].pop(),
                   lambda s: s["data"]["trifecta"].pop(),
                   lambda s: s["data"]["trifecta"][0].update(raceId="026220261019"),
                   lambda s: s["data"]["trifecta"][0].update(odds="9999.9")]
        for change in changes:
            with self.subTest(change=changes.index(change)):
                self.source = copy.deepcopy(valid)
                change(self.source)
                self.write_source()
                row = self.run_at()["races"][0]
                self.assertIsNone(row["budget"])
                self.assertTrue(row["reason"])

    def test_infeasible_is_frozen_at_same_budget(self):
        for quote in self.source["data"]["trifecta"]:
            quote["odds"] = 1.1
        self.write_source()
        row = self.run_at()["races"][0]
        self.assertEqual(row["status"], "infeasible")
        self.assertEqual(row["budget"]["budget_yen"], row["initial_budget"]["budget_yen"])
        frozen = copy.deepcopy(row["budget"])
        for quote in self.source["data"]["trifecta"]:
            quote["odds"] = 100
        self.write_source()
        self.assertEqual(self.run_at()["races"][0]["budget"], frozen)

    def test_registration_omits_closed_nonallocated_and_other_day(self):
        rows = [self.original]
        for index in range(2, 5):
            row = copy.deepcopy(self.original)
            row.update(race_id=f"{index:02d}6220261019", race_no=index)
            rows.append(row)
        rows[1]["close_at"] = (self.now-timedelta(minutes=1)).timestamp()
        rows[2]["budget"]["status"] = "infeasible"
        rows[3]["date"] = "2026-10-18"
        self.write_original(rows)
        payload = self.run_at()
        self.assertEqual([r["id"] for r in payload["races"]], [self.rid])
        self.assertEqual(self.run_at(self.close+timedelta(days=1))["races"][0]["budget"], payload["races"][0]["budget"])

    def test_official_settlement_paired_complete_ties_and_no_regression(self):
        self.run_at()
        self.write_result(complete=False)
        payload = self.run_at(self.close+timedelta(minutes=6))
        self.assertEqual(payload["races"][0]["budget_settlement"]["status"], "payout_pending")
        self.assertEqual(payload["comparison"]["paired_races"], 0)
        self.write_result(complete=True)
        payload = self.run_at(self.close+timedelta(minutes=7))
        row = payload["races"][0]
        for key, settlement in (("budget", "budget_settlement"), ("initial_budget", "initial_budget_settlement")):
            expected = sum({"1-2-3": 1000, "2-1-3": 1600}.get(a["buy"], 0)*a["stake_yen"]//100
                           for a in row[key]["allocations"])
            self.assertEqual(row[settlement]["payout_yen"], expected)
        self.assertEqual(payload["comparison"]["paired_races"], 1)
        self.assertEqual(payload["comparison"]["initial"]["invested_yen"], payload["comparison"]["near_close"]["invested_yen"])
        self.write_result(complete=False)
        self.assertEqual(self.run_at(self.close+timedelta(minutes=8))["races"][0]["budget_settlement"], row["budget_settlement"])

    def test_known_results_block_new_allocation_and_refunds_hold_both_versions(self):
        self.write_result()
        row = self.run_at()["races"][0]
        self.assertEqual(row["status"], "excluded")
        self.assertIsNone(row["budget"])
        (self.root / near.FEED).unlink()
        (self.root / "outputs/latest_results.json").unlink()
        self.run_at()
        result = self.write_result()
        result["refund_all"] = True
        near.save_json(self.root / "outputs/latest_results.json", [result])
        payload = self.run_at(self.close+timedelta(minutes=6))
        for key in ("budget_settlement", "initial_budget_settlement"):
            self.assertEqual(payload["races"][0][key]["status"], "refund_pending")
        self.assertEqual(payload["comparison"]["paired_races"], 0)

    def test_publication_gate_drops_only_new_plans_and_refreshes_counts(self):
        baseline = {}
        self.run_at()
        clock = lambda: self.close-timedelta(seconds=59)
        self.assertEqual(near.gate_publication(self.root, baseline, clock=clock), [self.rid])
        payload = near.read_json(self.root / near.FEED)
        self.assertIsNone(payload["races"][0]["budget"])
        self.assertEqual(payload["comparison"]["missed"], 1)
        self.assertEqual(payload["comparison"]["allocated"], 0)
        (self.root / near.FEED).unlink()
        published = self.run_at()
        self.assertEqual(near.gate_publication(self.root, published, clock=clock), [])
        self.assertEqual(near.read_json(self.root / near.FEED), published)

    def test_nexus_only_refunds_hold_both_plans_when_race_identity_matches(self):
        baseline = self.run_at()
        original_bytes = (self.root / near.LEDGER).read_bytes()
        self.write_result()
        cases = [(dict(refunded=["1-2-3"]), True),
                 (dict(refund_all=True), True),
                 (dict(cancelled=True), True),
                 (dict(cancelled=True, race_id="026220261019"), False),
                 (dict(cancelled=True, date="2026-10-18"), False),
                 (dict(cancelled=True, race_no=2), False)]
        for changes, held in cases:
            with self.subTest(changes=changes):
                near.save_json(self.root / near.FEED, baseline)
                nexus = dict(race_id=self.rid, date="2026-10-19", race_no=1)
                nexus.update(changes)
                near.save_json(self.root / "outputs/nexus_live.json", dict(races=[nexus]))
                nexus_bytes = (self.root / "outputs/nexus_live.json").read_bytes()
                payload = self.run_at(self.close+timedelta(minutes=6))
                row = payload["races"][0]
                for key in ("budget_settlement", "initial_budget_settlement"):
                    self.assertEqual(row[key]["status"], "refund_pending" if held else "settled")
                self.assertEqual(payload["comparison"]["paired_races"], 0 if held else 1)
                for key in ("tickets", "initial_budget", "budget"):
                    self.assertEqual(row[key], baseline["races"][0][key])
                self.assertEqual((self.root / near.LEDGER).read_bytes(), original_bytes)
                self.assertEqual((self.root / "outputs/nexus_live.json").read_bytes(), nexus_bytes)


if __name__ == "__main__":
    unittest.main()
