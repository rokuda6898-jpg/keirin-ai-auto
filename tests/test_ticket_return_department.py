import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from ticket_return_department import save_snapshots, build_ticket_return_department


class TicketReturnTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.now = datetime(2026, 10, 6, 12, tzinfo=ZoneInfo("Asia/Tokyo"))

    def plan(self, race="001"):
        return {"race_id": race, "venue": "test", "race_no": 1,
                "timing_eligible": True, "close_at": (self.now + timedelta(hours=1)).timestamp()}

    def tickets(self):
        return pd.DataFrame([
            {"race_id": "001", "bet_type": "trifecta", "buy": "1-2-3", "ticket_group": "本線", "stake_yen": 200, "purchase_authorized": False},
            {"race_id": "001", "bet_type": "trifecta", "buy": "2-1-3", "ticket_group": "穴", "stake_yen": 100, "purchase_authorized": False}])

    def result(self, odds=10):
        (self.root / "latest_results.json").write_text(json.dumps([
            {"race_id": "001", "official_result_available": True,
             "actual_trifecta": "1-2-3", "actual_trifecta_odds": odds}]))

    def test_full_portfolio_returns_and_skip(self):
        save_snapshots([self.plan(), self.plan("002")], self.tickets(), self.now, self.root)
        self.result()
        report = build_ticket_return_department(self.root)
        self.assertEqual(report["shadow"]["total"]["flat_100yen"]["return_rate"], 5)
        self.assertAlmostEqual(report["shadow"]["total"]["allocated"]["return_rate"], 2000 / 300)
        self.assertEqual(report["shadow"]["hole"]["hits"], 0)
        self.assertEqual(report["authorized_purchase"]["bet_races"], 0)
        self.assertFalse(report["target_validated"])
        self.assertEqual(report["frozen_races"], 2)
        self.assertEqual(report["settled_races"], 1)

    def test_latest_snapshot_replaces_whole_ticket_set_including_skip(self):
        save_snapshots([self.plan()], self.tickets(), self.now, self.root)
        save_snapshots([self.plan()], pd.DataFrame(), self.now + timedelta(minutes=1), self.root)
        self.result()
        report = build_ticket_return_department(self.root)
        self.assertEqual(report["shadow"]["total"]["skip_rate"], 1)
        self.assertIsNone(report["shadow"]["total"]["hit_rate"])
        (self.root / "latest_results.json").write_text("[]")
        self.assertEqual(build_ticket_return_department(self.root)["settled_races"], 1)

    def test_closed_snapshot_rejected_and_original_not_overwritten(self):
        save_snapshots([self.plan()], self.tickets(), self.now, self.root)
        save_snapshots([self.plan()], pd.DataFrame(), self.now, self.root)
        save_snapshots([self.plan()], pd.DataFrame(), self.now + timedelta(hours=2), self.root)
        rows = (self.root / "company/ticket_return_snapshots.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(json.loads(rows[0])["tickets"]), 2)

    def test_all_24_tickets_are_audited_without_top10_cap(self):
        from itertools import permutations
        buys = list(permutations(range(1, 8), 3))[:24]
        tickets = pd.DataFrame([{"race_id": "001", "bet_type": "trifecta",
            "buy": "-".join(map(str, buy)), "ticket_group": "本線" if index < 12 else "穴",
            "stake_yen": 100, "purchase_authorized": False} for index, buy in enumerate(buys)])
        save_snapshots([self.plan()], tickets, self.now, self.root)
        self.result()
        report = build_ticket_return_department(self.root)
        self.assertEqual(report["shadow"]["total"]["flat_100yen"]["stake_yen"], 2400)
        self.assertEqual(report["shadow"]["hole"]["flat_100yen"]["stake_yen"], 1200)
        self.assertEqual(report["monthly"]["2026-10"]["flat_100yen"]["stake_yen"], 2400)
        self.assertTrue((self.root / "company/ticket_return_department.html").exists())

    def test_missing_payout_not_loss_and_old_strategy_not_mixed(self):
        old = self.plan(); old["strategy_version"] = "old"
        save_snapshots([old], self.tickets(), self.now, self.root)
        self.result(None)
        self.assertEqual(build_ticket_return_department(self.root)["settled_races"], 0)
        self.result(10)
        self.assertEqual(build_ticket_return_department(self.root)["settled_races"], 0)

    def test_today_and_rolling_results_keep_date_boundaries(self):
        from betting_logic import STRATEGY_VERSION
        today=datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y-%m-%d")
        yesterday=(datetime.now(ZoneInfo("Asia/Tokyo"))-timedelta(days=1)).strftime("%Y-%m-%d")
        folder=self.root/"company";folder.mkdir()
        rows=[]
        for index in range(105):
            rows.append({"race_id":str(index),"date":today if index==104 else yesterday,"close_at":index,"strategy_version":STRATEGY_VERSION,"tickets":[{"buy":"1-2-3","group":"本線","stake_yen":100,"purchase_authorized":False}],"actual_trifecta":"1-2-3","payout_per_100yen":550})
        (folder/"ticket_return_settled.json").write_text(json.dumps(rows),encoding="utf-8")
        report=build_ticket_return_department(self.root)
        self.assertEqual(report['continuous']['today']['total']['bet_races'],1)
        self.assertEqual(report['continuous']['all']['total']['bet_races'],105)
        self.assertEqual(report['continuous']['last50']['total']['bet_races'],50)
        self.assertEqual(report['continuous']['last100']['total']['bet_races'],100)
        self.assertEqual(report['continuous']['today']['total']['flat_100yen']['return_rate'],5.5)


if __name__ == "__main__":
    unittest.main()
