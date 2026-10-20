"""Prospective publication, immutable decisions, and common-cohort settlement."""
import copy
from datetime import datetime, timedelta
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import nexus_forward as forward
from nexus_input_snapshots import SNAPSHOTS, canonical, read_json, save_json
from nexus_model.forecast_confidence import from_joint
from nexus_model.replay_evidence import bind_prediction, from_inference

NOW = datetime(2026, 10, 20, 15, 0, tzinfo=ZoneInfo("Asia/Tokyo"))


class Clock:
    def __init__(self):
        self.value = NOW

    def __call__(self):
        return self.value


class ForwardTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.clock = Clock()
        self.cars = [1, 2, 3, 4]
        self.q = [value / 300 for value in range(1, 25)]
        self.manifest = {"id": "nexus-formation-5385-v1", "train": ["2023-10-09", "2025-10-08"]}
        save_json(self.root / "models/nexus53/manifest.json", self.manifest)
        self.manifest_hash = hashlib.sha256((self.root / "models/nexus53/manifest.json").read_bytes()).hexdigest()
        self.row = self.source_row()
        self.save_source()
        self.save_quotes()

    def source_row(self):
        block = dict(first=[1, 2], second=self.cars[:], third=self.cars[:], count=12)
        triples = [t for t in itertools.product(block["first"], block["second"], block["third"]) if len(set(t)) == 3]
        tickets = sorted("-".join(map(str, t)) for t in triples)
        marginal = dict.fromkeys(self.cars, 0.0)
        for t, probability in zip(itertools.permutations(self.cars, 3), self.q):
            marginal[t[0]] += probability
        prediction = dict(model_id=self.manifest["id"], policy_id="flexible-first-v1",
            count=12, tickets=tickets, blocks=[block], category="MAIN",
            marks={str(c): mark for c, mark in zip(self.cars, "◎○▲△")},
            first_probabilities={str(c): marginal[c] for c in self.cars},
            captured_at=(NOW - timedelta(seconds=20)).isoformat(),
            input_captured_at=(NOW - timedelta(seconds=30)).isoformat(),
            source_sha256="a" * 64, input_sha256="b" * 64, history_through="2026-10-19",
            allocations={"5000": {"available": True, "stakes": {tickets[0]: 5000}}},
            auto_budget=5000, odds={tickets[0]: 1},
            confidence=from_joint(self.cars, self.q, tickets, annotated_at=(NOW - timedelta(seconds=20)).isoformat()))
        prediction["replay_evidence"] = from_inference(self.cars, self.q,
            {"tickets": triples, "blocks": [block]}, self.manifest, self.manifest_hash)
        row = dict(race_id="014620261020", date="2026-10-20", race_no=1, venue="fixture",
                   session="night", close_at=(NOW + timedelta(minutes=5)).isoformat(),
                   start_at=(NOW + timedelta(minutes=6)).isoformat(), source_url="https://example.test/race",
                   riders=[dict(car=c, name=f"選手{c}", pid=str(100 + c)) for c in self.cars],
                   prediction=prediction, closed=False, cancelled=False, result_complete=False)
        bind_prediction(prediction, race_id=row["race_id"], race_date=row["date"], close_at=row["close_at"])
        return row

    def save_source(self, rows=None):
        save_json(self.root / "outputs/nexus_live.json", dict(schema=1, date="2026-10-20",
                                                               races=[self.row] if rows is None else rows))

    def save_quotes(self, price=100):
        close = datetime.fromisoformat(self.row["close_at"]).timestamp()
        values = [dict(key=list(t), odds=price, raceId=self.row["race_id"], unitPrice=100, absent=False)
                  for t in itertools.permutations(self.cars, 3)]
        self.quote_path = self.root / SNAPSHOTS / self.row["date"] / "preclose" / (self.row["race_id"] + ".json")
        save_json(self.quote_path, dict(schema=1, source_kind="existing_public_page_response",
            captured_at=(NOW - timedelta(seconds=10)).isoformat(), sha256="c" * 64,
            input_sha256=self.row["prediction"]["input_sha256"],
            meta=dict(race_id=self.row["race_id"], date=self.row["date"], close_at=close),
            data=dict(race=dict(id=self.row["race_id"], number=1, closeAt=close, cancel=False),
                      schedule=dict(date="20261020"), finalOdds=False, results=[],
                      entries=[dict(number=c, absent=False) for c in self.cars], trifecta=values)))

    def run_forward(self, **kwargs):
        return forward.run(self.root, clock=self.clock, **kwargs)

    def result(self, payouts, **extra):
        self.clock.value = NOW + timedelta(minutes=10)
        self.row.update(result_complete=True, closed=True, winners=list(payouts), payouts=payouts,
                        result_captured_at=(NOW + timedelta(minutes=7)).isoformat(),
                        result_source_sha256="d" * 64, **extra)
        self.save_source()

    def test_freezes_complete_ten_and_both_allocations_with_common_budget(self):
        source_before = canonical(self.row)
        report = self.run_forward()
        row = report["races"][0]
        self.assertEqual(row["candidate"]["count"], 10)
        self.assertEqual(row["baseline"]["tickets"], self.row["prediction"]["tickets"])
        self.assertLessEqual(len(row["candidate"]["blocks"]), 4)
        self.assertEqual(row["baseline"]["auto_budget"], row["candidate"]["auto_budget"])
        self.assertEqual(row["candidate"]["auto_budget"], 5000)
        for side in forward.SIDES:
            self.assertEqual(sum(row[side]["allocations"]["5000"]["stakes"].values()), 5000)
            self.assertEqual(row[side]["odds_at"], (NOW - timedelta(seconds=10)).isoformat())
        self.assertNotIn("replay_evidence", row["baseline"])
        self.assertEqual(row["candidate"]["confidence"]["status"], "assessed")
        self.assertEqual(row["saved_at"], NOW.isoformat())
        self.assertEqual(report["summary"]["flat"]["races"], 0)
        self.assertEqual(source_before, canonical(read_json(self.root / "outputs/nexus_live.json")["races"][0]))

    def test_result_or_closed_race_never_reaches_selector(self):
        for case in ("clock", "closed", "result", "refund"):
            with self.subTest(case=case):
                (self.root / forward.FEED).unlink(missing_ok=True)
                row = copy.deepcopy(self.row)
                self.clock.value = NOW
                if case == "clock":
                    self.clock.value += timedelta(minutes=6)
                elif case == "closed":
                    row["closed"] = True
                elif case == "result":
                    row.update(result_complete=True, winners=["1-2-3"], payouts={"1-2-3":1000})
                else:
                    row["refunded"] = ["1-2-3"]
                self.save_source([row])
                with patch.object(forward, "select_tickets", side_effect=AssertionError("must not select")):
                    report = self.run_forward()
                self.assertNotIn("candidate", report["races"][0])
                self.assertEqual(report["summary"]["flat"]["races"], 0)

    def test_invalid_q_identity_and_cars_rejected_without_repair(self):
        for case in ("q", "race", "car", "model", "timestamp", "input_hash"):
            with self.subTest(case=case):
                (self.root / forward.FEED).unlink(missing_ok=True)
                row = copy.deepcopy(self.row)
                evidence = row["prediction"]["replay_evidence"]
                if case == "q":
                    evidence["q"][0] = float("inf")
                    # Invalid numeric records are rejected on load as well as selection.
                    evidence["q"][0] = 1.0
                elif case == "race":
                    evidence["race_id"] = "024620261020"
                    evidence["evidence_sha256"] = hashlib.sha256(canonical({k:v for k,v in evidence.items() if k != "evidence_sha256"}).encode()).hexdigest()
                elif case == "car":
                    row["riders"][-1]["car"] = 5
                elif case == "model":
                    row["prediction"]["model_id"] = "another-model"
                elif case == "timestamp":
                    row["prediction"]["captured_at"] = (NOW + timedelta(minutes=1)).isoformat()
                else:
                    row["prediction"]["input_sha256"] = "e" * 64
                self.save_source([row])
                with patch.object(forward, "select_tickets", side_effect=AssertionError("invalid scores must not select")):
                    report = self.run_forward()
                self.assertNotIn("candidate", report["races"][0])

    def test_delayed_calculation_and_publication_are_discarded(self):
        real_selector = forward.select_tickets
        def slow(*args):
            value = real_selector(*args)
            self.clock.value = NOW + timedelta(minutes=4, seconds=1)
            return value
        with patch.object(forward, "select_tickets", side_effect=slow):
            report = self.run_forward()
        self.assertNotIn("candidate", report["races"][0])
        self.clock.value = NOW
        report = self.run_forward()
        self.assertIn("candidate", report["races"][0])
        self.clock.value = NOW + timedelta(minutes=4)
        dropped = forward.gate_publication(self.root, {}, self.clock)
        self.assertEqual(dropped, [self.row["race_id"]])
        self.assertNotIn("candidate", read_json(self.root / forward.FEED)["races"][0])

    def test_canonical_source_rollback_invalidates_private_forecast(self):
        report = self.run_forward()
        self.row["prediction"].pop("replay_evidence")
        self.save_source()
        dropped = forward.gate_publication(self.root, {}, self.clock)
        self.assertEqual(dropped, [self.row["race_id"]])
        self.assertEqual(read_json(self.root / forward.FEED)["coverage"]["saved"], 0)
        # An original source preserved in history remains valid evidence.
        original = self.source_row()["prediction"]
        self.row["previous_predictions"] = [original]
        self.save_source()
        save_json(self.root / forward.FEED, report)
        self.assertEqual(forward.gate_publication(self.root, {}, self.clock), [])

    def test_publication_rechecks_live_schedule_cars_and_outcome_flags(self):
        frozen = self.run_forward()
        original_source = copy.deepcopy(self.row)
        for changed in ("close", "start", "cars", "cancelled", "closed", "result"):
            with self.subTest(changed=changed):
                save_json(self.root / forward.FEED, frozen)
                self.row = copy.deepcopy(original_source)
                if changed == "close":
                    self.row["close_at"] = (NOW + timedelta(minutes=4)).isoformat()
                elif changed == "start":
                    self.row["start_at"] = (NOW + timedelta(minutes=7)).isoformat()
                elif changed == "cars":
                    self.row["riders"][-1]["car"] = 5
                elif changed == "result":
                    self.row["result_complete"] = True
                else:
                    self.row[changed] = True
                self.save_source()
                self.assertEqual(forward.gate_publication(self.root, {}, self.clock), [self.row["race_id"]])

    def test_nonfinite_publication_lead_is_not_silently_clamped(self):
        for lead in (float("nan"), float("inf"), float("-inf"), True):
            with self.subTest(lead=lead), self.assertRaises(ValueError):
                self.run_forward(min_lead_seconds=lead)
        self.assertFalse((self.root / forward.FEED).exists())

    def test_rerun_never_changes_frozen_forecasts_or_late_allocations(self):
        self.quote_path.unlink()
        first = self.run_forward()
        frozen = {key: copy.deepcopy(first["races"][0][key]) for key in forward.FROZEN}
        self.save_quotes()
        self.clock.value += timedelta(minutes=1)
        with patch.object(forward, "select_tickets", side_effect=AssertionError("frozen")), \
             patch.object(forward, "allocation_snapshot", side_effect=AssertionError("frozen")):
            second = self.run_forward()
        self.assertEqual(frozen, {key: second["races"][0][key] for key in forward.FROZEN})
        self.assertEqual(second["races"][0]["allocation_status"], "missing_quotes")

    def test_dead_heat_and_common_budget_settlement_use_same_cohort(self):
        first = self.run_forward()
        candidate = first["races"][0]["candidate"]["tickets"][0]
        baseline = first["races"][0]["baseline"]["tickets"][0]
        self.result({candidate:1700, baseline:1100})
        report = self.run_forward(settle_only=True)
        flat, allocated = report["summary"]["flat"], report["summary"]["allocated"]
        self.assertEqual(flat["race_ids"], allocated["race_ids"])
        self.assertEqual(flat["baseline"]["stake_yen"], 1200)
        self.assertEqual(flat["candidate"]["stake_yen"], 1000)
        self.assertEqual(flat["baseline"]["payout_yen"], 1100)
        self.assertEqual(flat["candidate"]["payout_yen"], 1700)
        self.assertEqual(allocated["baseline"]["stake_yen"], 5000)
        self.assertEqual(allocated["candidate"]["stake_yen"], 5000)
        self.assertEqual(report["races"][0]["forecast_sha256"], first["races"][0]["forecast_sha256"])
        # Losing a rolling source row must not erase the final result.
        self.save_source([])
        retained = self.run_forward(settle_only=True)
        self.assertEqual(retained["summary"], report["summary"])

    def test_refunds_exclude_both_sides_even_when_ticket_hits(self):
        first = self.run_forward()
        winner = first["races"][0]["candidate"]["tickets"][0]
        self.result({winner:3000}, refunded=["1-2-3"])
        report = self.run_forward(settle_only=True)
        self.assertEqual(report["races"][0]["status"], "excluded")
        self.assertEqual(report["summary"]["flat"]["races"], 0)
        self.assertEqual(report["summary"]["allocated"]["races"], 0)

    def test_missing_quotes_and_infeasible_common_budget_do_not_invent_stakes(self):
        for price in (None, 1):
            with self.subTest(price=price):
                (self.root / forward.FEED).unlink(missing_ok=True)
                self.clock.value = NOW
                self.row = self.source_row()
                self.save_source()
                if price is None:
                    self.quote_path.unlink(missing_ok=True)
                else:
                    self.save_quotes(price)
                report = self.run_forward()
                row = report["races"][0]
                self.assertIsNone(row["candidate"]["auto_budget"])
                self.assertIsNone(row["baseline"]["auto_budget"])
                self.result({row["candidate"]["tickets"][0]:1500})
                settled = self.run_forward(settle_only=True)
                self.assertEqual(settled["summary"]["flat"]["races"], 1)
                self.assertEqual(settled["summary"]["allocated"]["races"], 0)

    def test_unobservable_and_malformed_results_never_enter_metrics(self):
        first = self.run_forward()
        winner = first["races"][0]["candidate"]["tickets"][0]
        for failure in ("future", "fraction", "wrong_winner"):
            with self.subTest(failure=failure):
                save_json(self.root / forward.FEED, first)
                self.result({winner:1500})
                if failure == "future":
                    self.row["result_captured_at"] = (self.clock.value + timedelta(minutes=1)).isoformat()
                elif failure == "fraction":
                    self.row["payouts"][winner] = 1500.5
                else:
                    self.row["winners"] = ["1-2-3"]
                self.save_source()
                report = self.run_forward(settle_only=True)
                self.assertEqual(report["races"][0]["status"], "invalid_result")
                self.assertEqual(report["summary"]["flat"]["races"], 0)

    def test_settle_only_cannot_start_or_create_new_forecasts(self):
        self.assertIsNone(self.run_forward(settle_only=True))
        self.assertFalse((self.root / forward.FEED).exists())
        original = self.row["prediction"].pop("replay_evidence")
        self.save_source()
        self.run_forward()
        self.row["prediction"]["replay_evidence"] = original
        self.save_source()
        with patch.object(forward, "select_tickets", side_effect=AssertionError("settle only")):
            report = self.run_forward(settle_only=True)
        self.assertNotIn("candidate", report["races"][0])

    def test_invalid_existing_feed_and_changed_frozen_plan_are_not_overwritten(self):
        report = self.run_forward()
        report["races"][0]["candidate"]["marks"]["1"] = "tampered"
        save_json(self.root / forward.FEED, report)
        before = (self.root / forward.FEED).read_bytes()
        with self.assertRaisesRegex(ValueError, "integrity"):
            self.run_forward()
        self.assertEqual((self.root / forward.FEED).read_bytes(), before)
        save_json(self.root / forward.FEED, {"schema":0})
        before = (self.root / forward.FEED).read_bytes()
        with self.assertRaises(ValueError):
            self.run_forward()
        self.assertEqual((self.root / forward.FEED).read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
