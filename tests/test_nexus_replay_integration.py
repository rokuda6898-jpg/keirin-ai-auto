"""Exercise full-score capture and both publication paths without HTTP/models."""
import ast
import copy
import hashlib
import itertools
import json
import os
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import test_nexus_saved_inputs as fixtures
from nexus_model import base, team
from nexus_model.forecast_confidence import from_joint
from nexus_model.publication import allocation_snapshot, is_open_prediction, settle_prediction
from nexus_model.replay_evidence import bind_prediction, from_inference, verify


def load_definitions(path, names, namespace):
    """Run the actual production definitions while replacing external adapters."""
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in parsed.body if getattr(node, "name", None) in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def plan(cars):
    return dict(tickets=list(itertools.permutations(cars, 3)),
                blocks=[dict(first=cars[:], second=cars[:], third=cars[:], count=6)])


class EvidenceEngine(fixtures.FakeEngine):
    def predict(self, matrix):
        prediction = super().predict(matrix)
        cars = matrix["rows"][0]["cars"]
        size = len(cars) * (len(cars) - 1) * (len(cars) - 2)
        q = [value / (size * (size + 1) / 2) for value in range(1, size + 1)]
        prediction["q"] = q
        prediction["confidence"] = from_joint(cars, q, prediction["tickets"],
                                                annotated_at=fixtures.NOW.isoformat())
        prediction["replay_evidence"] = from_inference(
            cars, q, plan(cars[:3]),
            dict(id="fixture", train=["2023-10-09", "2025-10-08"]), "a" * 64)
        return prediction


class EngineReplayTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = dict(id="fixture", assets={}, train=["2023-10-09", "2025-10-08"],
                             team_selection=dict(final_temperature=1),
                             score_selection={"6": "general_only_free"},
                             formation_selection={"6": "shape_only"},
                             v2_calibration=dict(judge=1, experts=[1, 1, 1, 1]))
        self.manifest_bytes = (json.dumps(self.manifest, indent=2) + "\n").encode()
        (self.root / "manifest.json").write_bytes(self.manifest_bytes)
        self.original = plan([7, 4, 2])

        def live_repair(cars, q, cap, original):
            original.clear()  # Prove the evidence was copied before this mutation.
            original.update(plan([1, 4, 2]), policy_id="flexible-first-v1",
                            plan_changed=True, search={})
            return original

        self.namespace = load_definitions(ROOT / "src/nexus_model/engine.py", {"Engine"}, dict(
            Path=Path, json=json, hashlib=hashlib, itertools=itertools, np=np,
            lgb=types.SimpleNamespace(Booster=Mock(side_effect=AssertionError("no model files"))),
            base=base, team=types.SimpleNamespace(
                class_row=lambda d, i: np.zeros(1), calibrate=team.calibrate,
                cap=team.cap, three=team.three),
            v2=types.SimpleNamespace(judge_row=lambda d, i: np.zeros(1),
                names=lambda k: ["specialist_query_category"],
                features=lambda d, i, cars, first, second: np.zeros((len(cars), 1))),
            optimizer=types.SimpleNamespace(select=lambda *a: plan([7, 1, 4])),
            pair=types.SimpleNamespace(select=lambda *a: copy.deepcopy(self.original)),
            open_formation=types.SimpleNamespace(select=live_repair),
            from_inference=from_inference,
            from_joint=lambda cars, q, tickets: from_joint(
                cars, q, tickets, annotated_at=fixtures.NOW.isoformat())))
        self.engine = self.namespace["Engine"](self.root)
        judge = types.SimpleNamespace(predict=lambda *a, **k: np.array([[.8, .1, .05, .05]]))
        head = types.SimpleNamespace(predict=lambda matrix, **k: np.arange(1., len(matrix) + 1))
        self.engine.models = {"team_judge-1": judge, "v2_head-1": judge,
                              **{f"v2_head{k}": head for k in range(3)}}

    def predict(self):
        return self.engine.predict(dict(rows=[dict(cars=[7, 1, 4, 2])]))

    def test_exact_q_and_pair_plan_survive_live_repair(self):
        result = self.predict()
        evidence = result["replay_evidence"]
        self.assertTrue(verify(evidence))
        self.assertEqual(evidence["model_manifest_sha256"], hashlib.sha256(self.manifest_bytes).hexdigest())
        self.assertEqual(evidence["cars"], [7, 1, 4, 2])
        self.assertEqual(evidence["q"], result["q"])
        self.assertEqual(len(evidence["q"]), 24)
        self.assertEqual(evidence["original_formations"], self.original["blocks"])
        self.assertNotEqual(evidence["original_tickets"], result["tickets"])
        self.assertEqual(result["tickets"], sorted("-".join(map(str, t)) for t in plan([1, 4, 2])["tickets"]))
        result["q"][0] = 0
        self.assertTrue(verify(evidence))

    def test_evidence_failure_preserves_every_existing_prediction_field(self):
        expected = self.predict()
        expected.pop("replay_evidence")
        self.namespace["from_inference"] = Mock(side_effect=ValueError("evidence unavailable"))
        actual = self.predict()
        self.assertEqual(actual.pop("replay_evidence_error"), "ValueError: evidence unavailable")
        self.assertEqual(actual, expected)


class PublicationFixture(unittest.TestCase):
    def setUp(self):
        self.saved = fixtures.SavedInputsTests()
        self.saved.setUp()
        self.addCleanup(self.saved.doCleanups)
        self.snap, _ = self.saved.snapshot()
        self.saved.schedule([self.snap])

    def prediction(self):
        return self.saved.live()["races"][0].get("prediction")

    def assert_bound(self, prediction):
        evidence = prediction["replay_evidence"]
        self.assertTrue(verify(evidence))
        self.assertEqual(evidence["race_id"], self.snap["meta"]["race_id"])
        self.assertEqual(evidence["snapshot_at"], prediction["captured_at"])
        self.assertEqual(evidence["input_captured_at"], self.snap["captured_at"])
        self.assertEqual(evidence["source_sha256"], self.snap["sha256"])
        self.assertEqual(evidence["feature_history_through"], "2026-10-18")
        self.assertEqual(evidence["evidence_kind"], "prospective_preclose_inference")
        self.assertEqual(len(evidence["q"]), 210)
        self.assertNotIn("q", prediction)
        self.assertLess(datetime.fromisoformat(evidence["snapshot_at"]),
                        datetime.fromisoformat(evidence["close_at"]))


class SavedPublisherReplayTests(PublicationFixture):
    def test_new_forecast_keeps_bound_full_q_in_day_ledger_and_feed(self):
        report = self.saved.run_model(EvidenceEngine)
        self.assertEqual(report["predicted"], 1)
        prediction = self.prediction()
        self.assert_bound(prediction)
        self.assertEqual(prediction["replay_evidence"]["input_sha256"], self.snap["input_sha256"])
        ledger = fixtures.capture.read_json(self.saved.root / "outputs/nexus_forecasts/2026-10-19.json")
        self.assertEqual(ledger[0]["prediction"], prediction)

    def test_legacy_forecast_is_not_regenerated_or_rebound_open_or_closed(self):
        self.saved.run_model(EvidenceEngine)
        expected = copy.deepcopy(self.prediction())
        expected.pop("replay_evidence")
        feed = self.saved.live()
        feed["races"][0]["prediction"] = expected
        fixtures.capture.save_json(self.saved.root / "outputs/nexus_live.json", feed)
        fixtures.capture.save_json(self.saved.root / "outputs/nexus_forecasts/2026-10-19.json", feed["races"])
        with patch.object(fixtures.publisher, "bind_prediction", side_effect=AssertionError("no backfill")) as bind:
            for advance in (1, 180):
                self.saved.clock.value += timedelta(minutes=advance)
                report = self.saved.run_model(EvidenceEngine)
                self.assertEqual(report["generated"], [])
                self.assertEqual(self.prediction(), expected)
            bind.assert_not_called()

    def test_binding_failure_keeps_forecast_and_reports_missing_evidence(self):
        with patch.object(fixtures.publisher, "bind_prediction", side_effect=ValueError("invalid evidence")):
            report = self.saved.run_model(EvidenceEngine)
        self.assertEqual(report["predicted"], 1)
        self.assertEqual(self.prediction()["tickets"], fixtures.prediction()["tickets"])
        self.assertEqual(self.prediction()["allocations"], {})
        self.assertNotIn("replay_evidence", self.prediction())
        self.assertTrue(any("invalid evidence" in error["reason"] for error in report["errors"]))

    def test_deadline_crossed_during_binding_discards_new_forecast(self):
        def slow_bind(*args, **kwargs):
            result = bind_prediction(*args, **kwargs)
            self.saved.clock.value += timedelta(hours=3)
            return result
        with patch.object(fixtures.publisher, "bind_prediction", side_effect=slow_bind):
            report = self.saved.run_model(EvidenceEngine)
        self.assertEqual(report["predicted"], 0)
        self.assertIsNone(self.prediction())


class SourcePublisherReplayTests(PublicationFixture):
    def setUp(self):
        super().setUp()
        model = self.saved.root / "models/nexus53"
        manifest = fixtures.capture.read_json(model / "manifest.json")
        manifest.update(train=["2023-10-09", "2025-10-08"], historical={}, live_policy={})
        fixtures.capture.save_json(model / "manifest.json", manifest)
        meta = dict(self.snap["meta"], session="morning", cancel=False)
        source = types.SimpleNamespace(now=self.saved.clock, JST=fixtures.capture.JST,
                    calendar=lambda *a: [meta], race=lambda *a: self.snap,
                    outcomes=lambda data: ({}, [], False, False))
        self.namespace = load_definitions(ROOT / "src/nexus_publish.py", {"read", "save", "iso", "run"}, dict(
            Path=Path, json=json, os=os, date=date, datetime=datetime, timedelta=timedelta,
            ThreadPoolExecutor=ThreadPoolExecutor, source=source,
            inputs=types.SimpleNamespace(History=types.SimpleNamespace(load=lambda *a: self.saved.history),
                                         build=fixtures.build),
            update_history=types.SimpleNamespace(update=lambda *a: None), Engine=EvidenceEngine,
            is_open_prediction=is_open_prediction, allocation_snapshot=allocation_snapshot,
            settle_prediction=settle_prediction, bind_prediction=bind_prediction,
            OUT=self.saved.root / "outputs", MODEL=model, STATE=self.saved.root / "data/nexus",
            CACHE=self.saved.root / "data/nexus/sources"))
        self.namespace["OUT"].mkdir(exist_ok=True)

    def test_new_source_forecast_is_bound_but_closed_forecast_is_unchanged(self):
        self.namespace["run"]()
        self.assert_bound(self.prediction())
        expected = copy.deepcopy(self.prediction())
        self.saved.clock.value += timedelta(hours=3)
        self.namespace["bind_prediction"] = Mock(side_effect=AssertionError("no backfill"))
        self.namespace["run"]()
        self.namespace["bind_prediction"].assert_not_called()
        self.assertEqual(self.prediction(), expected)

    def test_binding_failure_keeps_source_forecast(self):
        self.namespace["bind_prediction"] = Mock(side_effect=ValueError("invalid evidence"))
        self.namespace["run"]()
        self.assertEqual(self.prediction()["tickets"], fixtures.prediction()["tickets"])
        self.assertNotIn("replay_evidence", self.prediction())
        status = fixtures.capture.read_json(self.saved.root / "data/nexus/run_status.json")
        self.assertTrue(any("invalid evidence" in error["error"] for error in status["errors"]))

    def test_source_deadline_crossed_during_binding_discards_new_forecast(self):
        def slow_bind(*args, **kwargs):
            result = bind_prediction(*args, **kwargs)
            self.saved.clock.value += timedelta(hours=3)
            return result
        self.namespace["bind_prediction"] = slow_bind
        self.namespace["run"]()
        self.assertIsNone(self.prediction())


if __name__ == "__main__":
    unittest.main()
