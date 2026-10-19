"""A Git retry must reuse evidence and retain newer forecasts without HTTP."""
import ast
from copy import deepcopy
import contextlib
from datetime import datetime
import io
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:
    import requests
except ModuleNotFoundError:
    requests = types.ModuleType("requests")
    requests.get = Mock(side_effect=AssertionError("HTTP forbidden in retry"))
    sys.modules["requests"] = requests

import quick_results_retry as retry
from nexus_input_snapshots import SNAPSHOTS, read_json, save_json
import nexus_publish_saved_inputs

DAY = "2026-10-19"
RID = "016220261019"
EARLY = "2026-10-19T20:30:00+09:00"
LATE = "2026-10-19T20:35:00+09:00"


def result(rid=RID, observed=EARLY, winners=("1-2-3",), payouts=None, official=True):
    return dict(race_id=rid, actual_trifecta="|".join(winners), actual_trifecta_buys=list(winners),
                payouts_trifecta_json=json.dumps({"1-2-3": 1200} if payouts is None else payouts),
                official_result_available=official, result_source="winticket",
                result_observed_at_jst=observed)


def offline_hooks(outputs):
    # Execute the actual two production hooks, without loading unrelated source
    # collectors. The real Nexus sync is imported by the production hook.
    module = types.ModuleType("settle_results")
    module.__dict__.update(OUTPUT_DIR=outputs, datetime=datetime, ZoneInfo=ZoneInfo,
                           json=json, np=np, pd=pd, re=re)
    tree = ast.parse((ROOT / "src/settle_results.py").read_text())
    names = {"patch_index_with_official_results", "sync_nexus_results_after_settlement"}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), "real_offline_settle_hooks", "exec"), module.__dict__)
    module.run_settlement = Mock(side_effect=AssertionError("full settlement forbidden"))
    module.fetch_results = Mock(side_effect=AssertionError("HTTP collection forbidden"))
    return module


class QuickResultsRetry(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root, self.cache = Path(temporary.name) / "repo", Path(temporary.name) / "retry"
        self.outputs = self.root / "outputs"
        self.outputs.mkdir(parents=True)
        self.hooks = offline_hooks(self.outputs)
        context = contextlib.ExitStack()
        self.addCleanup(context.close)
        context.enter_context(patch.dict(sys.modules, {"settle_results": self.hooks}))
        context.enter_context(patch.object(retry, "today_jst", return_value=DAY))
        context.enter_context(patch("requests.get", side_effect=AssertionError("HTTP forbidden")))
        context.enter_context(patch.object(nexus_publish_saved_inputs, "run", side_effect=AssertionError("inference forbidden")))
        context.enter_context(patch("nexus_model.publication.allocation_snapshot", side_effect=AssertionError("allocation forbidden")))
        context.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def write_results(self, rows):
        save_json(self.root / retry.RESULTS, rows)

    def test_newer_main_complete_result_and_other_races_survive(self):
        cached = result()
        self.write_results([cached])
        retry.snapshot(self.cache, self.root)
        newer = result(observed="2026-10-19T11:35:00+00:00", payouts={"1-2-3": 1800})
        extra = result("026220261019", observed=LATE)
        self.write_results([newer, extra])
        retry.replay(self.cache, self.root)
        self.assertEqual(read_json(self.root / retry.RESULTS), [newer, extra])

    def test_complete_payoff_never_regresses_to_newer_pending_or_missing_tie(self):
        complete = result(winners=("1-2-3", "2-1-3"), payouts={"1-2-3": 1200, "2-1-3": 2000})
        for incomplete in [result(observed=LATE, official=False),
                           result(observed=LATE, winners=("1-2-3", "2-1-3"), payouts={"1-2-3": 1200})]:
            merged, accepted = retry.merge_results([complete], [incomplete], DAY)
            self.assertEqual(merged, [complete])
            self.assertEqual(accepted, [])
        merged, _ = retry.merge_results([incomplete], [complete], DAY)
        self.assertEqual(merged, [complete])

    def test_newer_complete_cached_result_keeps_original_observation_time(self):
        current, cached = result(), result(observed=LATE, payouts={"1-2-3": 2400})
        merged, accepted = retry.merge_results([current], [cached], DAY)
        self.assertEqual(merged, [cached])
        self.assertEqual(accepted, [RID])
        tied, accepted = retry.merge_results([current], [result(payouts={"1-2-3": 900})], DAY)
        self.assertEqual(tied, [current])
        self.assertEqual(accepted, [])

    def test_replay_uses_latest_main_forecast_and_actual_offline_hooks(self):
        self.write_results([result()])
        save_json(self.outputs / "nexus_live.json", {"old_forecast": "must not enter cache"})
        retry.snapshot(self.cache, self.root)
        self.assertFalse((self.cache / "outputs/nexus_live.json").exists())
        p = dict(captured_at="2026-10-19T19:50:00+09:00", tickets=["1-2-3", "2-1-3"],
                 odds={"1-2-3": 12}, odds_at="2026-10-19T19:49:00+09:00",
                 allocations={"5000": {"available": True, "stakes": {"1-2-3": 5000}}},
                 allocation_saved_at="2026-10-19T19:50:00+09:00", auto_budget=5000)
        row = dict(race_id=RID, date=DAY, close_at="2026-10-19T20:00:00+09:00", prediction=p)
        live = dict(date=DAY, updated_at=LATE, policy_updated_at=EARLY, races=[row])
        save_json(self.outputs / "nexus_live.json", live)
        save_json(self.outputs / "nexus_forecasts" / (DAY + ".json"), [row])
        self.write_results([result(official=False, winners=(), payouts={})])
        html = '<h1>latest main forecast</h1><div class="race-result" data-result-for="' + RID + '">結果待ち</div>'
        (self.outputs / "index.html").write_text(html)
        retry.replay(self.cache, self.root)
        updated = read_json(self.outputs / "nexus_live.json")
        self.assertEqual(updated["races"][0]["prediction"], p)
        self.assertEqual(updated["updated_at"], live["updated_at"])
        self.assertEqual(updated["policy_updated_at"], live["policy_updated_at"])
        self.assertEqual(updated["races"][0]["settlement"]["flat"]["returned"], 1200)
        self.assertEqual(read_json(self.outputs / "nexus_forecasts" / (DAY + ".json"))[0]["prediction"], p)
        rendered = (self.outputs / "index.html").read_text()
        self.assertIn("latest main forecast", rendered)
        self.assertIn("確定 1-2-3", rendered)
        self.hooks.run_settlement.assert_not_called()
        self.hooks.fetch_results.assert_not_called()

    def test_raw_keeps_newer_main_capture_and_restores_new_cached_outcome(self):
        self.write_results([result()])
        relative = SNAPSHOTS / DAY / "outcomes" / (RID + ".json")
        another = SNAPSHOTS / DAY / "outcomes/026220261019.json"
        save_json(self.root / relative, {"captured_at": EARLY, "source": "cached"})
        save_json(self.root / another, {"captured_at": EARLY, "source": "second outcome"})
        retry.snapshot(self.cache, self.root)
        newer = {"captured_at": LATE, "source": "newer main"}
        save_json(self.root / relative, newer)
        (self.root / another).unlink()
        retry.replay(self.cache, self.root)
        self.assertEqual(read_json(self.root / relative), newer)
        self.assertEqual(read_json(self.root / another)["source"], "second outcome")

    def test_previous_day_cache_is_not_mixed_into_new_latest_feed(self):
        self.write_results([result()])
        retry.snapshot(self.cache, self.root)
        tomorrow = result("016220261020", observed="2026-10-20T10:30:00+09:00")
        self.write_results([tomorrow])
        with patch.object(retry, "today_jst", return_value="2026-10-20"):
            report = retry.replay(self.cache, self.root)
        self.assertEqual(read_json(self.root / retry.RESULTS), [tomorrow])
        self.assertEqual(report["skipped_reason"], "cached_result_date_is_not_current")

    def test_sync_failure_is_not_a_successful_retry(self):
        self.write_results([result()])
        retry.snapshot(self.cache, self.root)
        with patch.object(self.hooks, "sync_nexus_results_after_settlement", return_value={"status": "error", "error": "broken ledger"}):
            with self.assertRaisesRegex(RuntimeError, "broken ledger"):
                retry.replay(self.cache, self.root)

    def test_invalid_identity_fails_before_any_snapshot_copy(self):
        self.write_results([dict(result(), date="2026-10-20")])
        with self.assertRaisesRegex(ValueError, "identity"):
            retry.snapshot(self.cache, self.root)
        self.assertFalse((self.cache / retry.RESULTS).exists())


if __name__ == "__main__":
    unittest.main()
