"""Synthetic four-race fixtures are not annual forecasts or performance evidence."""
import copy
from dataclasses import replace
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import nexus_roi_research as roi


class NexusAnnualResearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ledger = self.root / "synthetic-only.jsonl"
        self.cohort = self.root / "cohort.json"
        self.manifest = self.root / "manifest.json"
        self.run_dir = self.root / "run"
        self.spec = replace(roi.ANNUAL, benchmark_id="unit-test-only-four-races",
                            races=4, hits=2, stake_yen=2400, payout_yen=3100,
                            model_fingerprint=None)
        model = roi.read_json(roi.ROOT / "models" / "nexus53" / "manifest.json")
        model["historical"] = dict(races=4, hits=2, hit_rate=.5,
                                    flat_stake=2400, flat_return=3100, roi=3100/2400)
        self.manifest.write_text(json.dumps(model), encoding="utf-8")
        model_sha = roi.file_sha(self.manifest)
        cars = [1, 2, 3, 4, 5]
        triples = list(itertools.permutations(cars, 3))
        weights = [1000 - i if t[0] == 1 else 1 for i, t in enumerate(triples)]
        q = [v / sum(weights) for v in weights]
        box = dict(first=[1, 2, 3], second=[1, 2, 3], third=[1, 2, 3], count=6)
        original = [roi.ticket_text(t) for t in itertools.permutations([1, 2, 3], 3)]
        self.rows = []
        for day, payoff in [("2025-10-09", {"1-2-3": 1000}),
                            ("2026-04-08", {"5-4-3": 2400}),
                            ("2026-04-09", {"1-3-2": 1500}),
                            ("2026-10-08", {})]:
            self.rows.append(dict(schema=1, race_id="0112" + day.replace("-", ""), date=day,
                                  cars=cars, q=q, original_tickets=original,
                                  original_formations=[box], model_id=roi.MODEL_ID,
                                  model_manifest_sha256=model_sha,
                                  training_cutoff_exclusive="2025-10-09",
                                  outcome=dict(status="settled" if payoff else "refunded",
                                               payouts_per_100_yen=payoff,
                                               refund_tickets=[], refund_all=not payoff)))
        self.rows = copy.deepcopy(self.rows)
        cohort = dict(schema=1, benchmark_id=self.spec.benchmark_id,
                      model_manifest_sha256=model_sha,
                      source_description="Unit-test-only fixture; never annual evidence",
                      source_sha256="b" * 64, race_ids=[r["race_id"] for r in self.rows])
        self.cohort.write_text(json.dumps(cohort), encoding="utf-8")
        self.save_rows()

    def save_rows(self):
        self.ledger.write_text("".join(json.dumps(r) + "\n" for r in self.rows), encoding="utf-8")

    def audit(self, **kwargs):
        return roi.audit(self.ledger, self.cohort, self.manifest, spec=kwargs.get("spec", self.spec))

    def prepare(self):
        return roi.prepare(self.ledger, self.cohort, self.manifest, self.run_dir, spec=self.spec)

    def assert_blocked(self, code, callback):
        with self.assertRaises(roi.ReplayError) as caught:
            callback()
        self.assertEqual(code, caught.exception.code)

    def test_exact_baseline_and_refund_100yen_accounting(self):
        races, report = self.audit()
        self.assertEqual((4, 2, 24, 2400, 3100), tuple(report["baseline"][k] for k in
                         ("races", "hits", "tickets", "stake_yen", "payout_yen")))
        self.assertEqual(600, report["baseline"]["refund_yen"])
        self.assertEqual(4, report["historical_timing_unverified_races"])
        self.assertFalse(report["historical_input_availability_proven"])
        self.assertEqual(0, roi.settle(races[-1], races[-1].original)["hits"])

    def test_baseline_totals_must_match_before_research(self):
        self.assert_blocked("benchmark_mismatch", lambda: self.audit(spec=replace(self.spec, payout_yen=3200)))

    def test_full_q_missing_does_not_select_a_complete_subset(self):
        self.rows[-1]["q"] = None
        self.save_rows()
        self.assertEqual(3, self.audit()[1]["full_q_races"])
        self.assert_blocked("full_q_required", self.prepare)
        self.assertFalse(self.run_dir.exists())

    def test_missing_or_extra_race_stops_even_before_aggregate(self):
        self.rows.pop()
        self.save_rows()
        self.assert_blocked("cohort_mismatch", self.audit)

    def test_duplicate_race_is_not_silently_deduplicated(self):
        self.rows.append(copy.deepcopy(self.rows[0]))
        self.save_rows()
        self.assert_blocked("duplicate_race", self.audit)

    def test_identity_and_period_cannot_be_changed(self):
        self.rows[0]["date"] = "2025-10-10"
        self.save_rows()
        self.assert_blocked("race_identity", self.audit)
        self.rows[0]["date"] = "2025-10-08"
        self.rows[0]["race_id"] = "011220251008"
        self.save_rows()
        self.assert_blocked("outside_period", self.audit)

    def test_missing_result_blocks_all_races(self):
        self.rows[-1]["outcome"]["status"] = "pending"
        self.save_rows()
        self.assert_blocked("missing_result", self.prepare)

    def test_exact_formations_and_no_double_stake(self):
        self.rows[0]["original_formations"][0]["count"] = 5
        self.save_rows()
        self.assert_blocked("formation_expansion", self.audit)
        self.rows[0]["original_formations"][0]["count"] = 6
        self.rows[0]["original_formations"] *= 2
        self.save_rows()
        self.assert_blocked("formation_overlap", self.audit)

    def test_model_hash_and_fixed_annual_training_cutoff(self):
        self.rows[-1]["training_cutoff_exclusive"] = "2026-04-09"
        self.save_rows()
        self.assert_blocked("training_leakage", self.audit)
        self.rows[-1]["training_cutoff_exclusive"] = "2025-10-09"
        self.rows[-1]["model_manifest_sha256"] = "a" * 64
        self.save_rows()
        self.assert_blocked("model_hash_mismatch", self.audit)

    def test_declared_history_and_saved_forecast_timing(self):
        self.rows[0]["feature_history_through"] = "2025-10-09"
        self.save_rows()
        self.assert_blocked("history_leakage", self.audit)
        self.rows[0]["feature_history_through"] = "2025-10-08"
        self.rows[0]["snapshot_at"] = "2025-10-09T09:01:00+09:00"
        self.rows[0]["close_at"] = "2025-10-09T09:00:00+09:00"
        self.save_rows()
        self.assert_blocked("forecast_timing", self.audit)

    def test_full_q_shape_mass_and_nonfinite_values(self):
        self.rows[0]["q"] = [1.0]
        self.save_rows()
        self.assert_blocked("incomplete_q", self.audit)
        self.rows[0]["q"] = [float("nan")] * 60
        self.save_rows()
        self.assert_blocked("incomplete_q", self.audit)

    def test_input_capture_must_precede_saved_forecast(self):
        raw = self.rows[0]
        raw.update(snapshot_at="2025-10-09T09:00:00+09:00", close_at="2025-10-09T09:30:00+09:00",
                   feature_history_through="2025-10-08", input_captured_at="2025-10-09T09:40:00+09:00")
        self.save_rows()
        self.assert_blocked("input_timing", self.audit)
        raw["input_captured_at"] = "2025-10-09T08:59:00+09:00"
        self.save_rows()
        self.assertFalse(self.audit()[0][0].timing_unverified)
        raw.pop("input_captured_at")
        self.save_rows()
        self.assertTrue(self.audit()[0][0].timing_unverified)

    def test_dead_heats_and_partial_refunds_are_separate(self):
        raw = copy.deepcopy(self.rows[0])
        raw["outcome"] = dict(status="settled", payouts_per_100_yen={"1-2-3": 1000, "1-3-2": 2000},
                              refund_tickets=["2-1-3"], refund_all=False)
        race = roi._parse_race(raw, roi.file_sha(self.manifest), self.spec)
        result = roi.settle(race, race.original)
        self.assertEqual((1, 600, 3100, 100), tuple(result[k] for k in ("hits", "stake_yen", "payout_yen", "refund_yen")))
        raw["outcome"]["refund_tickets"] = ["1-2-3"]
        self.assert_blocked("refund_conflict", lambda: roi._parse_race(raw, roi.file_sha(self.manifest), self.spec))

    def test_frozen_input_mutation_stops_research(self):
        self.prepare()
        self.rows[0]["q"] = list(reversed(self.rows[0]["q"]))
        self.save_rows()
        self.assert_blocked("source_changed", lambda: roi.develop(self.run_dir, spec=self.spec))

    def test_staged_selection_only_uses_development_then_one_candidate(self):
        original_bytes = self.ledger.read_bytes()
        calls = []
        def tracked(cars, q, points):
            self.assertIsInstance(cars, tuple)
            self.assertIsInstance(q, tuple)
            calls.append(points)
            return roi.select_tickets(cars, q, points)
        self.prepare()
        dev = roi.develop(self.run_dir, spec=self.spec, selector=tracked)
        self.assertEqual([6, 6, 8, 8, 10, 10, 12, 12], calls)
        self.assertFalse(dev["chronological_candidate_results_computed"])
        self.assertEqual(2, dev["baseline"]["races"])
        self.assert_blocked("development_locked", lambda: roi.develop(self.run_dir, spec=self.spec))
        result = roi.validate(self.run_dir, spec=self.spec, selector=tracked)
        self.assertEqual([dev["selected_candidate"]["points"]] * 2, calls[-2:])
        self.assertEqual(4, result["annual"]["candidate"]["races"])
        self.assertEqual(2, result["chronological"]["candidate"]["races"])
        self.assertEqual(3, result["sensitivity"]["comparison"]["candidate"]["races"])
        self.assertFalse(result["untouched_holdout"])
        self.assertFalse(result["live_policy_changed"])
        self.assertEqual(original_bytes, self.ledger.read_bytes())
        before, n_calls = (self.run_dir / "validation.json").read_bytes(), len(calls)
        reused = roi.validate(self.run_dir, spec=self.spec, selector=tracked)
        self.assertTrue(reused["reused_completed_result"])
        self.assertEqual(n_calls, len(calls))
        self.assertEqual(before, (self.run_dir / "validation.json").read_bytes())

    def test_identical_interrupted_validation_can_resume(self):
        protocol = self.prepare()
        roi.develop(self.run_dir, spec=self.spec)
        selection = roi.read_json(self.run_dir / "selection.json")
        roi._write_once(self.run_dir / "validation_started.json",
                        dict(protocol_sha256=protocol["protocol_sha256"], selection_sha256=selection["selection_sha256"]))
        later = [r for r in self.audit()[0] if r.day > self.spec.development_last]
        predicted = roi._evaluate(later, selection["selected_candidate"])
        path = self.run_dir / "validation_predictions.jsonl"
        roi._write_once(path, predicted, jsonl=True)
        before = path.read_bytes()
        report = roi.validate(self.run_dir, spec=self.spec)
        self.assertEqual(2, report["chronological"]["candidate"]["races"])
        self.assertEqual(before, path.read_bytes())

    def test_changed_interrupted_prediction_is_not_replaced(self):
        self.prepare()
        roi.develop(self.run_dir, spec=self.spec)
        roi._write_once(self.run_dir / "validation_predictions.jsonl", [{"changed": True}], jsonl=True)
        self.assert_blocked("validation_predictions_changed", lambda: roi.validate(self.run_dir, spec=self.spec))
        self.assertFalse((self.run_dir / "validation.json").exists())

    def test_formations_exact_at_all_four_frozen_point_counts(self):
        race = self.audit()[0][0]
        for points in roi.POINTS:
            with self.subTest(points=points):
                tickets, blocks = roi.select_tickets(race.cars, race.q, points)
                self.assertEqual(points, len(tickets))
                self.assertEqual(tickets, roi.expand_formations(blocks, race.cars))
                self.assertLessEqual(len(blocks), 4)

    def test_changed_development_selection_cannot_open_validation(self):
        self.prepare()
        roi.develop(self.run_dir, spec=self.spec)
        path = self.run_dir / "selection.json"
        obj = roi.read_json(path)
        obj["selected_candidate"] = roi.candidate_specs()[-1]
        path.write_text(json.dumps(obj), encoding="utf-8")
        self.assert_blocked("selection_changed", lambda: roi.validate(self.run_dir, spec=self.spec))

    def test_production_cli_cannot_treat_small_fixture_as_annual(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = roi.main(["audit", "--ledger", str(self.ledger), "--cohort", str(self.cohort),
                             "--model-manifest", str(self.manifest)])
        self.assertEqual(2, code)
        message = json.loads(err.getvalue())
        self.assertEqual("blocked", message["status"])
        self.assertFalse(message["annual_candidate_result_generated"])
        self.assertEqual("", out.getvalue())

    def test_real_model_reference_and_no_data_preflight(self):
        model = roi._model_info(roi.ROOT / "models" / "nexus53" / "manifest.json", roi.ANNUAL)
        self.assertEqual(roi.MODEL_FINGERPRINT, model["frozen_model_fingerprint"])
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = roi.main(["prepare", "--ledger", str(self.root / "absent.jsonl"),
                             "--cohort", str(self.root / "absent-cohort.json"),
                             "--output-dir", str(self.run_dir)])
        self.assertEqual(2, code)
        self.assertEqual("input_unreadable", json.loads(err.getvalue())["error"])
        self.assertFalse(self.run_dir.exists())


if __name__ == "__main__":
    unittest.main()
