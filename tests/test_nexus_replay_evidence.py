import copy
import itertools
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nexus_model.replay_evidence import bind_prediction, from_inference, verify


def inference():
    # Deliberately not sorted, with an absent car, to detect q/car reordering.
    cars = [7, 1, 4]
    tickets = list(itertools.permutations(cars, 3))
    plan = dict(tickets=[list(t) for t in tickets],
                blocks=[dict(first=cars[:], second=cars[:], third=cars[:], count=6)])
    q = [1 / 21, 2 / 21, 3 / 21, 4 / 21, 5 / 21, 6 / 21]
    spec = dict(id="test-frozen-model", train=["2023-10-09", "2025-10-08"])
    return cars, q, plan, spec


def prediction():
    evidence = from_inference(*inference(), manifest_sha256="a" * 64)
    return dict(replay_evidence=evidence,
                captured_at="2026-10-20T09:00:00+09:00",
                input_captured_at="2026-10-20T08:59:00+09:00",
                history_through="2026-10-19", source_sha256="b" * 64,
                input_sha256="c" * 64)


def bind(p):
    return bind_prediction(p, race_id="041220261020", race_date="2026-10-20",
                           close_at="2026-10-20T09:27:00+09:00")


class ReplayEvidence(unittest.TestCase):
    def test_q_keeps_absence_and_original_car_order_without_rounding(self):
        cars, q, plan, spec = inference()
        evidence = from_inference(cars, q, plan, spec, "a" * 64)
        actual = dict(zip(itertools.permutations(evidence["cars"], 3), evidence["q"]))
        self.assertEqual(actual[(7, 1, 4)], 1 / 21)
        self.assertEqual(actual[(4, 1, 7)], 6 / 21)
        self.assertEqual(evidence["training_cutoff_exclusive"], "2025-10-09")
        plan["blocks"][0]["first"].clear()
        self.assertEqual(evidence["original_count"], 6)
        self.assertEqual(len(evidence["original_formations"][0]["first"]), 3)
        self.assertTrue(verify(evidence))

    def test_a_missing_probability_is_never_reconstructed(self):
        for bad in ([1 / 6] * 5, [0.1] * 6, [0.1] * 5 + [float("nan")],
                    [0.1] * 5 + [True], [-0.1, 0.3, 0.2, 0.2, 0.2, 0.2]):
            cars, _, plan, spec = inference()
            with self.subTest(q=bad), self.assertRaises(ValueError):
                from_inference(cars, bad, plan, spec, "a" * 64)

    def test_exact_formations_and_active_roster_are_required(self):
        cars, q, plan, spec = inference()
        for change in ("ticket", "duplicate", "block", "roster"):
            modified = copy.deepcopy(plan)
            roster = cars[:]
            if change == "ticket":
                modified["tickets"][0] = [7, 1, 2]
            elif change == "duplicate":
                modified["tickets"][-1] = modified["tickets"][0]
            elif change == "block":
                modified["blocks"][0]["first"] = [7]
            else:
                roster[0] = roster[1]
            with self.subTest(change=change), self.assertRaises(ValueError):
                from_inference(roster, q, modified, spec, "a" * 64)

    def test_preclose_evidence_binds_both_input_and_forecast_times(self):
        p = prediction()
        self.assertTrue(bind(p))
        record = p["replay_evidence"]
        self.assertTrue(verify(record))
        self.assertEqual(record["snapshot_at"], p["captured_at"])
        self.assertEqual(record["source_sha256"], p["source_sha256"])
        self.assertEqual(record["feature_history_through"], "2026-10-19")
        self.assertNotIn("outcome", record)
        with self.assertRaisesRegex(ValueError, "already bound"):
            bind(p)

    def test_deadline_future_input_and_today_history_are_rejected(self):
        for field, value in (("captured_at", "2026-10-20T09:27:00+09:00"),
                             ("input_captured_at", "2026-10-20T09:01:00+09:00"),
                             ("captured_at", "2026-10-19T09:00:00+09:00"),
                             ("history_through", "2026-10-20"),
                             ("source_sha256", "missing")):
            p = prediction()
            p[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                bind(p)

    def test_altered_scores_or_metadata_invalidate_evidence(self):
        original = prediction()
        bind(original)
        for field in ("q", "snapshot_at", "original_tickets", "model_manifest_sha256"):
            p = copy.deepcopy(original)
            record = p["replay_evidence"]
            if field == "q":
                record[field] = record[field][::-1]
            elif field == "original_tickets":
                record[field].pop()
            else:
                record[field] += "changed"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "hash mismatch"):
                verify(record)

    def test_old_forecasts_are_not_backfilled(self):
        p = dict(captured_at="2026-10-19T08:00:00+09:00", tickets=["1-2-3"])
        before = copy.deepcopy(p)
        self.assertFalse(bind(p))
        self.assertEqual(p, before)


if __name__ == "__main__":
    unittest.main()
