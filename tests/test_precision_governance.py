import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

import cross_source_audit
import model_drift_audit
import settle_results
import train_recency_challenger


class PrecisionGovernanceTests(unittest.TestCase):
    def test_recency_weights_halve_at_half_life(self):
        newest = pd.Timestamp("2026-10-01")
        dates = pd.Series([newest, newest - pd.Timedelta(days=120)])
        weights = train_recency_challenger.recency_weights(dates, 120)
        self.assertAlmostEqual(float(weights[0]), 1.0, places=6)
        self.assertAlmostEqual(float(weights[1]), 0.5, places=6)

    def test_live_drift_flags_large_distribution_shift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            baseline_path = root / "baseline.json"
            output_path = root / "drift.json"

            train_x = pd.DataFrame({
                "feature_a": list(range(100)),
                "feature_b": [1.0] * 100,
            })
            source = train_x.copy()
            model_drift_audit.build_feature_baseline(
                train_x, source, path=baseline_path
            )

            live_x = pd.DataFrame({
                "feature_a": [1000.0] * 30,
                "feature_b": [1.0] * 30,
            })
            payload = model_drift_audit.audit_live_drift(
                live_x, live_x.copy(),
                baseline_path=baseline_path,
                output_path=output_path,
            )
            self.assertIn(payload["status"], {"amber", "red"})
            flagged = {x["feature"] for x in payload["top_flags"]}
            self.assertIn("feature_a", flagged)

    def test_trifecta_top10_accuracy_tracks_exact_ticket_hit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger_path = root / "ledger.csv"
            results_path = root / "results.csv"
            accuracy_path = root / "accuracy.json"

            rows = []
            for race_id in ["r1", "r2"]:
                for rank in range(1, 11):
                    rows.append({
                        "date": "2026-10-06",
                        "venue": "A",
                        "race_no": 1,
                        "race_id": race_id,
                        "ticket_rank": rank,
                        "buy": f"1-2-{rank}",
                        "prediction_created_at_jst": "2026-10-06T10:00:00+09:00",
                    })
            pd.DataFrame(rows).to_csv(ledger_path, index=False)

            official = pd.DataFrame([
                {
                    "race_id": "r1",
                    "actual_trifecta": "1-2-5",
                    "official_result_available": True,
                },
                {
                    "race_id": "r2",
                    "actual_trifecta": "9-8-7",
                    "official_result_available": True,
                },
            ])

            old_ledger = settle_results.TRIFECTA_TOP10_LEDGER_CSV
            old_results = settle_results.TRIFECTA_TOP10_RESULTS_CSV
            old_accuracy = settle_results.TRIFECTA_TOP10_ACCURACY_JSON
            try:
                settle_results.TRIFECTA_TOP10_LEDGER_CSV = ledger_path
                settle_results.TRIFECTA_TOP10_RESULTS_CSV = results_path
                settle_results.TRIFECTA_TOP10_ACCURACY_JSON = accuracy_path
                settle_results.update_trifecta_top10_accuracy(official)
            finally:
                settle_results.TRIFECTA_TOP10_LEDGER_CSV = old_ledger
                settle_results.TRIFECTA_TOP10_RESULTS_CSV = old_results
                settle_results.TRIFECTA_TOP10_ACCURACY_JSON = old_accuracy

            payload = json.loads(accuracy_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["races"], 2)
            self.assertEqual(payload["hits"], 1)
            self.assertAlmostEqual(payload["hit_rate"], 0.5)

    def test_cross_source_race_id_conversion_and_expected_top3(self):
        url = cross_source_audit.netkeirin_url_from_race_id("016320261005")
        self.assertEqual(
            url,
            "https://keirin.netkeiba.com/race/result/?race_id=202610056301",
        )

        group = pd.DataFrame([
            {"car_no": 4, "finish_pos": 2},
            {"car_no": 1, "finish_pos": 1},
            {"car_no": 7, "finish_pos": 3},
            {"car_no": 3, "finish_pos": 4},
        ])
        self.assertEqual(cross_source_audit.expected_top3(group), "1-4-7")

    def test_cross_source_summary_counts_only_comparable_agreement(self):
        results = pd.DataFrame([
            {"status": "match"},
            {"status": "match"},
            {"status": "mismatch"},
            {"status": "unavailable"},
            {"status": "error"},
        ])
        payload = cross_source_audit.summarize(results)
        self.assertEqual(payload["matched_races"], 2)
        self.assertEqual(payload["mismatched_races"], 1)
        self.assertAlmostEqual(payload["agreement_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
