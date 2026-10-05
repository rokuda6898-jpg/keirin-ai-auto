import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import commander_selector
import live_snapshot_store
import production_replay_backtest
import settle_results
import trifecta_reranker


class EngineV2Tests(unittest.TestCase):
    def test_production_replay_actual_trifecta(self):
        race = pd.DataFrame([
            {"car_no": 5, "finish_pos": 2},
            {"car_no": 1, "finish_pos": 1},
            {"car_no": 7, "finish_pos": 3},
            {"car_no": 2, "finish_pos": 4},
        ])
        self.assertEqual(production_replay_backtest.actual_trifecta(race), "1-5-7")

    def test_trifecta_reranker_feature_frame(self):
        race = pd.DataFrame([
            {
                "car_no": 1, "p_win": 0.5, "p_second": 0.3, "p_third": 0.2,
                "rider_strength": 10, "score": 90, "line_id": 1,
                "line_position": 1, "line_size": 2, "number_of_lines": 2,
                "entries_number": 3,
            },
            {
                "car_no": 2, "p_win": 0.3, "p_second": 0.4, "p_third": 0.3,
                "rider_strength": 9, "score": 88, "line_id": 1,
                "line_position": 2, "line_size": 2, "number_of_lines": 2,
                "entries_number": 3,
            },
            {
                "car_no": 3, "p_win": 0.2, "p_second": 0.3, "p_third": 0.5,
                "rider_strength": 8, "score": 85, "line_id": 2,
                "line_position": 1, "line_size": 1, "number_of_lines": 2,
                "entries_number": 3,
            },
        ])
        cand = pd.DataFrame([{"buy": "1-2-3", "prob": 0.12}])
        out = trifecta_reranker.build_candidate_features(race, cand)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["buy"], "1-2-3")
        self.assertEqual(float(out.iloc[0]["same_line_12"]), 1.0)
        self.assertEqual(float(out.iloc[0]["same_line_13"]), 0.0)
        self.assertGreater(float(out.iloc[0]["first_win_gap_second"]), 0)

    def test_commander_candidate_features_measure_disagreement(self):
        base = pd.Series({
            "predicted_win_prob": 0.50,
            "top1_top2_margin": 0.10,
            "second_pick_prob": 0.40,
            "second_pick_car_no": 2,
            "line_position": 1,
            "line_size": 3,
            "second_pick_line_position": 2,
            "race_attack_pressure": 6,
            "other_line_attack_pressure": 4,
        })
        variants = pd.DataFrame([
            {"variant": "production", "predicted_winner_car_no": 1},
            {"variant": "core_model", "predicted_winner_car_no": 1},
            {"variant": "second_wheel", "predicted_winner_car_no": 2},
        ])
        out = commander_selector.race_candidate_features(base, variants)
        second = out[out["variant"].eq("second_wheel")].iloc[0]
        self.assertEqual(float(second["differs_from_production"]), 1.0)
        self.assertEqual(float(second["candidate_is_second_pick"]), 1.0)
        self.assertEqual(float(second["variant_count"]), 3.0)
        self.assertEqual(float(second["unique_pick_count"]), 2.0)

    def test_snapshot_time_buckets(self):
        self.assertEqual(live_snapshot_store.snapshot_bucket(4 * 3600), "morning")
        self.assertEqual(live_snapshot_store.snapshot_bucket(40 * 60), "40m")
        self.assertEqual(live_snapshot_store.snapshot_bucket(20 * 60), "20m")
        self.assertEqual(live_snapshot_store.snapshot_bucket(10 * 60), "10m")
        self.assertIsNone(live_snapshot_store.snapshot_bucket(4 * 60))

    def test_trifecta_settlement_compares_variants(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger = root / "ledger.csv"
            results_csv = root / "results.csv"
            accuracy = root / "accuracy.json"

            rows = []
            for variant in ["production_top10", "trifecta_reranker"]:
                for rank in range(1, 11):
                    if variant == "production_top10":
                        buy = f"1-2-{rank}"
                    else:
                        buy = "9-8-7" if rank == 1 else f"2-3-{rank}"
                    rows.append({
                        "date": "2026-10-06",
                        "venue": "A",
                        "race_no": 1,
                        "race_id": "r1",
                        "variant": variant,
                        "ticket_rank": rank,
                        "buy": buy,
                        "prediction_created_at_jst": "2026-10-06T10:00:00+09:00",
                    })
            pd.DataFrame(rows).to_csv(ledger, index=False)

            official = pd.DataFrame([{
                "race_id": "r1",
                "actual_trifecta": "9-8-7",
                "official_result_available": True,
            }])

            old_ledger = settle_results.TRIFECTA_TOP10_LEDGER_CSV
            old_results = settle_results.TRIFECTA_TOP10_RESULTS_CSV
            old_accuracy = settle_results.TRIFECTA_TOP10_ACCURACY_JSON
            old_output = settle_results.OUTPUT_DIR
            try:
                settle_results.TRIFECTA_TOP10_LEDGER_CSV = ledger
                settle_results.TRIFECTA_TOP10_RESULTS_CSV = results_csv
                settle_results.TRIFECTA_TOP10_ACCURACY_JSON = accuracy
                settle_results.OUTPUT_DIR = root
                settle_results.update_trifecta_top10_accuracy(official)
            finally:
                settle_results.TRIFECTA_TOP10_LEDGER_CSV = old_ledger
                settle_results.TRIFECTA_TOP10_RESULTS_CSV = old_results
                settle_results.TRIFECTA_TOP10_ACCURACY_JSON = old_accuracy
                settle_results.OUTPUT_DIR = old_output

            payload = json.loads(accuracy.read_text(encoding="utf-8"))
            self.assertEqual(payload["races"], 1)
            self.assertEqual(payload["hits"], 0)
            by_variant = {x["variant"]: x for x in payload["variants"]}
            self.assertEqual(by_variant["production_top10"]["hits"], 0)
            self.assertEqual(by_variant["trifecta_reranker"]["hits"], 1)


    def test_final_trifecta_uses_latest_snapshot_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ledger = root / "prediction_ledger.csv"
            results_csv = root / "final_results.csv"
            accuracy = root / "final_accuracy.json"

            pd.DataFrame([
                {
                    "race_id": "r1", "date": "2026-10-06", "venue": "A", "race_no": 1,
                    "bet_type": "trifecta", "buy": "9-8-7", "candidate_rank": 1,
                    "prediction_created_at_jst": "2026-10-06T10:00:00+09:00",
                    "strategy_version": "old",
                },
                {
                    "race_id": "r1", "date": "2026-10-06", "venue": "A", "race_no": 1,
                    "bet_type": "trifecta", "buy": "1-2-3", "candidate_rank": 1,
                    "prediction_created_at_jst": "2026-10-06T10:20:00+09:00",
                    "strategy_version": "latest",
                },
            ]).to_csv(ledger, index=False)

            official = pd.DataFrame([{
                "race_id": "r1",
                "actual_trifecta": "9-8-7",
                "official_result_available": True,
            }])

            old_ledger = settle_results.PREDICTION_LEDGER_CSV
            old_results = settle_results.FINAL_TRIFECTA_RESULTS_CSV
            old_accuracy = settle_results.FINAL_TRIFECTA_ACCURACY_JSON
            try:
                settle_results.PREDICTION_LEDGER_CSV = ledger
                settle_results.FINAL_TRIFECTA_RESULTS_CSV = results_csv
                settle_results.FINAL_TRIFECTA_ACCURACY_JSON = accuracy
                settle_results.update_final_trifecta_ticket_accuracy(official)
            finally:
                settle_results.PREDICTION_LEDGER_CSV = old_ledger
                settle_results.FINAL_TRIFECTA_RESULTS_CSV = old_results
                settle_results.FINAL_TRIFECTA_ACCURACY_JSON = old_accuracy

            payload = json.loads(accuracy.read_text(encoding="utf-8"))
            self.assertEqual(payload["races"], 1)
            self.assertEqual(payload["hits"], 0)
            self.assertEqual(payload["definition"], "latest pre-race selected trifecta ticket set per race")


if __name__ == "__main__":
    unittest.main()
