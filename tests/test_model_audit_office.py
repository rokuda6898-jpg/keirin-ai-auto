import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

import model_audit_office
import predict


class ModelAuditOfficeTests(unittest.TestCase):
    def test_same_race_challenger_can_reach_external_validation_gate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            company = root / "company"
            company.mkdir()
            variants = root / "variants.csv"
            misses = root / "misses.csv"
            validation = root / "v4.json"

            rows = []
            for i in range(320):
                race_id = f"r{i:03d}"
                actual = 1
                production_pick = 1 if i < 160 else 2
                if i < 10:
                    challenger_pick = 2  # 10 losses
                elif i < 160:
                    challenger_pick = 1
                elif i < 190:
                    challenger_pick = 1  # 30 gains
                else:
                    challenger_pick = 2
                for variant, pick in [
                    ("production", production_pick),
                    ("second_wheel", challenger_pick),
                ]:
                    rows.append({
                        "race_id": race_id,
                        "variant": variant,
                        "predicted_winner_car_no": pick,
                        "actual_winner_car_no": actual,
                        "official_result_available": True,
                    })
            pd.DataFrame(rows).to_csv(variants, index=False)

            miss_rows = []
            for i in range(100):
                miss_rows.append({
                    "race_id": f"m{i:03d}",
                    "predicted_winner_car_no": 1,
                    "second_pick_car_no": 2,
                    "actual_winner_car_no": 2 if i < 35 else 1,
                    "top1_top2_margin": 0.02 if i < 50 else 0.20,
                    "official_result_available": True,
                })
            pd.DataFrame(miss_rows).to_csv(misses, index=False)
            validation.write_text(json.dumps({
                "top1_final": {"rate": 0.55},
                "top1_core": {"rate": 0.58},
            }), encoding="utf-8")

            payload = model_audit_office.build_model_audit(
                output_dir=root,
                company_dir=company,
                variant_path=variants,
                miss_path=misses,
                validation_summary_path=validation,
            )

            self.assertFalse(payload["audit"]["comparison_context"]["accuracy_decline_established"])
            eligible = payload["promotion_board"]["eligible_for_external_validation"]
            self.assertEqual(len(eligible), 1)
            self.assertEqual(eligible[0]["variant"], "second_wheel")
            self.assertEqual(eligible[0]["changed_races"], 40)
            self.assertEqual(eligible[0]["net_hits"], 20)
            self.assertGreater(eligible[0]["delta_pp"], 1.5)
            self.assertTrue((company / "model_audit_office.json").exists())
            self.assertTrue((company / "department_league.json").exists())
            self.assertTrue((company / "promotion_board.json").exists())

    def test_policy_shadow_variants_use_previous_only_threshold(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            company = root / "company"
            company.mkdir()
            (company / "second_pick_specialist.json").write_text(
                json.dumps({
                    "best_shadow_rule": {
                        "margin_threshold": 0.15,
                        "status": "shadow_candidate_only",
                    }
                }),
                encoding="utf-8",
            )

            pred = pd.DataFrame([
                {
                    "date": "2026-10-06", "venue": "A", "race_no": 1, "race_id": "r1",
                    "start_at": 1, "close_at": 1, "car_no": 1, "player_id": "p1",
                    "p_win": 0.60, "p_core": 0.55,
                },
                {
                    "date": "2026-10-06", "venue": "A", "race_no": 1, "race_id": "r1",
                    "start_at": 1, "close_at": 1, "car_no": 2, "player_id": "p2",
                    "p_win": 0.50, "p_core": 0.45,
                },
                {
                    "date": "2026-10-06", "venue": "B", "race_no": 2, "race_id": "r2",
                    "start_at": 2, "close_at": 2, "car_no": 1, "player_id": "p3",
                    "p_win": 0.80, "p_core": 0.30,
                },
                {
                    "date": "2026-10-06", "venue": "B", "race_no": 2, "race_id": "r2",
                    "start_at": 2, "close_at": 2, "car_no": 2, "player_id": "p4",
                    "p_win": 0.30, "p_core": 0.70,
                },
            ])

            with mock.patch.object(predict, "OUTPUT_DIR", root):
                shadow = predict.build_policy_shadow_variants(pred)

            reversal = shadow[shadow["variant"].eq("second_pick_reversal_live")].set_index("race_id")
            router = shadow[shadow["variant"].eq("difficulty_router")].set_index("race_id")

            self.assertEqual(int(reversal.loc["r1", "car_no"]), 2)
            self.assertEqual(int(reversal.loc["r2", "car_no"]), 1)
            self.assertEqual(int(router.loc["r1", "car_no"]), 1)
            self.assertEqual(int(router.loc["r2", "car_no"]), 2)


if __name__ == "__main__":
    unittest.main()
