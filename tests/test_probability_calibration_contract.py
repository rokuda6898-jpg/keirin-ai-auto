import unittest
from pathlib import Path


class ProbabilityCalibrationContractTests(unittest.TestCase):
    def test_production_and_fast_candidate_persist_calibrators(self):
        production = Path("src/train.py").read_text(encoding="utf-8")
        candidate = Path("src/train_top1_candidate.py").read_text(encoding="utf-8")

        self.assertIn("IsotonicRegression", production)
        self.assertIn('"calibrator": calibrator', production)
        self.assertIn("chronological_calibration_block_70_to_85pct", production)

        self.assertIn("IsotonicRegression", candidate)
        self.assertIn('"calibrator": calibrator', candidate)
        self.assertIn("latest_15pct_chronological_dates", candidate)
        self.assertNotIn('"calibrator": None', candidate)

    def test_broad_trifecta_shadow_can_recover_filtered_ticket(self):
        predict = Path("src/predict.py").read_text(encoding="utf-8")
        self.assertIn("make_trifecta_candidates(g, top_k_riders=None)", predict)
        self.assertIn("score_trifecta_reranker(g, broad_tri)", predict)
        self.assertIn('"reranker_broad_all"', predict)


if __name__ == "__main__":
    unittest.main()
