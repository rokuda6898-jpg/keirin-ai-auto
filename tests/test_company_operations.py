import json
import tempfile
import unittest
from pathlib import Path
from company_operations import build_company_operations


class OperationsTests(unittest.TestCase):
    def test_missing_evidence_never_claims_validated_profit_or_promotes(self):
        with tempfile.TemporaryDirectory() as directory:
            report=build_company_operations(Path(directory))
            self.assertEqual(report['site_status'],'未確認')
            self.assertFalse(report['target_validated'])
            self.assertFalse(report['release_policy']['automatic_promotion'])
            self.assertEqual(len(report['roles']),4)
            review = report['strategy_review']
            self.assertEqual(sum(review['weights'].values()), 100)
            self.assertFalse(review['optimality_validated'])
            self.assertEqual(len(review['agenda']), 3)
            self.assertIn('点数を増やさない', review['risk_budget_policy'])

    def test_future_quote_evidence_flags_data_quality(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'company').mkdir()
            (root/'company/validation_coverage.json').write_text(json.dumps({'time_checks':{'quote_after_snapshot_or_close':1}}))
            self.assertEqual(build_company_operations(root)['roles'][0]['status'],'要確認')
