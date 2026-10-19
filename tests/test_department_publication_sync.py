import ast
import re
import tempfile
import unittest
from pathlib import Path

from site_retirement import apply_retirement

ROOT = Path(__file__).resolve().parents[1]


class DepartmentPublicationTests(unittest.TestCase):
    def test_company_hub_no_longer_offers_retired_forecast(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'company.html').write_text(
                '<a href="./#forecast"><h3>会議の最終予想・独立予想</h3>'
                '<p>それぞれの予想印と買い目を切り替え。</p></a>'
                '<a href="original/company/all_department_predictions.html">全部署</a>'
                '<script src="site-wide.js" data-root="./"></script>', encoding='utf-8')
            apply_retirement(root)
            text = (root/'company.html').read_text(encoding='utf-8')
            self.assertNotIn('独立予想', text)
            self.assertIn('会社の最終予想', text)
            self.assertIn('all_department_predictions.html', text)
            self.assertIn('site-wide.js?v=20261019-forecast', text)
            apply_retirement(root)
            self.assertEqual(text, (root/'company.html').read_text(encoding='utf-8'))

    def test_selected_model_publication_preserves_visible_incomplete_coverage(self):
        workflow = (ROOT/'.github/workflows/pages.yml').read_text(encoding='utf-8')
        trigger = re.search(r"(?m)^    workflows:\s*(\[[^\n]*\])\s*$", workflow)
        self.assertIsNotNone(trigger, 'workflow_run must list publishing workflows')
        workflows = ast.literal_eval(trigger.group(1))
        self.assertIn('conditional-trifecta-live', workflows)
        self.assertIn('nexus-formation-publish', workflows)
        self.assertIn("github.event.workflow_run.head_branch == 'main'", workflow)
        self.assertIn("contains(fromJSON('[\"success\", \"failure\"]'), github.event.workflow_run.conclusion)", workflow)
        self.assertIn('ref: main', workflow)
        self.assertIn('PUBLISHED_FORECAST_SNAPSHOT', workflow)
        self.assertIn('company/all_department_predictions.json', workflow)
        self.assertIn('conditional_live.json', workflow)


if __name__ == '__main__':
    unittest.main()

