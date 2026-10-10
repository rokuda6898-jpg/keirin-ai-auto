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
            self.assertIn('site-wide.js?v=20261011-retirement', text)
            apply_retirement(root)
            self.assertEqual(text, (root/'company.html').read_text(encoding='utf-8'))

    def test_detail_publication_is_triggered_only_by_successful_main_forecasts(self):
        workflow = (ROOT/'.github/workflows/pages.yml').read_text(encoding='utf-8')
        self.assertIn("workflows: ['daily-keirin-predict', 'intraday-keirin-odds']", workflow)
        self.assertIn("github.event.workflow_run.head_branch == 'main'", workflow)
        self.assertIn("github.event.workflow_run.conclusion == 'success'", workflow)
        self.assertIn('ref: main', workflow)
        self.assertIn('PUBLISHED_FORECAST_SNAPSHOT', workflow)
        self.assertIn('company/all_department_predictions.json', workflow)
        self.assertIn('conditional_live.json', workflow)


if __name__ == '__main__':
    unittest.main()
