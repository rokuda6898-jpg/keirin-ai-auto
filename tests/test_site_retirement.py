import tempfile
import unittest
from pathlib import Path
from site_retirement import apply_retirement, remove_legacy_navigation


class SiteRetirementTests(unittest.TestCase):
    def test_only_legacy_product_navigation_removed(self):
        text = '<a href="independent.html"><span>旧独立</span></a><a href="conditional.html">条件付き</a><button id="shadow-mode">旧式</button><button id="company-mode">会社</button>'
        result = remove_legacy_navigation(text)
        self.assertNotIn('independent.html',result)
        self.assertNotIn('shadow-mode',result)
        self.assertIn('conditional.html',result)
        self.assertIn('company-mode',result)

    def test_nested_links_redirect_labels_and_history_preservation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); (root/'original/company').mkdir(parents=True)
            (root/'index.html').write_text('<a href="independent.html">旧式</a><script src="app.js?v=old"></script>',encoding='utf-8')
            (root/'conditional.html').write_text('<h1>条件付き3連単</h1><h2>独立予想</h2>',encoding='utf-8')
            (root/'original/company/report.html').write_text('<a href="../../independent.html">旧独立</a><p>過去の成績</p>',encoding='utf-8')
            (root/'app.js').write_text("const titles={shadow:'結果・成績｜独立予想'};",encoding='utf-8')
            history=root/'original/company/fusion_shadow_live_ledger.jsonl'
            history.write_text('historical evidence\n',encoding='utf-8')
            apply_retirement(root)
            self.assertNotIn('independent.html',(root/'index.html').read_text())
            self.assertIn('20261011-retirement',(root/'index.html').read_text())
            self.assertNotIn('独立予想',(root/'conditional.html').read_text())
            self.assertIn('条件付き3連単',(root/'app.js').read_text())
            self.assertIn('url=conditional.html',(root/'independent.html').read_text())
            self.assertIn('過去の成績',(root/'original/company/report.html').read_text())
            self.assertEqual(history.read_text(),'historical evidence\n')


if __name__=='__main__': unittest.main()
