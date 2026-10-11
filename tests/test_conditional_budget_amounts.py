from pathlib import Path
import shutil
import subprocess
import unittest

ROOT=Path(__file__).resolve().parents[1]

class AmountDisplayTests(unittest.TestCase):
    def test_ticket_amount_mapping_and_missing_states(self):
        if not shutil.which('node'):self.skipTest('Node unavailable')
        subprocess.run(['node',str(ROOT/'tests/test_conditional_budget_amounts.js')],check=True)
    def test_amount_assets_loaded_and_races_before_performance(self):
        html=(ROOT/'site/conditional.html').read_text(encoding='utf-8')
        self.assertIn('conditional-budget.css?v=20261011-ticket-amounts',html)
        self.assertIn('conditional-budget.js?v=20261011-ticket-amounts',html)
        self.assertLess(html.index('id="races"'),html.index('id="budget-panel"'))
        self.assertEqual(html.count('id="races"'),1)
    def test_pipeline_initializes_then_settles(self):
        text=(ROOT/'.github/workflows/conditional-live.yml').read_text(encoding='utf-8')
        self.assertLess(text.index('python src/conditional_budget_initial.py'),text.index('python src/conditional_live.py'))
        self.assertIn("src/conditional_budget*.py",text)
        self.assertIn('git add outputs/conditional_budget_initialization.json',text)

if __name__=='__main__':unittest.main()
