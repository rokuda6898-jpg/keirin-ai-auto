from pathlib import Path
import shutil
import subprocess
import unittest

class ConditionalBudgetUITests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node.js not installed')
    def test_budget_stats_and_browser_script_syntax(self):
        root=Path(__file__).resolve().parents[1]
        subprocess.run(['node','--check',str(root/'site/conditional-budget.js')], check=True)
        subprocess.run(['node',str(root/'tests/test_conditional_budget_ui.js')], check=True)
