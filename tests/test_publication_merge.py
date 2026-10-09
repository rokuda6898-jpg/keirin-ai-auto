import subprocess
import tempfile
import unittest
from pathlib import Path
from publication_merge import reconcile, union_lines


class PublicationMergeTests(unittest.TestCase):
    def test_union_preserves_exact_evidence(self):
        self.assertEqual(union_lines(b'{"id":1}\n', b'{"id":1}\n{"id":2}\n'),
                         b'{"id":1}\n{"id":2}\n')

    def exercise(self, rejected=False, code_change=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            def git(*args):
                return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.DEVNULL)
            def write(name, value):
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(value, encoding='utf-8')
            def commit(message):
                git('add', '.')
                git('commit', '-m', message)
            git('init')
            git('config', 'user.name', 'Test')
            git('config', 'user.email', 'test@example.com')
            ledger = 'outputs/company/company_decision_ledger.jsonl'
            write(ledger, '{"id":0}\n')
            write('outputs/latest_results.json', 'old')
            commit('base')
            git('branch', 'remote')
            write(ledger, '{"id":0}\n{"id":1}\n')
            write('outputs/latest_results.json', 'local')
            commit('local forecast')
            original = git('rev-parse', 'HEAD').strip()
            git('checkout', 'remote')
            write(ledger, '{"id":0}\n{"id":2}\n')
            write('outputs/latest_results.json', 'remote')
            if code_change:
                write('src/model.py', 'changed')
            commit('remote result')
            git('update-ref', 'refs/remotes/origin/main', 'HEAD')
            git('checkout', '--detach', original.decode())
            ok = reconcile(root, lambda _: ['invalid'] if rejected else [])
            if rejected or code_change:
                self.assertFalse(ok)
                self.assertEqual(git('rev-parse', 'HEAD').strip(), original)
                self.assertEqual((root/'outputs/latest_results.json').read_text(), 'local')
            else:
                self.assertTrue(ok)
                self.assertEqual(set((root/ledger).read_text().splitlines()),
                                 {'{"id":0}', '{"id":1}', '{"id":2}'})
                self.assertEqual((root/'outputs/latest_results.json').read_text(), 'remote')
            self.assertFalse(git('status', '--porcelain').strip())

    def test_concurrent_predictions_preserved_and_newer_results_win(self):
        self.exercise()

    def test_invalid_release_rolls_back(self):
        self.exercise(rejected=True)

    def test_code_change_requires_regeneration(self):
        self.exercise(code_change=True)
