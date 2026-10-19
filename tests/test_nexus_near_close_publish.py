"""A failed comparison must preserve the already published forecast evidence."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from nexus_input_snapshots import read_json, save_json

# Exercise the production integration function without loading ML assets.
source = Path(__file__).resolve().parents[1] / 'src/nexus_publish_saved_inputs.py'
tree = ast.parse(source.read_text())
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == 'update_near_close_comparison')
namespace = dict(Path=Path, subprocess=subprocess, sys=sys,
                 read_json=read_json, save_json=save_json)
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
update = namespace['update_near_close_comparison']


class PublicationFailureIsolation(unittest.TestCase):
    def exercise_failure(self, baseline):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / 'outputs/conditional_near_close.json'
            canonical = root / 'outputs/nexus_live.json'
            save_json(canonical, {'prediction': 'already preserved'})
            original = canonical.read_bytes()
            if baseline:
                save_json(plan_path, baseline)

            def failed_process(*args, **kwargs):
                save_json(plan_path, {'races': [{'id': 'unpublished partial write'}]})
                raise subprocess.CalledProcessError(1, args[0])

            with patch.object(subprocess, 'run', side_effect=failed_process):
                healthy, changed = update(root, baseline, generate=True)
            self.assertFalse(healthy)
            self.assertEqual(changed, ['comparison_error'])
            self.assertEqual(canonical.read_bytes(), original)
            self.assertEqual(read_json(plan_path, {}), baseline)
            self.assertEqual(read_json(root / 'outputs/nexus_coverage_status.json')
                             ['near_close_comparison']['status'], 'error')

    def test_partial_write_restores_public_baseline(self):
        self.exercise_failure({'schema': 1, 'races': [{'id': 'saved allocation'}]})

    def test_first_failed_attempt_does_not_publish_partial_file(self):
        self.exercise_failure({})

    def test_successful_gate_keeps_existing_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = {'schema': 1, 'races': []}
            save_json(root / 'outputs/conditional_near_close.json', baseline)
            healthy, late = update(root, baseline)
            self.assertTrue(healthy)
            self.assertEqual(late, [])
            self.assertEqual(read_json(root / 'outputs/conditional_near_close.json'), baseline)


if __name__ == '__main__':
    unittest.main()
