"""Optional ROI output cannot replace public evidence or block canonical feeds."""
import ast
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from nexus_input_snapshots import read_json, save_json


def integration_namespace():
    source = ROOT / 'src/nexus_publish_saved_inputs.py'
    names = {'record_roi_comparison_error', 'update_roi_comparison', 'publish_ci'}
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(ROOT=ROOT, Path=Path, copy=copy, os=os, subprocess=subprocess,
                     sys=sys, tempfile=tempfile, read_json=read_json, save_json=save_json,
                     SNAPSHOTS=Path('data/raw/nexus_page_snapshots'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace


class RoiPublicationIsolation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.namespace = integration_namespace()
        self.update = self.namespace['update_roi_comparison']
        self.baseline = {'schema': 1, 'races': [{'id': 'public', 'shadow': {'tickets': ['1-2-3']}}]}
        self.path = self.root / 'outputs/conditional_roi.json'
        self.module = types.ModuleType('conditional_roi')

    def test_failed_generation_restores_public_baseline_and_canonical_files(self):
        for baseline in (self.baseline, {}):
            with self.subTest(baseline=bool(baseline)):
                canonical = self.root / 'outputs/nexus_live.json'
                conditional = self.root / 'outputs/conditional_live.json'
                save_json(canonical, {'original': 'nexus'})
                save_json(conditional, {'original': 'conditional'})
                original = (canonical.read_bytes(), conditional.read_bytes())
                if baseline:
                    save_json(self.path, baseline)

                def fail(*args, **kwargs):
                    save_json(self.path, {'races': ['partial unpublished write']})
                    raise subprocess.CalledProcessError(1, args[0])

                with patch.object(subprocess, 'run', side_effect=fail):
                    healthy, late = self.update(self.root, baseline, generate=True)
                self.assertFalse(healthy)
                self.assertEqual(late, ['comparison_error'])
                self.assertEqual(read_json(self.path, {}), baseline)
                self.assertEqual((canonical.read_bytes(), conditional.read_bytes()), original)
                self.assertEqual(read_json(self.root / 'outputs/nexus_coverage_status.json')
                                 ['roi_comparison']['status'], 'error')

    def test_failed_gate_restores_original_even_if_gate_mutates_its_baseline_argument(self):
        expected = copy.deepcopy(self.baseline)

        def fail(root, baseline, lead_seconds):
            self.assertEqual(lead_seconds, 60)
            baseline['races'].append({'id': 'invalid private mutation'})
            save_json(self.path, {'races': ['partial unpublished write']})
            raise ValueError('invalid comparison')

        self.module.gate_publication = fail
        with patch.dict(sys.modules, {'conditional_roi': self.module}):
            healthy, late = self.update(self.root, self.baseline)
        self.assertFalse(healthy)
        self.assertEqual(late, ['comparison_error'])
        self.assertEqual(self.baseline, expected)
        self.assertEqual(read_json(self.path), expected)

    def exercise_publish(self, fail_after_commit=False, invalid_baseline=False):
        state = {'events': [], 'gates': 0, 'committed': False}
        expected_baseline = copy.deepcopy(self.baseline)
        self.namespace['load_saved_rows'] = lambda root, feed: {}
        self.namespace['copy_newer_inputs'] = lambda source, destination: None
        self.namespace['gate_saved_publication'] = lambda *args: []
        self.namespace['update_near_close_comparison'] = lambda *args, **kwargs: (True, [])
        self.namespace['update_forward_comparison'] = lambda *args, **kwargs: (True, [])

        def gate(root, baseline, lead_seconds):
            state['events'].append('roi_gate')
            state['gates'] += 1
            self.assertEqual(lead_seconds, 60)
            self.assertEqual(baseline, expected_baseline)
            if fail_after_commit and state['committed']:
                save_json(root / 'outputs/conditional_roi.json', {'races': ['bad final gate write']})
                raise RuntimeError('late gate failed')
            return []

        self.module.gate_publication = gate

        def process(args, cwd=None, check=True, **kwargs):
            cwd = Path(cwd) if cwd else self.root
            if args[0] == 'git':
                if args[1:3] == ['worktree', 'add']:
                    checkout = Path(args[4])
                    checkout.mkdir(parents=True)
                    state['checkout'] = checkout
                    save_json(checkout / 'outputs/nexus_live.json', {'original': 'nexus'})
                    save_json(checkout / 'outputs/conditional_live.json', {'original': 'conditional'})
                    if invalid_baseline:
                        (checkout / 'outputs/conditional_roi.json').write_text('{invalid json')
                    else:
                        save_json(checkout / 'outputs/conditional_roi.json', expected_baseline)
                if args[1:3] == ['diff', '--cached']:
                    return types.SimpleNamespace(returncode=1)
                if 'commit' in args:
                    state['events'].append('amend' if '--amend' in args else 'commit')
                    state['committed'] = True
                if args[1] == 'add':
                    state['events'].append('git_add')
                    if '--all' in args and 'outputs/conditional_roi.json' in args:
                        state['roi_restage_after_gate'] = True
                if args[1] == 'push':
                    state['events'].append('push')
                    roi_path = cwd / 'outputs/conditional_roi.json'
                    if invalid_baseline:
                        self.assertEqual(roi_path.read_text(), '{invalid json')
                    else:
                        state['pushed_roi'] = read_json(roi_path)
                    self.assertEqual(read_json(cwd / 'outputs/nexus_live.json'), {'original': 'nexus'})
                    self.assertEqual(read_json(cwd / 'outputs/conditional_live.json'), {'original': 'conditional'})
            else:
                name = Path(args[1]).name
                if name == 'nexus_publish_saved_inputs.py':
                    save_json(cwd / 'outputs/nexus_coverage_status.json',
                              {'schedule_complete': True, 'missing_open': []})
                if name == 'conditional_roi.py':
                    state['events'].append('roi_generate')
                    save_json(cwd / 'outputs/conditional_roi.json',
                              {'schema': 1, 'races': [{'id': 'public'}, {'id': 'new private'}]})
            return types.SimpleNamespace(returncode=0)

        with patch.dict(os.environ, {'GITHUB_ACTIONS': 'true'}), \
                patch.dict(sys.modules, {'conditional_roi': self.module}), \
                patch.object(subprocess, 'run', side_effect=process):
            code = self.namespace['publish_ci'](self.root)
        self.assertEqual(code, 0)
        self.assertIn('push', state['events'])
        if invalid_baseline:
            self.assertEqual(state['gates'], 0)
            self.assertNotIn('roi_generate', state['events'])
        else:
            self.assertEqual(state['gates'], 3)
            events = state['events']
            self.assertLess(events.index('roi_generate'), events.index('roi_gate'))
            self.assertLess(events.index('roi_gate'), events.index('git_add'))
            self.assertIn('roi_gate', events[events.index('commit')+1:events.index('push')])
        return state

    def test_generation_commit_and_push_have_60_second_gates(self):
        state = self.exercise_publish()
        self.assertEqual(len(state['pushed_roi']['races']), 2)

    def test_final_gate_failure_amends_back_to_public_baseline_then_publishes_main_feeds(self):
        state = self.exercise_publish(fail_after_commit=True)
        self.assertEqual(state['pushed_roi'], self.baseline)
        self.assertTrue(state.get('roi_restage_after_gate'))
        self.assertIn('amend', state['events'])

    def test_corrupt_optional_baseline_is_preserved_and_does_not_block_main_publication(self):
        self.exercise_publish(invalid_baseline=True)


if __name__ == '__main__':
    unittest.main()
