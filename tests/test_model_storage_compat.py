"""The storage certificate cannot hide source, model or training-date changes."""
import ast
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from datetime import date, timedelta
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import model_storage_compat as compat


class ModelStorageCompatibilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root/'models/individual_three_year'
        self.folder.mkdir(parents=True)
        (self.root/'src').mkdir()
        for name in (*compat.LEGACY_SOURCES, 'model_storage_compat.py'):
            shutil.copyfile(ROOT/'src'/name, self.root/'src'/name)
        self.manifest_path = self.folder/'manifest.json'
        shutil.copyfile(ROOT/'models/individual_three_year/manifest.json', self.manifest_path)
        shutil.copyfile(ROOT/compat.PROOF_PATH, self.root/compat.PROOF_PATH)
        self.original_manifest_bytes = self.manifest_path.read_bytes()
        self.manifest = json.loads(self.original_manifest_bytes)
        self.proof_path = self.root/compat.PROOF_PATH
        self.proof = json.loads(self.proof_path.read_bytes())
        self.current = dict(self.manifest['source_hashes'])
        for name in compat.LEGACY_SOURCES:
            self.current[name] = compat.sha256(compat.source_bytes(self.root/'src'/name))

    def accepted(self):
        return compat.source_hashes_match(self.manifest, self.current,
            root=self.root, manifest_path=self.manifest_path)

    def save_proof(self):
        self.proof_path.write_text(json.dumps(self.proof), encoding='utf-8')

    def change_source(self, name, old, new, *, update_certificate=False):
        path = self.root/'src'/name
        source = path.read_text(encoding='utf-8')
        self.assertIn(old, source)
        path.write_text(source.replace(old, new, 1), encoding='utf-8')
        self.current[name] = compat.sha256(compat.source_bytes(path))
        if update_certificate:
            self.proof['sources'][name]['after_sha256'] = self.current[name]
            self.save_proof()

    def loader(self, *, digest=None):
        tree = ast.parse((ROOT/'src/individual_three_year.py').read_bytes())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load')
        self.load_model = Mock(return_value='unchanged-trained-model')
        namespace = {'json': json, 'FOLDER': self.folder, 'ROOT': self.root,
            'VERSION': 'three_calendar_years_421_v1', 'storage_compat': compat,
            'hashes': lambda: self.current,
            'study': types.SimpleNamespace(digest_file=lambda _: digest or self.manifest['model_sha256']),
            'joblib': types.SimpleNamespace(load=self.load_model)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-three-year-loader>', 'exec'), namespace)
        return namespace['load']

    def test_exact_reviewed_transition_preserves_original_manifest(self):
        self.assertTrue(self.accepted())
        self.assertEqual(self.manifest_path.read_bytes(), self.original_manifest_bytes)
        self.assertEqual(set(name for name in self.current
            if self.current[name] != self.manifest['source_hashes'][name]), set(compat.LEGACY_SOURCES))

    def test_ordinary_exact_sources_do_not_need_a_certificate(self):
        self.manifest['source_hashes'] = dict(self.current)
        self.proof_path.unlink()
        self.assertTrue(self.accepted())

    def test_unknown_change_in_any_other_dependency_is_rejected(self):
        for name in set(self.current)-set(compat.LEGACY_SOURCES):
            with self.subTest(name=name):
                old = self.current[name]
                self.current[name] = '0'*64
                self.assertFalse(self.accepted())
                self.current[name] = old

    def test_missing_dependency_is_rejected(self):
        self.current.pop('fixed_year_study.py')
        self.assertFalse(self.accepted())

    def test_unknown_source_hash_even_for_a_comment_is_rejected(self):
        self.change_source('department_experiment_v2.py', 'import hashlib', 'import hashlib  # unreviewed')
        self.assertFalse(self.accepted())

    def test_training_math_change_is_rejected_even_if_new_full_hash_is_replaced(self):
        self.change_source('individual_three_year.py', '4.,', '5.,', update_certificate=True)
        self.assertFalse(self.accepted())

    def test_other_module_code_is_rejected_even_if_new_full_hash_is_replaced(self):
        self.change_source('department_experiment_v2.py', 'import math', 'import decimal as math', update_certificate=True)
        self.assertFalse(self.accepted())

    def test_manifest_cutoff_and_bytes_cannot_be_rewritten(self):
        self.manifest['training_cutoff_exclusive'] = '2026-10-08'
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')
        self.proof['training_cutoff_exclusive'] = '2026-10-08'
        self.save_proof()
        self.assertFalse(self.accepted())

    def test_verifier_replacement_is_rejected(self):
        path = self.root/'src/model_storage_compat.py'
        path.write_bytes(path.read_bytes()+b'\n# unreviewed replacement\n')
        self.assertFalse(self.accepted())

    def test_missing_corrupt_or_unrecognized_certificate_is_rejected(self):
        for content in ('{', '{}', '{"schema":"allow-everything"}'):
            self.proof_path.write_text(content, encoding='utf-8')
            self.assertFalse(self.accepted())
        self.proof_path.unlink()
        self.assertFalse(self.accepted())

    def test_before_hash_cannot_be_changed(self):
        self.proof['sources']['department_experiment_v2.py']['before_sha256'] = '0'*64
        self.save_proof()
        self.assertFalse(self.accepted())

    def test_ast_certificate_must_match_at_runtime(self):
        self.proof['sources']['individual_three_year.py']['unchanged_ast_sha256'] = '0'*64
        self.save_proof()
        self.assertFalse(self.accepted())

    def test_actual_loader_accepts_only_original_model_digest(self):
        loader = self.loader()
        manifest, model = loader('2026-10-20')
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(model, 'unchanged-trained-model')
        self.load_model.assert_called_once_with(self.folder/'model.joblib')
        loader = self.loader(digest='0'*64)
        with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
            loader('2026-10-20')
        self.load_model.assert_not_called()

    def test_actual_loader_retains_exclusive_training_date_guard(self):
        loader = self.loader()
        with self.assertRaisesRegex(ValueError, 'trained after forecast date'):
            loader('2026-10-08')
        self.load_model.assert_not_called()
        loader('2026-10-09')
        self.load_model.assert_called_once()

    def test_actual_loader_rejects_version_change_before_loading(self):
        self.manifest['version'] = 'unreviewed-model'
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')
        loader = self.loader()
        with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
            loader('2026-10-20')
        self.load_model.assert_not_called()


class FusionShadowStorageCompatibilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root/'src').mkdir()
        (self.root/'models').mkdir()
        for name in (*compat.SHADOW_LEGACY_SOURCES, 'model_storage_compat.py'):
            shutil.copyfile(ROOT/'src'/name, self.root/'src'/name)
        self.manifest_path = self.root/'models/fusion_shadow_refit_first_anchor.json'
        shutil.copyfile(ROOT/'models/fusion_shadow_refit_first_anchor.json', self.manifest_path)
        shutil.copyfile(ROOT/compat.SHADOW_PROOF_PATH, self.root/compat.SHADOW_PROOF_PATH)
        self.original_manifest_bytes = self.manifest_path.read_bytes()
        self.manifest = json.loads(self.original_manifest_bytes)
        self.proof_path = self.root/compat.SHADOW_PROOF_PATH
        self.proof = json.loads(self.proof_path.read_bytes())
        self.current = dict(self.manifest['source_hashes'])
        self.current['department_experiment_v2.py'] = compat.sha256(
            compat.source_bytes(self.root/'src/department_experiment_v2.py'))
        self.source_path = self.root/'src/fusion_shadow_live.py'
        self.pipeline = compat.shadow_pipeline_sha256(self.source_path.read_bytes())

    def accepted(self):
        return compat.fusion_shadow_match(self.manifest, self.current, self.pipeline,
            root=self.root, manifest_path=self.manifest_path)

    def loader(self, *, digest=None):
        tree = ast.parse((ROOT/'src/fusion_shadow_live.py').read_bytes())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_model')
        self.load_model = Mock(return_value='unchanged-shadow-model')
        self.model_path = self.root/'models/fusion_shadow_refit_first_anchor.joblib'
        namespace = {'json': json, 'MANIFEST': self.manifest_path, 'MODEL': self.model_path,
            'ROOT': self.root, 'storage_compat': compat, 'source_hashes': lambda: self.current,
            'model_pipeline_hash': lambda: self.pipeline,
            'digest': lambda _: digest or self.manifest['model_sha256'],
            'pd': types.SimpleNamespace(Timestamp=lambda value: types.SimpleNamespace(date=lambda: date.fromisoformat(value))),
            'joblib': types.SimpleNamespace(load=self.load_model)}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual-shadow-loader>', 'exec'), namespace)
        return namespace['load_model']

    def test_exact_shadow_dependency_and_pipeline_transition(self):
        self.assertTrue(self.accepted())
        self.assertEqual(self.manifest_path.read_bytes(), self.original_manifest_bytes)
        self.assertEqual(self.proof['pipeline']['before_sha256'], self.manifest['model_pipeline_hash'])
        self.assertNotEqual(self.pipeline, self.manifest['model_pipeline_hash'])

    def test_unknown_pipeline_hash_is_rejected(self):
        self.pipeline = '0'*64
        self.assertFalse(self.accepted())

    def test_unreviewed_loader_full_hash_is_rejected(self):
        self.source_path.write_bytes(self.source_path.read_bytes()+b'\n# unreviewed loader revision\n')
        self.assertFalse(self.accepted())

    def test_ranking_change_is_rejected_even_with_updated_full_and_pipeline_hashes(self):
        source = self.source_path.read_text(encoding='utf-8')
        tree = ast.parse(source)
        rank = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'rank_race_candidates')
        rank.body.append(ast.Assign(targets=[ast.Name(id='unreviewed_math', ctx=ast.Store())], value=ast.Constant(value=1)))
        self.source_path.write_text(ast.unparse(ast.fix_missing_locations(tree))+'\n', encoding='utf-8')
        self.pipeline = compat.shadow_pipeline_sha256(self.source_path.read_bytes())
        self.proof['sources']['fusion_shadow_live.py']['after_sha256'] = compat.sha256(self.source_path.read_bytes())
        self.proof['pipeline']['after_sha256'] = self.pipeline
        self.proof_path.write_text(json.dumps(self.proof), encoding='utf-8')
        self.assertFalse(self.accepted())

    def test_unknown_change_in_other_shadow_dependencies_is_rejected(self):
        for name in set(self.current)-{'department_experiment_v2.py'}:
            with self.subTest(name=name):
                old = self.current[name]
                self.current[name] = '0'*64
                self.assertFalse(self.accepted())
                self.current[name] = old

    def test_profiles_are_not_interchangeable(self):
        shutil.copyfile(ROOT/compat.PROOF_PATH, self.proof_path)
        self.assertFalse(self.accepted())

    def test_original_shadow_manifest_pipeline_cannot_be_rewritten(self):
        self.manifest['model_pipeline_hash'] = self.pipeline
        self.manifest_path.write_text(json.dumps(self.manifest), encoding='utf-8')
        self.assertFalse(self.accepted())

    def test_actual_shadow_loader_retains_model_digest_guard(self):
        loader = self.loader()
        manifest, model = loader('2026-10-20')
        self.assertEqual(manifest, self.manifest)
        self.assertEqual(model, 'unchanged-shadow-model')
        self.load_model.assert_called_once_with(self.model_path)
        loader = self.loader(digest='0'*64)
        with self.assertRaisesRegex(ValueError, 'shadow model or source'):
            loader('2026-10-20')
        self.load_model.assert_not_called()

    def test_actual_shadow_loader_retains_training_date_guard(self):
        loader = self.loader()
        cutoff = date.fromisoformat(self.manifest['training_cutoff_exclusive'])
        with self.assertRaisesRegex(ValueError, 'forecast date'):
            loader((cutoff-timedelta(days=1)).isoformat())
        self.load_model.assert_not_called()
        loader(cutoff.isoformat())
        self.load_model.assert_called_once()

    def test_actual_shadow_loader_rejects_missing_certificate_before_loading(self):
        self.proof_path.unlink()
        loader = self.loader()
        with self.assertRaisesRegex(ValueError, 'dependencies do not match'):
            loader('2026-10-20')
        self.load_model.assert_not_called()


if __name__ == '__main__':
    unittest.main()
