"""Storage integrity tests using only local bytes, temporary Git and stdlib."""
import gzip
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import ledger_io as io
import preserve_validation
import publication_merge
from department_experiment_v2 import append, read_lines, seal


class LedgerStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / 'outputs/company' / io.LARGE_LEDGER
        self.path.parent.mkdir(parents=True)

    def tearDown(self):
        self.temp.cleanup()

    def values(self, path=None):
        return list(read_lines(path or self.path))

    def git(self, *args):
        return subprocess.check_output(
            ['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', *args],
            cwd=self.root, stderr=subprocess.STDOUT).decode().strip()

    def commit(self, message):
        self.git('add', '-A')
        self.git('commit', '-m', message)
        return self.git('rev-parse', 'HEAD')

    def test_initial_compression_preserves_exact_bytes_duplicates_blanks_and_eof(self):
        original = '{"選手":1}\r\n\r\n{"選手":1}\r\n{"選手":2}'.encode()
        self.path.write_bytes(original)
        proof = io.compact_ledger(self.path, force=True)
        self.assertEqual(gzip.decompress(io.archive_path(self.path).read_bytes()), original)
        self.assertEqual(self.path.read_bytes(), b'')
        self.assertEqual(proof['operation'], 'lossless_legacy_compression')
        self.assertEqual(proof['before'], proof['after'])
        self.assertEqual(proof['before'], {'sha256': hashlib.sha256(original).hexdigest(),
                                         'bytes': len(original), 'lines': 4})
        self.assertEqual(self.values(), [{'選手': 1}, {'選手': 1}, {'選手': 2}])

    def test_gzip_is_deterministic_across_directories(self):
        self.path.write_bytes(b'{"r":1}\n' * 40)
        other = self.root / 'other' / io.LARGE_LEDGER
        other.parent.mkdir()
        other.write_bytes(self.path.read_bytes())
        io.compact_ledger(self.path, force=True)
        io.compact_ledger(other, force=True)
        self.assertEqual(io.archive_path(self.path).read_bytes(), io.archive_path(other).read_bytes())

    def test_streaming_compression_across_blocks_keeps_full_content(self):
        original = (b'{"record":"' + b'x' * 1021 + b'"}\n') * 2300 + b'{"last":true}'
        self.assertGreater(len(original), io.BLOCK_BYTES * 2)
        self.path.write_bytes(original)
        proof = io.compact_ledger(self.path, force=True)
        self.assertEqual(gzip.decompress(io.archive_path(self.path).read_bytes()), original)
        self.assertEqual(proof['before']['sha256'], hashlib.sha256(original).hexdigest())
        self.assertEqual(proof['before']['lines'], 2301)

    def test_append_preserves_frozen_records_and_uses_atomic_small_tail(self):
        before = seal({'race_id': 'old', 'snapshot_at': '2026-10-20T12:00:00+09:00'})
        self.path.write_text(json.dumps(before) + '\n')
        io.compact_ledger(self.path, force=True)
        archived = io.archive_path(self.path).read_bytes()
        append(self.path, {'race_id': 'new', 'snapshot_at': '2026-10-20T15:00:00+09:00'})
        self.assertEqual(self.values()[0], before)
        self.assertEqual([r['race_id'] for r in self.values()], ['old', 'new'])
        self.assertEqual(io.archive_path(self.path).read_bytes(), archived)
        self.assertEqual(len(self.path.read_text().splitlines()), 1)
        self.assertEqual(self.values()[1], seal(self.values()[1]))

    def test_interrupted_archive_swap_keeps_all_records_without_double_counting(self):
        original = b'{"r":1}\n{"r":2}\n'
        self.path.write_bytes(original)
        with patch.object(io, '_empty_tail', side_effect=OSError('simulated interrupted swap')):
            with self.assertRaises(OSError):
                io.compact_ledger(self.path, force=True)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(gzip.decompress(io.archive_path(self.path).read_bytes()), original)
        self.assertEqual(self.values(), [{'r': 1}, {'r': 2}])
        proof = io.compact_ledger(self.path, force=True)
        self.assertEqual(proof['operation'], 'reconcile_archive_and_tail')
        self.assertEqual(self.values(), [{'r': 1}, {'r': 2}])

    def test_corrupt_archive_stops_append_compaction_and_backup(self):
        io.archive_path(self.path).write_bytes(b'not-gzip')
        self.path.write_bytes(b'{"new":1}\n')
        before = self.path.read_bytes()
        for operation in (lambda: io.compact_ledger(self.path, force=True),
                          lambda: append(self.path, {'new': 2}),
                          lambda: preserve_validation.save(self.root / 'backup', self.root)):
            with self.assertRaises((OSError, EOFError)):
                operation()
            self.assertEqual(self.path.read_bytes(), before)
        with self.assertRaises((OSError, EOFError)):
            self.values()

    def test_failed_verification_or_replace_keeps_original_intact(self):
        original = b'{"record":1}\n'
        self.path.write_bytes(original)
        with patch.object(io, 'MAX_BLOB_BYTES', 1):
            with self.assertRaises(ValueError):
                io.compact_ledger(self.path, force=True)
        self.assertFalse(io.archive_path(self.path).exists())
        self.assertEqual(self.path.read_bytes(), original)
        with patch.object(io.os, 'replace', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):
                io.compact_ledger(self.path, force=True)
        self.assertEqual(self.path.read_bytes(), original)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_failed_tail_append_is_atomic_and_keeps_archive(self):
        self.path.write_bytes(b'{"r":1}\n')
        io.compact_ledger(self.path, force=True)
        archived = io.archive_path(self.path).read_bytes()
        with patch.object(io.os, 'replace', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):
                append(self.path, {'r': 2})
        self.assertEqual(self.path.read_bytes(), b'')
        self.assertEqual(io.archive_path(self.path).read_bytes(), archived)
        self.assertEqual(self.values(), [{'r': 1}])

    def test_append_reuses_only_unchanged_archive_validation(self):
        self.path.write_bytes(b'{"r":1}\n')
        io.compact_ledger(self.path, force=True)
        archive = io.archive_path(self.path)
        with patch.object(io, 'content_info', wraps=io.content_info) as verify:
            append(self.path, {'r': 2})
            append(self.path, {'r': 3})
            self.assertEqual(verify.call_count, 1)
            replacement = self.root / 'replacement.gz'
            replacement.write_bytes(archive.read_bytes())
            os.replace(replacement, archive)
            append(self.path, {'r': 4})
            self.assertEqual(verify.call_count, 2)
            stat = archive.stat()
            broken = bytearray(archive.read_bytes())
            broken[-8] ^= 1  # Same length, invalid CRC; restore original mtime.
            archive.write_bytes(broken)
            os.utime(archive, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            before_tail = self.path.read_bytes()
            with self.assertRaises(OSError):
                append(self.path, {'r': 5})
            self.assertEqual(verify.call_count, 3)
            self.assertEqual(self.path.read_bytes(), before_tail)

    def test_changed_source_cannot_be_replaced_by_older_snapshot(self):
        self.path.write_bytes(b'{"r":1}\n')

        def racing_blocks():
            yield b'{"r":1}\n'
            self.path.write_bytes(b'{"r":1}\n{"r":2}\n')

        with self.assertRaises(RuntimeError):
            io._atomic_output(self.path, racing_blocks(), compressed=True, sources=[self.path])
        self.assertFalse(io.archive_path(self.path).exists())
        self.assertEqual(self.values(), [{'r': 1}, {'r': 2}])

    def test_save_and_restore_union_legacy_and_archive_without_missing_history(self):
        self.path.write_bytes(b'{"r":1}\n')
        io.compact_ledger(self.path, force=True)
        self.path.write_bytes(b'{"r":2}\n')
        backup = self.root / 'backup'
        preserve_validation.save(backup, self.root)
        io.archive_path(self.path).unlink()
        self.path.write_bytes(b'{"r":1}\n{"r":3}\n')
        preserve_validation.restore(backup, self.root)
        self.assertEqual(self.values(), [{'r': 1}, {'r': 3}, {'r': 2}])
        self.assertEqual(self.path.read_bytes(), b'')
        first = io.archive_path(self.path).read_bytes()
        preserve_validation.restore(backup, self.root)
        self.assertEqual(io.archive_path(self.path).read_bytes(), first)

    def test_normal_save_migrates_before_existing_wildcard_staging(self):
        original = b'{"r":1}\n' * 20
        self.path.write_bytes(original)
        with patch.object(io, 'COMPRESS_AT_BYTES', 20):
            preserve_validation.save(self.root / 'backup', self.root)
        self.assertEqual(gzip.decompress(io.archive_path(self.path).read_bytes()), original)
        self.assertTrue(self.path.exists())
        self.assertEqual(self.path.read_bytes(), b'')
        self.assertEqual(set(self.path.parent.glob('annual_*')),
                         {self.path, io.archive_path(self.path)})

    def test_save_keeps_backup_when_reset_temporarily_removes_both_paths(self):
        self.path.write_bytes(b'{"r":1}\n')
        io.compact_ledger(self.path, force=True)
        backup = self.root / 'backup'
        preserve_validation.save(backup, self.root)
        self.path.unlink()
        io.archive_path(self.path).unlink()
        preserve_validation.save(backup, self.root)
        preserve_validation.restore(backup, self.root)
        self.assertEqual(self.values(), [{'r': 1}])

    def make_branches(self, remote_compressed, local_compressed):
        self.git('init', '-b', 'main')
        self.path.write_bytes(b'{"r":1}\n')
        base = self.commit('original evidence')
        self.git('checkout', '-b', 'remote')
        if remote_compressed:
            io.compact_ledger(self.path, force=True)
        append(self.path, {'r': 2})
        remote = self.commit('remote observation')
        self.git('checkout', '-b', 'local', base)
        if local_compressed:
            io.compact_ledger(self.path, force=True)
        append(self.path, {'r': 3})
        self.commit('local observation')
        self.git('update-ref', 'refs/remotes/origin/main', remote)
        return remote

    def test_publication_merge_keeps_old_and_new_branch_observations(self):
        self.make_branches(remote_compressed=True, local_compressed=False)
        self.assertTrue(publication_merge.reconcile(self.root, validate=lambda root: []))
        self.assertEqual({r['r'] for r in self.values()}, {1, 2, 3})
        self.assertEqual(self.path.read_bytes(), b'')
        committed, compressed = io.git_ledger_lines(self.root, 'HEAD', str(self.path.relative_to(self.root)))
        self.assertTrue(compressed)
        self.assertEqual({json.loads(x)['r'] for x in committed}, {1, 2, 3})
        self.assertEqual(self.git('status', '--porcelain'), '')

    def test_publication_merge_keeps_two_compressed_branches(self):
        self.make_branches(remote_compressed=True, local_compressed=True)
        self.assertTrue(publication_merge.reconcile(self.root, validate=lambda root: []))
        self.assertEqual({r['r'] for r in self.values()}, {1, 2, 3})

    def test_rejected_release_restores_premerge_plain_evidence(self):
        self.make_branches(remote_compressed=True, local_compressed=False)
        original_head = self.git('rev-parse', 'HEAD')
        original_bytes = self.path.read_bytes()
        self.assertFalse(publication_merge.reconcile(self.root, validate=lambda root: ['invalid release']))
        self.assertEqual(self.git('rev-parse', 'HEAD'), original_head)
        self.assertEqual(self.path.read_bytes(), original_bytes)
        self.assertFalse(io.archive_path(self.path).exists())
        self.assertEqual(self.git('status', '--porcelain'), '')

    def test_corrupt_remote_archive_aborts_merge_and_keeps_local_evidence(self):
        self.make_branches(remote_compressed=True, local_compressed=False)
        original_head = self.git('rev-parse', 'HEAD')
        original_bytes = self.path.read_bytes()
        self.git('checkout', 'remote')
        archive = io.archive_path(self.path)
        archive.write_bytes(archive.read_bytes()[:-4])
        corrupt = self.commit('truncated archive')
        self.git('update-ref', 'refs/remotes/origin/main', corrupt)
        self.git('checkout', 'local')
        self.assertFalse(publication_merge.reconcile(self.root, validate=lambda root: []))
        self.assertEqual(self.git('rev-parse', 'HEAD'), original_head)
        self.assertEqual(self.path.read_bytes(), original_bytes)
        self.assertEqual(self.git('status', '--porcelain'), '')

    def test_position_reconciliation_reads_compressed_git_evidence(self):
        remote = self.make_branches(remote_compressed=True, local_compressed=False)
        fake = {
            'department_position_experiment': types.SimpleNamespace(LEDGER='old.jsonl', build_report=lambda output: {}),
            'department_experiment_v2': types.SimpleNamespace(
                LEDGER=io.LARGE_LEDGER, ATTEMPTS='attempts.jsonl', OUTCOMES='outcomes.jsonl',
                build_report=lambda output: {'invalid_records': 0, 'invalid_attempts': 0, 'outcome_conflicts': []}),
            'equation_lab': types.SimpleNamespace(LEDGER='equation.jsonl', MODELS='models.jsonl',
                ATTEMPTS='equation-attempts.jsonl', PATHS='paths.jsonl', build_report=lambda output: {}),
            'fusion_lab': types.SimpleNamespace(FILES=(), build_report=lambda output: {}),
        }
        location = Path(__file__).resolve().parents[1] / 'src/reconcile_position_experiment.py'
        spec = importlib.util.spec_from_file_location('ledger_reconcile_under_test', location)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, fake):
            spec.loader.exec_module(module)
        module.reconcile(self.root, remote)
        self.assertEqual({r['r'] for r in self.values()}, {1, 2, 3})
        self.assertEqual(self.path.read_bytes(), b'')


if __name__ == '__main__':
    unittest.main()
