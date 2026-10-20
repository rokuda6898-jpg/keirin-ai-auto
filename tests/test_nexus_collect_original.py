"""File-boundary and archive-integrity checks; no original research execution."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/collect_original_nexus_annual.py"
SPEC = importlib.util.spec_from_file_location("nexus_original_collector", SCRIPT)
collector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector)


class OriginalCollectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.parent = Path(self.temp.name)
        self.root = self.parent / "元の研究 project"
        self.root.mkdir()
        self.output = self.parent / "original.zip"

    def write(self, relative, data=b'{"fixture": true}\n'):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def complete_inputs(self):
        for relative in collector.REQUIRED:
            self.write(relative, gzip.compress(b'{"fixture": true}\n', mtime=0))

    def link(self, path, target, directory=False):
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.symlink_to(target, target_is_directory=directory)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation is unavailable on this platform")

    def test_complete_archive_preserves_bytes_paths_hashes_and_scope(self):
        self.complete_inputs()
        supplement = self.write("outputs/older/verified_targets.jsonl.gz", gzip.compress(b"other"))
        receipt = self.write(f"{collector.PAIR}/annual_plans_receipt.json")
        source = self.write("work/repaired_history_ticket_study.py", b"raise RuntimeError('must not execute')\n")
        self.write("outputs/unrelated/secrets.json", b"must not collect")
        before = {p.relative_to(self.root).as_posix(): p.read_bytes()
                  for p in self.root.rglob("*") if p.is_file()}
        report = collector.collect(self.root, self.output)
        self.assertEqual(report["status"], "required_file_set_present")
        self.assertFalse(report["annual_races_validated"])
        self.assertFalse(report["annual_roi_calculated"])
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(json.loads(archive.read(collector.MANIFEST)), report)
            self.assertNotIn("outputs/unrelated/secrets.json", archive.namelist())
            for row in report["files"]:
                data = archive.read(row["path"])
                self.assertEqual(data, before[row["path"]])
                self.assertEqual(len(data), row["bytes"])
                self.assertEqual(hashlib.sha256(data).hexdigest(), row["sha256"])
            for path in (supplement, receipt, source):
                self.assertIn(path.relative_to(self.root).as_posix(), archive.namelist())
        for relative, data in before.items():
            self.assertEqual((self.root / relative).read_bytes(), data)

    def test_partial_cli_saves_recoverable_zip_and_reports_exact_missing_primary(self):
        for relative in collector.REQUIRED[:2]:
            self.write(relative, gzip.compress(b"partial"))
        self.write("outputs/other/verified_targets.jsonl.gz", gzip.compress(b"alternate"))
        result = subprocess.run([sys.executable, str(SCRIPT), "--project-root", str(self.root),
                                 "--output", str(self.output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["missing_required_files"], [collector.TARGETS])
        with zipfile.ZipFile(self.output) as archive:
            report = json.loads(archive.read(collector.MANIFEST))
            self.assertEqual(len(report["files"]), 3)
            self.assertEqual(report["status"], "partial_file_set")

    def test_existing_output_is_never_overwritten(self):
        self.output.write_bytes(b"keep original")
        with self.assertRaises(FileExistsError):
            collector.collect(self.root, self.output)
        self.assertEqual(self.output.read_bytes(), b"keep original")

    def test_required_source_symlink_outside_project_is_rejected(self):
        outside = self.parent / "outside.gz"
        outside.write_bytes(b"outside private content")
        self.link(self.root / collector.REQUIRED[0], outside)
        with self.assertRaisesRegex(ValueError, "leaves project root"):
            collector.collect(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_search_does_not_follow_external_directory_symlink(self):
        outside = self.parent / "external"
        outside.mkdir()
        (outside / "verified_targets.jsonl.gz").write_bytes(b"outside")
        self.link(self.root / "outputs/external", outside, directory=True)
        with self.assertRaisesRegex(ValueError, "leaves project root"):
            collector.collect(self.root, self.output)
        self.assertFalse(self.output.exists())

    def test_internal_file_symlink_is_copied_as_bytes(self):
        self.complete_inputs()
        source = self.write("work/current_ticket_constraints.json", b"inside source")
        self.link(self.root / "outputs/older/verified_targets.jsonl.gz", source)
        collector.collect(self.root, self.output)
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(archive.read("outputs/older/verified_targets.jsonl.gz"), b"inside source")

    def test_read_failure_removes_only_this_attempts_partial_zip(self):
        self.complete_inputs()
        with patch.object(collector, "copy_to_zip", side_effect=OSError("read failed")):
            with self.assertRaisesRegex(OSError, "read failed"):
                collector.collect(self.root, self.output)
        self.assertFalse(self.output.exists())
        self.assertTrue(all((self.root / p).exists() for p in collector.REQUIRED))


if __name__ == "__main__":
    unittest.main()
