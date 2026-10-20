#!/usr/bin/env python3
"""Copy original annual research evidence to a ZIP; never execute or rebuild it.

Python 3.9+, standard library only. Exit 0 means the required file set exists,
not that its 27,351 races or results have been validated. Exit 2 means partial.
"""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import zipfile


REPAIRED = "outputs/keirin_repaired_history_ticket_v1"
PAIR = "outputs/keirin_pair_third_rebuild_v1"
TARGETS = "outputs/keirin_three_year_training_v1/verified_targets.jsonl.gz"
REQUIRED = (
    f"{REPAIRED}/forecasts_2025-10-09.jsonl.gz",
    f"{REPAIRED}/annual_plans.jsonl.gz",
    TARGETS,
)
OPTIONAL = (
    f"{PAIR}/annual_plans.jsonl.gz", f"{PAIR}/scores_2025-10-09.jsonl.gz",
    "work/current_ticket_constraints.json",
    "work/repaired_history_ticket_study.py", "work/audit_repaired_history_tickets.py",
    "work/review_repaired_history_tickets.py", "work/pair_third_rebuild.py",
    "outputs/keirin_three_year_full_race_v1/final_reconciliation_receipt.json",
    "outputs/keirin_three_year_full_race_v1/final_report_receipt.json",
)
MANIFEST = "collection_manifest.json"


def within_project(path, root):
    """Resolve file and directory links before accepting any source path."""
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Source leaves project root: {path.relative_to(root)}") from exc
    return path


def discover(root):
    files, skipped_links = {}, []

    def fail_walk(error):
        raise error

    def add(path):
        within_project(path, root)
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path
        elif path.exists() or path.is_symlink():
            raise ValueError(f"Expected a readable file: {path.relative_to(root)}")

    for relative in REQUIRED + OPTIONAL:
        add(root / relative)
    for relative in (REPAIRED, PAIR):
        directory = within_project(root / relative, root)
        for name in ("selection.json", "protocol.json", "calibration.json",
                     "national_results.json", "delivery/receipt.json"):
            add(directory / name)
        if directory.is_dir():
            for path in sorted(directory.glob("*receipt*.json")):
                add(path)
    outputs = within_project(root / "outputs", root)
    if outputs.is_dir():
        for current, directories, names in os.walk(outputs, followlinks=False, onerror=fail_walk):
            for name in directories[:]:
                path = within_project(Path(current) / name, root)
                if path.is_symlink():
                    directories.remove(name)
                    skipped_links.append(path.relative_to(root).as_posix())
            if "verified_targets.jsonl.gz" in names:
                add(Path(current) / "verified_targets.jsonl.gz")
    return files, sorted(skipped_links)


def copy_to_zip(archive, relative, path, root):
    within_project(path, root)
    info = zipfile.ZipInfo(relative)
    info.compress_type = zipfile.ZIP_STORED if relative.endswith(".gz") else zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
        before = os.fstat(source.fileno())
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)
            digest.update(chunk)
            size += len(chunk)
        after = os.fstat(source.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if identity(before) != identity(after) or size != before.st_size:
        raise ValueError(f"Source changed during collection: {relative}")
    return {"path": relative, "bytes": size, "sha256": digest.hexdigest()}


def collect(project_root, output):
    root = Path(project_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("--project-root must be the original project directory")
    output = Path(output).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    files, skipped_links = discover(root)
    missing = [relative for relative in REQUIRED if relative not in files]
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "partial_file_set" if missing else "required_file_set_present",
        "collection_only": True,
        "annual_races_validated": False,
        "annual_roi_calculated": False,
        "scope": "Original NEXUS annual research: 2025-10-09 through 2026-10-08",
        "required_files": list(REQUIRED),
        "missing_required_files": missing,
        "optional_files_not_found": [p for p in OPTIONAL if p not in files],
        "skipped_directory_symlinks": skipped_links,
        "notes": ["File presence and SHA-256 only; no data parsing, replay, or result validation.",
                  "Other verified_targets matches are supplemental; the exact primary remains required.",
                  "Large raw race-information/odds/source archives are not included by default."],
        "files": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    handle = output.open("xb")  # Exclusive creation also protects against concurrent writers.
    try:
        with handle, zipfile.ZipFile(handle, "w", allowZip64=True) as archive:
            for relative, path in sorted(files.items()):
                report["files"].append(copy_to_zip(archive, relative, path, root))
            archive.writestr(MANIFEST, json.dumps(report, ensure_ascii=False, indent=2))
    except BaseException:
        output.unlink(missing_ok=True)  # Remove only this attempt's incomplete output.
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, help="Original Windows or Linux project root")
    parser.add_argument("--output", required=True, help="New ZIP path; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        report = collect(args.project_root, args.output)
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps({"status": report["status"], "output": str(Path(args.output).absolute()),
                      "files_collected": len(report["files"]),
                      "bytes_collected": sum(row["bytes"] for row in report["files"]),
                      "missing_required_files": report["missing_required_files"],
                      "annual_races_validated": False, "annual_roi_calculated": False},
                     ensure_ascii=False, indent=2))
    return 2 if report["missing_required_files"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
