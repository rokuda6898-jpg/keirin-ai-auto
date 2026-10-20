"""Lossless storage for the large position ledger; no network or model imports.

The historical JSONL name stays the logical name. A deterministic gzip stores
archived bytes, while the original path remains a small append tail. Ordinary
legacy compression preserves every byte, including duplicate and blank lines.
Union is used only when reconciling overlapping representations or branches.
"""
import argparse
import gzip
import hashlib
import json
import os
import subprocess
import tempfile
import zlib
from pathlib import Path

LARGE_LEDGER = "annual_position_v2_ledger.jsonl"
COMPRESS_AT_BYTES = 80 * 1024 * 1024
MAX_BLOB_BYTES = 95 * 1024 * 1024
BLOCK_BYTES = 1024 * 1024
_ARCHIVE_CHECKS = {}


def is_archived_ledger(path):
    return Path(path).name == LARGE_LEDGER


def archive_path(path):
    return Path(str(path) + ".gz")


def storage_paths(path):
    path = Path(path)
    return (archive_path(path), path) if is_archived_ledger(path) else (path,)


def existing_paths(path):
    return [p for p in storage_paths(path) if p.exists()]


def _stamp(path):
    if not path.exists():
        return None
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _chunks(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as handle:
        while block := handle.read(BLOCK_BYTES):
            yield block


class _Digest:
    def __init__(self):
        self.sha = hashlib.sha256()
        self.size = self.newlines = 0
        self.last = b""

    def add(self, block):
        self.sha.update(block)
        self.size += len(block)
        self.newlines += block.count(b"\n")
        if block:
            self.last = block[-1:]

    def result(self):
        return {"sha256": self.sha.hexdigest(), "bytes": self.size,
                "lines": self.newlines + int(bool(self.size) and self.last != b"\n")}


def content_info(path):
    """Reading the complete gzip also verifies its CRC and truncation markers."""
    digest = _Digest()
    for block in _chunks(Path(path)):
        digest.add(block)
    return digest.result()


def validated_archive_info(path):
    """Reuse a process-local check only while the exact archive inode is stable.

ctime detects an in-place rewrite even if size and mtime were restored. Every
replacement and changed archive is read through CRC and SHA again. Migration
uses content_info directly and never skips its complete verification.
"""
    path = Path(path)
    key = str(path.resolve())
    stamp = _stamp(path)
    cached = _ARCHIVE_CHECKS.get(key)
    if cached is not None and cached[0] == stamp:
        return dict(cached[1])
    info = content_info(path)
    if _stamp(path) != stamp:
        raise RuntimeError("archive changed during validation")
    _ARCHIVE_CHECKS[key] = (stamp, info)
    return dict(info)


def ledger_lines(path):
    """Yield JSONL bytes, accepting legacy, archive+tail and interrupted swaps.

Duplicates within one physical file survive. Only lines repeated from an
earlier representation are suppressed, so a crash after the archive swap and
before clearing the old tail cannot make the same observations count twice.
"""
    seen_prior = set()
    for physical in existing_paths(path):
        opener = gzip.open if str(physical).endswith(".gz") else open
        current = set()
        with opener(physical, "rb") as handle:
            for raw in handle:
                line = raw.rstrip(b"\r\n")
                if not line:
                    continue
                key = hashlib.sha256(line).digest()
                if key not in seen_prior:
                    yield line
                current.add(key)
        seen_prior.update(current)


def union_lines(*groups):
    """Explicit exact-line union for concurrent evidence, retaining first order."""
    seen = set()
    for group in groups:
        for raw in group:
            line = raw.rstrip(b"\r\n")
            if not line:
                continue
            key = hashlib.sha256(line).digest()
            if key not in seen:
                seen.add(key)
                yield line


def _temp_path(parent, name):
    descriptor, filename = tempfile.mkstemp(prefix="." + name + ".", dir=parent)
    os.close(descriptor)
    return Path(filename)


def _empty_tail(path, expected_stamp):
    if _stamp(path) != expected_stamp:
        raise RuntimeError("ledger changed before clearing archived tail")
    temporary = _temp_path(path.parent, path.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_output(path, blocks, *, compressed=False, sources=()):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    physical = archive_path(path) if compressed else path
    watch = {p: _stamp(p) for p in {*map(Path, sources), *storage_paths(path)}}
    temporary = _temp_path(path.parent, physical.name)
    expected = _Digest()
    try:
        with temporary.open("wb") as raw:
            if compressed:
                handle = gzip.GzipFile(filename="", mode="wb", fileobj=raw,
                                       compresslevel=6, mtime=0)
            else:
                handle = raw
            try:
                for block in blocks:
                    expected.add(block)
                    handle.write(block)
            finally:
                if compressed:
                    handle.close()
            raw.flush()
            os.fsync(raw.fileno())
        if temporary.stat().st_size >= MAX_BLOB_BYTES:
            raise ValueError("ledger representation exceeds safe Git blob size; originals retained")
        actual = _Digest()
        opener = gzip.open if compressed else open
        with opener(temporary, "rb") as check:
            while block := check.read(BLOCK_BYTES):
                actual.add(block)
        if actual.result() != expected.result():
            raise ValueError("ledger byte/hash/line verification failed; originals retained")
        if any(_stamp(p) != stamp for p, stamp in watch.items()):
            raise RuntimeError("ledger changed during atomic write; originals retained")
        os.replace(temporary, physical)
        if compressed:
            # An empty existing name is intentional: shell-expanded annual_*
            # staging in existing workflows must also stage the old blob change.
            _empty_tail(path, watch[path])
        return {"before": expected.result(), "after": actual.result(),
                "content_verified": True, "archive_bytes": physical.stat().st_size}
    finally:
        temporary.unlink(missing_ok=True)


def compact_ledger(path, *, force=False):
    path = Path(path)
    if not is_archived_ledger(path):
        raise ValueError("compression is limited to the position v2 ledger")
    archive = archive_path(path)
    archived = content_info(archive) if archive.exists() else None
    tail_size = path.stat().st_size if path.exists() else 0
    if not tail_size or (not force and tail_size < COMPRESS_AT_BYTES):
        return {"changed": False, "operation": "already_archived" if archived else "below_threshold",
                "logical_path": str(path), "archived_content": archived,
                "tail_bytes": tail_size}
    sources = existing_paths(path)
    if archive.exists():
        # A second representation can overlap after an interrupted swap or an
        # old/new branch merge. This is an explicit union, not byte compression.
        blocks = (line + b"\n" for line in ledger_lines(path))
        operation = "reconcile_archive_and_tail"
    else:
        blocks = _chunks(path)
        operation = "lossless_legacy_compression"
    proof = _atomic_output(path, blocks, compressed=True, sources=sources)
    return {"changed": True, "operation": operation, "logical_path": str(path),
            "archive_path": str(archive), "tail_bytes": path.stat().st_size, **proof}


def append_ledger_line(path, line):
    """Atomically append one complete record to the small active tail."""
    path = Path(path)
    if not is_archived_ledger(path):
        raise ValueError("atomic archive append is limited to the position v2 ledger")
    if len(line) >= COMPRESS_AT_BYTES:
        raise ValueError("single ledger record exceeds the safe append size")
    archive = archive_path(path)
    if archive.exists():
        validated_archive_info(archive)  # Recheck changed or replaced evidence.
    tail_size = path.stat().st_size if path.exists() else 0
    if tail_size and tail_size + len(line) >= COMPRESS_AT_BYTES:
        compact_ledger(path, force=True)
    sources = existing_paths(path)

    def blocks():
        last = b""
        if path.exists():
            for block in _chunks(path):
                last = block[-1:]
                yield block
        if last and last != b"\n":
            yield b"\n"
        yield line

    _atomic_output(path, blocks(), sources=sources)


def write_ledger_lines(path, lines, *, compressed=False, sources=()):
    return _atomic_output(Path(path), (line + b"\n" for line in lines),
                          compressed=compressed, sources=sources)


def merge_ledger_paths(target, *paths):
    target = Path(target)
    sources = [p for path in paths for p in existing_paths(path)]
    compressed = is_archived_ledger(target) and (
        any(str(p).endswith(".gz") for p in sources)
        or sum(p.stat().st_size for p in sources) >= COMPRESS_AT_BYTES)
    return write_ledger_lines(target, union_lines(*(ledger_lines(p) for p in paths)),
                              compressed=compressed, sources=sources)


def git_ledger_lines(root, ref, relative):
    """Read both representations from one immutable Git revision, never network."""
    names = [str(p) for p in storage_paths(Path(relative))]
    present = subprocess.check_output(
        ["git", "ls-tree", "--name-only", ref, "--", *names], cwd=root,
        encoding="utf-8").splitlines()
    groups = []
    compressed = False
    for name in names:
        if name not in present:
            continue
        data = subprocess.check_output(["git", "show", ref + ":" + name], cwd=root)
        if name.endswith(".gz"):
            try:
                data = gzip.decompress(data)
            except (OSError, EOFError, zlib.error) as exc:
                raise ValueError("corrupt compressed Git ledger: " + ref + ":" + name) from exc
            compressed = True
        groups.append(data.splitlines())
    return list(union_lines(*groups)), compressed


def stage_ledger(root, relative):
    names = [str(p) for p in storage_paths(Path(relative))]
    tracked = subprocess.check_output(
        ["git", "ls-files", "--", *names], cwd=root, encoding="utf-8").splitlines()
    selected = [n for n in names if n in tracked or (Path(root) / n).exists()]
    if selected:
        subprocess.run(["git", "add", "-A", "--", *selected], cwd=root, check=True,
                       capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    proof = compact_ledger(args.root / "outputs/company" / LARGE_LEDGER, force=True)
    proof["schema"] = "lossless-ledger-storage-v1"
    text = json.dumps(proof, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
