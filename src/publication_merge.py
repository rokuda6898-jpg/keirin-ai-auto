"""Fast, guarded merge of concurrent generated publications.

Never overwrite remote code/models or settled reports. Append-only evidence is
unioned verbatim; conflicting mutable output keeps the remote version. A full
release validation must pass before committing. Otherwise the caller rebuilds.
"""
import subprocess
from pathlib import Path
from preserve_validation import LEDGERS
from ledger_io import (COMPRESS_AT_BYTES, git_ledger_lines, is_archived_ledger, stage_ledger,
                       union_lines as union_ledger_lines, write_ledger_lines)


def union_lines(remote, local):
    lines = list(dict.fromkeys(remote.splitlines() + local.splitlines()))
    return b'\n'.join(line for line in lines if line) + b'\n'


def git(root, *args, check=True):
    return subprocess.run(['git', *args], cwd=root, check=check, capture_output=True).stdout


def reconcile(root, validate=None):
    root = Path(root)
    if git(root, 'status', '--porcelain', '--untracked-files=no').strip():
        return False
    base = git(root, 'merge-base', 'HEAD', 'origin/main').decode().strip()
    changed = git(root, 'diff', '--name-only', base, 'origin/main').decode().splitlines()
    # A new training, scoring, or workflow revision needs a fresh calculation.
    if any(not p.startswith(('outputs/', 'data/raw/')) for p in changed):
        return False
    result = subprocess.run(['git', 'merge', '--no-commit', '--no-ff', 'origin/main'],
                            cwd=root, capture_output=True)
    if not (root / '.git/MERGE_HEAD').exists():
        return result.returncode == 0
    try:
        conflicts = git(root, 'diff', '--name-only', '--diff-filter=U').decode().splitlines()
        handled = set()
        # An old branch may append plain JSONL while another has migrated to
        # gzip. Reconcile the logical ledger even if Git merged the two paths
        # without a textual conflict; neither representation may overwrite it.
        for name in LEDGERS:
            if not is_archived_ledger(name):
                continue
            local, local_compressed = git_ledger_lines(root, 'HEAD', name)
            remote, remote_compressed = git_ledger_lines(root, 'origin/main', name)
            if not local and not remote:
                continue
            write_ledger_lines(root / name, union_ledger_lines(remote, local),
                               compressed=local_compressed or remote_compressed
                               or sum(len(line) + 1 for line in local + remote)
                               >= COMPRESS_AT_BYTES)
            stage_ledger(root, name)
            handled.update((name, name + '.gz'))
        for name in conflicts:
            if name in handled:
                continue
            if name in LEDGERS:
                local = git(root, 'show', 'HEAD:'+name)
                remote = git(root, 'show', 'origin/main:'+name)
                (root/name).write_bytes(union_lines(remote, local))
                git(root, 'add', '--', name)
            elif name.startswith('outputs/') or name in (
                    'data/raw/today_entries.csv', 'data/raw/today_odds.csv'):
                # Preserve newer externally published mutable reports/results.
                git(root, 'checkout', 'origin/main', '--', name)
            else:
                raise ValueError('Unreviewed conflict: '+name)
        if validate is None:
            from release_guard import write_release_manifest, validate_release
            write_release_manifest(root)
            git(root, 'add', 'outputs/release_manifest.json')
            errors = validate_release(root)
        else:
            errors = validate(root)
        if errors:
            raise ValueError('Merged release validation failed: '+str(errors))
        if (root/'outputs/release_manifest.json').exists():
            git(root, 'add', 'outputs/release_manifest.json')
        git(root, 'commit', '-m', 'Reconcile concurrent publication without recalculating frozen picks')
        return True
    except (OSError, EOFError, RuntimeError, ValueError, subprocess.CalledProcessError):
        git(root, 'merge', '--abort')
        return False


if __name__ == '__main__':
    raise SystemExit(0 if reconcile(Path(__file__).resolve().parents[1]) else 1)

