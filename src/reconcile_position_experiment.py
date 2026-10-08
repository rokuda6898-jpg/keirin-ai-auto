"""Union recorded evidence after concurrent merges; never generate predictions.

Reporting deliberately uses only the standard library so Pages can regenerate
derived comparisons from the exact evidence it uploads.
"""
import argparse
import subprocess
from pathlib import Path

from department_position_experiment import LEDGER, build_report
from department_experiment_v2 import LEDGER as V2_LEDGER, ATTEMPTS, OUTCOMES, build_report as build_v2_report
from equation_lab import (LEDGER as EQUATION_LEDGER, MODELS as EQUATION_MODELS,
                          ATTEMPTS as EQUATION_ATTEMPTS, PATHS as EQUATION_PATHS,
                          build_report as build_equation_report)

ROOT = Path(__file__).resolve().parents[1]
LEDGERS = (LEDGER, 'annual_position_experiment_outcomes.jsonl', V2_LEDGER, ATTEMPTS, OUTCOMES,
           EQUATION_LEDGER, EQUATION_MODELS, EQUATION_ATTEMPTS, EQUATION_PATHS)


def reconcile(root=ROOT, ref=None):
    root = Path(root)
    if ref is not None:
        # Resolve once, fail closed on an invalid ref or unreadable Git object.
        commit = subprocess.check_output(
            ['git', 'rev-parse', '--verify', ref + '^{commit}'], cwd=root,
            encoding='utf-8').strip()
        for name in LEDGERS:
            relative = 'outputs/company/' + name
            exists = subprocess.check_output(
                ['git', 'ls-tree', '--name-only', commit, '--', relative],
                cwd=root, encoding='utf-8').strip()
            if not exists:
                continue
            incoming = subprocess.check_output(
                ['git', 'show', commit + ':' + relative], cwd=root,
                encoding='utf-8').splitlines()
            target = root / relative
            existing = target.read_text(encoding='utf-8').splitlines() if target.exists() else []
            seen = set(existing)
            for line in incoming:
                if line and line not in seen:
                    existing.append(line)
                    seen.add(line)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text('\n'.join(existing) + '\n', encoding='utf-8')
    v2 = build_v2_report(root / 'outputs')
    if v2['invalid_records'] or v2['invalid_attempts'] or v2['outcome_conflicts']:
        raise ValueError('position v2 evidence integrity requires review before publication')
    build_equation_report(root / 'outputs')
    # A concurrent manager may retain an older generated operations page.
    # Refresh its research navigation without generating any forecasts.
    navigation = root / 'outputs/company/operations.html'
    if navigation.exists():
        page = navigation.read_text(encoding='utf-8')
        if 'href="annual_equation_report.html"' not in page:
            if '</nav>' not in page:
                raise ValueError('company navigation missing; equation link cannot be published')
            page = page.replace('</nav>', '<p><a href="annual_equation_report.html">'
                                '３つの方程式の比較研究・社長への採用は検証後</a></p></nav>', 1)
            navigation.write_text(page, encoding='utf-8')
    return build_report(root / 'outputs')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ref', help='Git revision whose recorded evidence must also survive')
    args = parser.parse_args()
    report = reconcile(ref=args.ref)
    print(f"Position evidence reconciled: frozen={report['frozen_races']} "
          f"settled={report['settled_races']} invalid={report['invalid_ledger_rows']}")
