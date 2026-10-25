"""Quarantine unreadable supplemental observations before canonical processing.

Primary history and frozen model sources are never changed. Official collectors
rebuild this optional supplement; original corrupted bytes remain available.
"""
import hashlib
import os
import shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def repair(root=ROOT):
    path = Path(root)/'outputs/company/rider_official_observations.csv'
    if not path.exists():
        return False
    try:
        pd.read_csv(path, dtype={'race_id': str, 'player_id': str}, low_memory=False)
        return False
    except pd.errors.ParserError as exc:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        backup = path.with_name(path.name+'.corrupt-'+digest)
        if not backup.exists():
            shutil.copy2(path, backup)
        if backup.read_bytes() != path.read_bytes():
            raise ValueError('observation quarantine copy does not match')
        path.unlink()
        print(f'Quarantined malformed supplement at {backup}; official collectors will rebuild it: {exc}', flush=True)
        return True


if __name__ == '__main__':
    repair()
