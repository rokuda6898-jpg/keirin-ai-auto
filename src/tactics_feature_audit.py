"""Research-only audit of recorded forecast lines and historical result-event flags.

No synthetic line formations, result-derived race-day predictors, or production writes.
The event names are provider field identifiers, NOT verified tactical translations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

EVENT_FIELDS = (
    'result_event_back', 'result_event_spurtSucceeded',
    'result_event_thrustSucceeded', 'result_event_leftBehind',
    'result_event_splitLine', 'result_event_snatchSucceeded',
    'result_event_competeSucceeded', 'result_event_hasAccident',
)
LINE_FIELDS = ('line_id', 'line_position', 'line_verification_status')
READ_FIELDS = set(EVENT_FIELDS) | set(LINE_FIELDS) | {
    'race_id', 'date', 'car_no', 'player_id', 'race_class', 'race_type',
    'result_available', 'line_source', 'line_verification_scope',
}
BAD_TEXT = {'', 'none', 'nan', 'null', '<na>'}


def truth(value):
    """Parse explicitly recorded boolean flags, never cast missing -> False."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    key = str(value).strip().lower()
    if key in ('true', '1', '1.0', 'yes'):
        return True
    if key in ('false', '0', '0.0', 'no'):
        return False
    return None


def safe_text(value):
    return None if pd.isna(value) or str(value).strip().lower() in BAD_TEXT else str(value).strip()


def line_geometry(race):
    """One 1..N ordering in each declared line, no duplicate cars or riders."""
    n = len(race)
    cars = pd.to_numeric(race['car_no'], errors='coerce')
    if not 3 <= n <= 9 or cars.isna().any() or cars.nunique() != n or not cars.between(1, 9).all():
        return False
    if 'player_id' in race and race['player_id'].nunique() != n:
        return False
    grouped = defaultdict(list)
    for r in race.itertuples(index=False):
        line = safe_text(getattr(r, 'line_id', None))
        try:
            pos = float(getattr(r, 'line_position', None))
        except (TypeError, ValueError):
            return False
        if line is None or not 1 <= pos <= 9 or pos != int(pos):
            return False
        grouped[line].append(int(pos))
    return all(sorted(positions) == list(range(1, len(positions) + 1))
               for positions in grouped.values())


def _rate(n, total):
    return round(n / total, 6) if total else None


def analyze(frame, cutoff):
    required = {'race_id', 'date', 'car_no', 'player_id'}
    if not required.issubset(frame):
        raise ValueError(f'missing history identity fields: {sorted(required-set(frame))}')
    output = {'version': 'tactics_history_audit_v1', 'training_cutoff_exclusive': cutoff,
              'event_semantics': 'raw_provider_boolean_flags_not_verified_japanese_tactic_labels',
              'line_geometry_not_equivalent_to_provider_verified_or_timestamped_line': True,
              'data_point_in_time_verified': False, 'production_modified': False,
              'partitions': {}}
    for label, group in [('training', frame[frame.date.astype(str).str.slice(0, 10) < cutoff]),
                         ('holdout', frame[frame.date.astype(str).str.slice(0, 10) >= cutoff])]:
        line_report = Counter()
        for rid, race in group.groupby('race_id', sort=False):
            line_report['all_races'] += 1
            if race['car_no'].nunique() != len(race) or race['player_id'].nunique() != len(race):
                line_report['duplicate_rider_or_car_races'] += 1
            has_ids = 'line_id' in race and race.line_id.map(safe_text).notna().all()
            has_positions = 'line_position' in race and pd.to_numeric(race.line_position, errors='coerce').notna().all()
            verified_status = 'line_verification_status' in race and race.line_verification_status.astype(str).eq('verified').all()
            geometry = has_ids and has_positions and line_geometry(race)
            line_report['all_line_ids_present'] += int(has_ids)
            line_report['all_positions_present'] += int(has_positions)
            line_report['status_all_verified'] += int(verified_status)
            line_report['self_consistent_geometry'] += int(geometry)
            line_report['usable_verified_geometry'] += int(geometry and verified_status)
            line_report['geometry_without_verified_status'] += int(geometry and not verified_status)
            if not has_ids:
                line_report['missing_some_line_ids'] += 1
            if not has_positions:
                line_report['missing_some_positions'] += 1
        event_report = {}
        riders = group.player_id.astype(str)
        for name in EVENT_FIELDS:
            if name not in group:
                event_report[name] = {'status': 'field_missing', 'observed': 0}
                continue
            values = group[name].map(truth)
            known = values.notna()
            positive = values.eq(True)
            negative = values.eq(False)
            # A field that is only True when present lacks a denominator for success rates.
            event_report[name] = {
                'status': 'observed_boolean' if known.any() else 'all_missing',
                'observed': int(known.sum()), 'true': int(positive.sum()),
                'false': int(negative.sum()), 'unparsed_or_missing': int((~known).sum()),
                'distinct_players_observed': int(riders[known].nunique()),
                'distinct_players_positive': int(riders[positive].nunique()),
                'coverage_of_rows': _rate(int(known.sum()), len(group)),
                'explicit_denominator_available': bool(negative.any()),
            }
        line_counts = {k: int(v) for k, v in line_report.items()}
        for k in ('all_races', 'usable_verified_geometry', 'self_consistent_geometry',
                  'geometry_without_verified_status', 'missing_some_line_ids',
                  'missing_some_positions', 'status_all_verified'):
            line_counts.setdefault(k, 0)
        output['partitions'][label] = {
            'races': int(group.race_id.nunique()), 'rider_rows': int(len(group)),
            'first_date': str(group.date.min()) if len(group) else None,
            'last_date': str(group.date.max()) if len(group) else None,
            'line': line_counts, 'line_verified_rate': _rate(line_counts['usable_verified_geometry'], line_report['all_races']),
            'events': event_report,
            'line_status_rows': group.line_verification_status.astype(str).value_counts().head(12).to_dict() if 'line_verification_status' in group else {},
            'line_sources': group.line_source.astype(str).value_counts().head(12).to_dict() if 'line_source' in group else {},
            'line_scope': group.line_verification_scope.astype(str).value_counts().head(12).to_dict() if 'line_verification_scope' in group else {},
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', default='data/raw/history.csv')
    parser.add_argument('--out-dir', default='research/tactics_history_audit')
    parser.add_argument('--cutoff', default='2025-10-09')
    args = parser.parse_args()
    path = Path(args.history)
    if not path.is_file():
        raise SystemExit('Actual central archive missing; refuse proxy data')
    columns = set(pd.read_csv(path, nrows=0).columns)
    if not {'race_id','date','car_no','player_id'}.issubset(columns):
        raise SystemExit('Central history identity columns missing')
    history = pd.read_csv(path, usecols=lambda c: c in READ_FIELDS, dtype={'race_id':str,'player_id':str}, low_memory=False)
    summary = analyze(history, args.cutoff)
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            sha.update(block)
    summary['source_sha256'] = sha.hexdigest()
    target = Path(args.out_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target/'tactics_history_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    for part, value in summary['partitions'].items():
        print('TACTICS_AUDIT', json.dumps({'period':part, 'races':value['races'], 'lines':value['line'],
            'events':{k:{a:d.get(a) for a in ('observed','true','false','coverage_of_rows')}
                      for k,d in value['events'].items()}},ensure_ascii=False),flush=True)
    print('TACTICS_AUDIT_STATUS success', flush=True)


if __name__ == '__main__':
    main()
