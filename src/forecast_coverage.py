"""Reconcile forecasts against the full schedule, including missing racecards."""
import csv
import math
from pathlib import Path


def schedule_rows(path, day):
    if not Path(path).exists():
        return []
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return [row for row in csv.DictReader(stream) if row.get('date') == day]


def reconcile_coverage(races, schedule, saved_ids, now):
    by_id = {str(row['race_id']): dict(row) for row in races}
    for row in schedule:
        rid = str(row['race_id'])
        if rid not in by_id:
            by_id[rid] = {key: row.get(key) for key in
                         ('race_id', 'date', 'venue', 'race_no', 'close_at', 'start_at')}
            by_id[rid].update(status='予想なし', reason='開催一覧にあるが出走表未取得')
        else:
            for key in ('close_at', 'start_at'):
                if row.get(key):
                    by_id[rid][key] = row[key]
        if rid in saved_ids:
            by_id[rid].update(status='買い目固定済み', reason='')
    rows = list(by_id.values())
    upcoming_missing = []
    for row in rows:
        try:
            close = float(row.get('close_at'))
        except (TypeError, ValueError):
            close = None
        row['close_at'] = close if close is not None and math.isfinite(close) else None
        if row['status'] != '買い目固定済み' and (row['close_at'] is None or row['close_at'] > now.timestamp()):
            upcoming_missing.append(str(row['race_id']))
    return {'date': now.date().isoformat(), 'input_races': len(rows),
            'forecasted_races': sum(r['status'] == '買い目固定済み' for r in rows),
            'skipped_races': sum(r['status'] != '買い目固定済み' for r in rows),
            'upcoming_missing_race_ids': upcoming_missing,
            'schedule_verified': bool(schedule), 'races': rows}
