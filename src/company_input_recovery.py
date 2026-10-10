"""Recover complete company racecards, not conditional-model predictions.

Only recent, complete same-day input with a matching pre-close capture is used.
Canonical company models, department logic and immutable ledgers stay unchanged.
"""
import csv
import json
import math
import os
import re
from datetime import datetime
from pathlib import Path


def read_rows(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def recover(root, now, max_age_seconds=1800):
    root = Path(root)
    if now.tzinfo is None:
        raise ValueError('timezone-aware clock required')
    today = now.date().isoformat()
    schedule = [r for r in read_rows(root/'outputs/latest_race_schedule.csv') if r.get('date') == today]
    expected = {r['race_id']: r for r in schedule}
    if not expected or len(expected) != len(schedule):
        return False
    source = root/'data/raw/conditional_today_entries.csv'
    feed_path = root/'outputs/conditional_live.json'
    if not source.exists() or not feed_path.exists():
        return False
    try:
        feed = json.loads(feed_path.read_text(encoding='utf-8'))
        evidence = {r['id']: r for r in feed['races'] if r['date'] == today}
        raw = read_rows(source)
        groups = {}
        for row in raw:
            if row.get('date') != today or row.get('race_id') not in expected:
                return False
            groups.setdefault(row['race_id'], []).append(row)
        if set(groups) != set(expected):
            return False
        for rid, rows in groups.items():
            schedule_close = float(expected[rid]['close_at'])
            if not math.isfinite(schedule_close):
                return False
            witness = evidence.get(rid, {})
            prediction = witness.get('shadow') or {}
            stamp = datetime.fromisoformat(prediction.get('snapshot_at', ''))
            if stamp.tzinfo is None or not 0 <= now.timestamp()-stamp.timestamp() <= max_age_seconds:
                return False
            if stamp.timestamp() >= schedule_close:
                return False
            cars = [int(r['car_no']) for r in rows]
            players = [r['player_id'] for r in rows]
            count = len(rows)
            if not 3 <= count <= 9 or len(set(cars)) != count or len(set(players)) != count:
                return False
            if any(not p or not 1 <= c <= 9 for p, c in zip(players, cars)):
                return False
            witness_cars = {int(r['car']) for r in witness.get('riders', [])}
            if set(cars) != witness_cars:
                return False
            for row in rows:
                declared = int(float(row.get('declared_entries_number') or row['entries_number']))
                absent = {int(x) for x in re.findall(r'[1-9]', row.get('cancelled_car_numbers', ''))}
                if int(float(row['entries_number'])) != count or set(cars) != set(range(1, declared+1))-absent:
                    return False
                close = float(row['close_at'])
                if not math.isfinite(close) or abs(close-schedule_close) > 300:
                    return False
                if not row.get('source_url', '').startswith('https://www.winticket.jp/keirin/'):
                    return False
                # Refuse result-bearing data as a fresh pre-race input.
                if row.get('finish_pos', '').strip() not in ('', 'nan', 'NaN'):
                    return False
                if str(row.get('result_available', '0')).lower() not in ('0', '0.0', '', 'false'):
                    return False
        destination = root/'data/raw/today_entries.csv'
        temporary = destination.with_suffix('.csv.recovery.tmp')
        with temporary.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(raw[0]))
            writer.writeheader()
            writer.writerows(raw)
        os.replace(temporary, destination)
        count = sum(float(r['close_at']) > now.timestamp() for r in schedule)
        (root/'data/raw/upcoming_races_count.txt').write_text(str(count), encoding='ascii')
        print(json.dumps({'company_input_recovery': 'verified_racecards', 'date': today,
                          'races': len(groups), 'riders': len(raw), 'predictions_copied': False}), flush=True)
        return True
    except (ValueError, TypeError, KeyError, OSError):
        return False
