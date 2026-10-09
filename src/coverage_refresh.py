"""Recover full-day inputs when morning publication or company coverage failed."""
import csv
import argparse
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]


def refresh_mode(schedule, entries, saved, now):
    today = now.date().isoformat()
    rows = [r for r in schedule if r.get('date') == today]
    if not rows or not any(r.get('date') == today for r in entries):
        return 'daily'
    for row in rows:
        close = float(row['close_at'])
        if close > now.timestamp() and str(row['race_id']) not in saved:
            return 'all_upcoming'
    return 'near_close'


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    now = datetime.now(ZoneInfo('Asia/Tokyo'))
    path = ROOT/'outputs/company/company_decision_ledger.jsonl'
    saved = set()
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            stamp = datetime.fromisoformat(row['snapshot_at_jst'])
            if stamp.tzinfo and stamp.timestamp() < float(row['close_at']) and row.get('top12'):
                saved.add(str(row['race_id']))
    mode = refresh_mode(read_csv(ROOT/'outputs/latest_race_schedule.csv'),
                        read_csv(ROOT/'data/raw/today_entries.csv'), saved, now)
    print('Input refresh mode:', mode)
    if args.verify:
        if mode != 'near_close':
            raise SystemExit('Upcoming company coverage incomplete: '+mode)
        return
    if mode == 'daily':
        from fetch_today_entries import fetch_today_entries
        frame = fetch_today_entries()
        count = frame.loc[frame.close_at.astype(float) > now.timestamp(), 'race_id'].nunique()
        (ROOT/'data/raw/upcoming_races_count.txt').write_text(str(count), encoding='ascii')
    else:
        from fetch_upcoming_entries import fetch_upcoming
        fetch_upcoming(min_minutes=0 if mode == 'all_upcoming' else 5,
                       max_minutes=1440 if mode == 'all_upcoming' else 40)


if __name__ == '__main__':
    main()
