"""Give still-open, previously unallocated forecasts their FIRST priced plan.

Never changes a forecast or any existing budget decision. Later first quotes are
explicitly labeled as allocation-time odds, not the original prediction's odds.
This migration is a separate stage; conditional_live's new-forecast path is kept.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from conditional_budget import freeze, settle

ROOT = Path(__file__).resolve().parents[1]
LEDGER = 'outputs/company/conditional_live_ledger.jsonl'
REPORT = 'outputs/conditional_budget_initialization.json'


def now_jst():
    return datetime.now(ZoneInfo('Asia/Tokyo'))


def eligible(row, scheduled, result, now):
    """A late first allocation is prospective only; known results forbid it."""
    if not scheduled or 'budget' in row or row.get('actual'):
        return False
    if str(result.get('official_result_available', '')).lower() in ('true', '1'):
        return False
    try:
        if any(str(row[k]) != str(scheduled[k]) for k in ('race_id', 'date', 'venue')):
            return False
        if int(float(row['race_no'])) != int(float(scheduled['race_no'])):
            return False
        when = datetime.fromisoformat(row['snapshot_at_jst'])
        close = min(float(row['close_at']), float(scheduled['close_at']))
        if (now.tzinfo is None or when.tzinfo is None or when > now
                or row['date'] != now.astimezone(ZoneInfo('Asia/Tokyo')).date().isoformat()
                or not now.timestamp() + 60 < close or when.timestamp() >= close):
            return False
        cancelled = set(map(str, row.get('cancelled_cars', [])))
        return not any(set(t.split('-')) & cancelled for t in row['top12'])
    except (ValueError, TypeError, KeyError, OverflowError):
        return False


def initialize(rows, schedule, results, quote_loader, clock=now_jst, persist=None, workers=4):
    now = clock()
    if now.tzinfo is None:
        raise ValueError('timezone-aware clock required')
    candidates = [rid for rid, row in rows.items()
                  if eligible(row, schedule.get(rid), results.get(rid, {}), now)]
    candidates.sort(key=lambda rid: (float(rows[rid]['close_at']), rid))
    report = {'generated_at_jst': now.isoformat(), 'attempted': 0, 'allocated': 0,
              'infeasible': 0, 'unavailable': 0, 'expired': 0,
              'odds_basis': 'initial_allocation_after_forecast', 'examples': [], 'failures': []}

    def prepare(rid):
        original = rows[rid]
        target = dict(original)
        target['close_at'] = min(float(original['close_at']), float(schedule[rid]['close_at']))
        if not eligible(original, schedule.get(rid), results.get(rid, {}), clock()):
            return rid, None
        plan = freeze(target, quote_loader, clock)
        # The old forecast time is immutable. Never relabel today's quote as old.
        plan['odds_basis'] = 'initial_allocation_after_forecast'
        plan['prediction_snapshot_at_jst'] = original['snapshot_at_jst']
        if plan['status'] == 'allocated':
            plan['reason'] = '初回配分時オッズで配分成立（元の予想時オッズとは別）'
        return rid, plan

    if not candidates:
        return report
    # Limit read-only HTTP concurrency; only the main thread writes the ledger.
    with ThreadPoolExecutor(max_workers=max(1, min(int(workers), 4))) as executor:
        futures = [executor.submit(prepare, rid) for rid in candidates]
        for future in as_completed(futures):
            rid, plan = future.result()
            report['attempted'] += 1
            row = rows[rid]
            if plan is None or not eligible(row, schedule.get(rid), results.get(rid, {}), clock()):
                report['expired'] += 1
                continue
            row['budget'] = plan
            row['budget_settlement'] = settle(plan, row['top12'], None, row.get('cancelled_cars', []))
            if persist:
                persist(rows)  # Checkpoint FIRST quote before another network request.
            report[plan['status']] += 1
            if plan['status'] == 'allocated' and len(report['examples']) < 3:
                report['examples'].append({
                    'race_id': rid, 'date': row['date'], 'venue': row['venue'],
                    'race_no': row['race_no'], 'close_at': row['close_at'],
                    'prediction_snapshot_at_jst': row['snapshot_at_jst'],
                    'odds_captured_at_jst': plan['odds_captured_at_jst'], 'provider': plan['provider'],
                    'total_stake_yen': plan['total_stake_yen'],
                    'allocations': [{'buy': a['buy'], 'stake_yen': a['stake_yen']} for a in plan['allocations']]})
            elif plan['status'] == 'unavailable' and len(report['failures']) < 5:
                report['failures'].append({'race_id': rid, 'reason': plan['reason']})
    return report


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    temp.replace(path)


def run(root=ROOT, clock=now_jst, quote_loader=None):
    from public_live import frozen, read_csv, read_json
    from conditional_budget_odds import capture
    root = Path(root)
    rows = frozen(root / LEDGER)
    items = read_csv(root / 'outputs/latest_race_schedule.csv')
    schedule = {str(s['race_id']): s for s in items}
    if not items or len(schedule) != len(items):
        raise ValueError('missing or duplicate schedule')
    results = {str(r['race_id']): r for r in read_json(root / 'outputs/latest_results.json', [])}

    def persist(saved):
        path = root / LEDGER
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.jsonl.tmp')
        temp.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False,
            separators=(',', ':')) + '\n' for _, row in sorted(saved.items())), encoding='utf-8')
        temp.replace(path)

    report = initialize(rows, schedule, results, quote_loader or capture, clock, persist)
    atomic_json(root / REPORT, report)
    print('CONDITIONAL_FIRST_ALLOCATION ' + json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    run()
