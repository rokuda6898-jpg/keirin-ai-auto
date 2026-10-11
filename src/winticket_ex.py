"""WINTICKET EX collection. Observation only: never changes production picks.

Provider EX keys are deliberately not translated by their English names:
exLeftBehind = chigiri (leaving a follower); exSplitLine = chigirare.
No totals are treated as action-choice probabilities, and no current values
are attached to historical forecasts. Forecast joins require capture <= decision.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import tempfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

JST = ZoneInfo('Asia/Tokyo')
METRICS = {
    'tsuppari': 'exThrust', 'kamashi': 'exSpurt',
    'chigiri': 'exLeftBehind', 'chigirare': 'exSplitLine',
    'tobitsuki': 'exSnatch', 'seri': 'exCompete',
}
VALID = {'observed', 'no_history'}
SCHEMA = 'winticket_ex_v1'
URL_PATTERN = r'https://www\.winticket\.jp/keirin/[^/]+/racecard/\d{10}/\d+/\d+'


def finite(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError, OverflowError):
        return None


def integer(value):
    n = finite(value)
    return int(n) if n is not None and n >= 0 and n == int(n) else None


def rid(value):
    value = str(value)
    if not re.fullmatch(r'\d{11,12}', value):
        raise ValueError('invalid race ID')
    return value.zfill(12)


def metric(value):
    out = dict(total=None, succeeded=None, rate=None,
               reported_percentage=None, status='missing')
    if not isinstance(value, dict):
        return out
    n, k, p = integer(value.get('total')), integer(value.get('succeeded')), finite(value.get('percentage'))
    out.update(total=n, succeeded=k, reported_percentage=p, status='invalid')
    if n is None or k is None or k > n:
        return out
    if p is not None and not 0 <= p <= 100:
        return out
    if n == 0:
        if k == 0 and p in (None, 0):
            out['status'] = 'no_history'
        return out
    rate = k / n
    # WINTICKET's integer display can truncate, e.g. 2/3 => 66%.
    if p is not None and abs(100 * rate - p) > 1.000001:
        return out
    out.update(rate=rate, status='observed')
    return out


def source_metrics(record):
    return {name: metric(record.get(key)) for name, key in METRICS.items()}


def posterior_rate(value, prior_mean, prior_weight):
    """Optional shrinkage. Priors must be fitted on strictly earlier data."""
    if not 0 <= prior_mean <= 1 or not math.isfinite(prior_mean) or not math.isfinite(prior_weight) or prior_weight <= 0:
        raise ValueError('invalid prior')
    if value['status'] not in VALID:
        return None
    return (value['succeeded'] + prior_mean * prior_weight) / (value['total'] + prior_weight)


def snapshot(data, source_url, captured_at, expected_race_id=None, scheduled_close=None):
    """Capture all active riders. Incomplete fields/IDs/deadlines fail closed."""
    if not re.fullmatch(URL_PATTERN, source_url):
        raise ValueError('invalid WINTICKET source URL')
    if captured_at.tzinfo is None:
        raise ValueError('timezone-aware capture required')
    race = data.get('race', {})
    race_id = rid(race.get('id', ''))
    if expected_race_id is not None and race_id != rid(expected_race_id):
        raise ValueError('cross-race source')
    close, start = finite(race.get('closeAt')), finite(race.get('startAt'))
    if close is None or start is None or min(close, start) <= 0:
        raise ValueError('missing deadline')
    if scheduled_close is not None:
        sc = finite(scheduled_close)
        if sc is None or abs(sc - close) > 300:
            raise ValueError('schedule deadline mismatch')
        close = min(close, sc)
    declared = integer(race.get('entriesNumber'))
    if declared is None or not 3 <= declared <= 9:
        raise ValueError('invalid declared field')
    entries = [e for e in data.get('entries', [])
               if not (e.get('absent') and integer(e.get('number')) == 0)]
    cars = [integer(e.get('number')) for e in entries]
    if len(cars) != declared or set(cars) != set(range(1, declared + 1)):
        raise ValueError('incomplete or duplicated field')
    active = [e for e in entries if not e.get('absent')]
    ids = [str(e.get('playerId', '')) for e in active]
    if any(not p for p in ids) or len(ids) != len(set(ids)) or len(active) < 3:
        raise ValueError('invalid active rider IDs')
    records = {}
    for r in data.get('records', []):
        p = str(r.get('playerId', ''))
        if p in records:
            raise ValueError('duplicate rider record')
        records[p] = r
    if not set(ids).issubset(records):
        raise ValueError('missing active rider record')
    rows = []
    for e in sorted(active, key=lambda e: int(e['number'])):
        p = str(e['playerId'])
        r = records[p]
        rows.append({'car_no': int(e['number']), 'player_id': p,
                     'metrics': source_metrics(r),
                     'source_ex': {key: r.get(key) for key in METRICS.values()}})
    available = sum(m['status'] in VALID for row in rows for m in row['metrics'].values())
    observed = sum(m['status'] == 'observed' for row in rows for m in row['metrics'].values())
    # Forecast structure is kept as supplied; competing/ambiguous slots are not
    # converted into an invented single-file formation.
    lines = data.get('linePrediction')
    preclose = (captured_at.timestamp() < min(close, start) - 300
                and not race.get('cancel') and not data.get('finalOdds'))
    return {'schema': SCHEMA, 'race_id': race_id, 'source_url': source_url,
            'captured_at': captured_at.astimezone(timezone.utc).isoformat(),
            'close_at': close, 'start_at': start, 'preclose_eligible': preclose,
            'declared_entries': declared, 'active_entries': len(rows),
            'cancelled_cars': [int(e['number']) for e in entries if e.get('absent')],
            'available_metrics': available, 'observed_metrics': observed,
            'expected_metrics': 6 * len(rows),
            'coverage': 'complete' if available == 6 * len(rows) else 'partial',
            'period_start': None, 'period_end': None,
            'period_status': 'not_explicit_in_record_payload',
            'source_provider': 'winticket', 'commercial_use_status': 'review_required',
            'line_prediction': lines, 'riders': rows}


def line_pairs(snap):
    active = {r['car_no'] for r in snap['riders']}
    seen, pairs = [], []
    try:
        for line in snap['line_prediction']['lines']:
            cars = []
            for group in line['entries']:
                if len(group['numbers']) != 1:
                    return []
                c = integer(group['numbers'][0])
                cars.append(c)
                seen.append(c)
            # Chigiri is a line-leader statistic. Do not apply it to the
            # second-wheel rider as though it measured second-to-third splits.
            if len(cars) >= 2:
                pairs.append((cars[0], cars[1]))
    except (KeyError, TypeError):
        return []
    return pairs if len(seen) == len(set(seen)) and set(seen) == active else []


def feature_rows(snap, decision_at):
    if decision_at.tzinfo is None:
        raise ValueError('timezone-aware decision required')
    captured = datetime.fromisoformat(snap['captured_at'])
    if not snap['preclose_eligible'] or captured > decision_at:
        raise ValueError('EX not known before forecast')
    if decision_at.timestamp() >= min(snap['close_at'], snap['start_at']):
        raise ValueError('decision after deadline')
    out = {}
    for r in snap['riders']:
        row = {'race_id': snap['race_id'], 'player_id': r['player_id'], 'car_no': r['car_no']}
        for name, m in r['metrics'].items():
            for key in ('total', 'succeeded', 'rate', 'status'):
                row[f'ex_{name}_{key}'] = m[key]
        row.update(ex_leader_chigiri_rate=None, ex_follower_chigirare_rate=None,
                   ex_pair_split_interaction=None, ex_pair_min_total=None)
        out[r['car_no']] = row
    by_car = {r['car_no']: r for r in snap['riders']}
    for leader, follower in line_pairs(snap):
        a = by_car[leader]['metrics']['chigiri']
        b = by_car[follower]['metrics']['chigirare']
        for car in (leader, follower):
            out[car]['ex_leader_chigiri_rate'] = a['rate']
            out[car]['ex_follower_chigirare_rate'] = b['rate']
            if a['status'] == b['status'] == 'observed':
                # Interaction feature, NOT an independently calibrated chance.
                out[car]['ex_pair_split_interaction'] = a['rate'] * b['rate']
                out[car]['ex_pair_min_total'] = min(a['total'], b['total'])
    return list(out.values())


def save_snapshot(snap, directory):
    if not snap['preclose_eligible']:
        raise ValueError('refuse retrospective EX as pre-race evidence')
    raw = json.dumps(snap, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    digest = hashlib.sha256(raw).hexdigest()
    day = datetime.fromtimestamp(snap['close_at'], JST).strftime('%Y-%m-%d')
    folder = Path(directory) / day
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"{snap['race_id']}_{digest[:20]}.json"
    fd, temporary = tempfile.mkstemp(dir=folder, prefix='.tmp-')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != raw:
                raise ValueError('snapshot hash collision')
    finally:
        os.unlink(temporary)
    return target


def fetch_public(url):
    if not re.fullmatch(URL_PATTERN, url):
        raise ValueError('invalid source URL')
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (compatible; keirin-ai-auto/1.0)', 'Accept-Language': 'ja'})
    with urllib.request.urlopen(req, timeout=30) as response:
        html = response.read().decode('utf-8')
    captured = datetime.now(timezone.utc)  # after completion; never job-start time
    m = re.search(r'window\.__PRELOADED_STATE__\s*=\s*(\{.*?\});\s*window\.__CONFIG__', html, re.S)
    if not m:
        raise ValueError('preloaded state missing')
    state = json.loads(m.group(1))
    candidates = [q.get('state', {}).get('data') for q in state.get('tanStackQuery', {}).get('queries', [])
                  if 'FETCH_KEIRIN_RACE' in (q.get('queryKey') or [])]
    candidates = [d for d in candidates if isinstance(d, dict) and d.get('race')]
    if len(candidates) != 1:
        raise ValueError('ambiguous or missing race payload')
    return candidates[0], captured


def collect(root, fetcher=fetch_public, clock=lambda: datetime.now(timezone.utc), max_fetches=100):
    root = Path(root)
    folder = root / 'outputs/ex_data/snapshots'
    schedule_path = root / 'outputs/latest_race_schedule.csv'
    with schedule_path.open(encoding='utf-8') as f:
        schedule = list(csv.DictReader(f))
    now = clock()
    today = now.astimezone(JST).date().isoformat()
    scheduled = [r for r in schedule if r.get('date') == today]
    if not scheduled:
        raise ValueError('current-day schedule missing; not zero races')
    ids = [rid(r['race_id']) for r in scheduled]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate scheduled race IDs')
    report = {'schema': SCHEMA, 'date': today, 'started_at': now.isoformat(), 'scheduled': len(scheduled),
              'fetched': 0, 'cached': 0, 'closed': 0, 'failures': [], 'races': [],
              'forecast_model_changed': False}
    attempts = 0
    for item in sorted(scheduled, key=lambda r: finite(r.get('close_at')) or 0):
        race_id = rid(item['race_id'])
        close = finite(item.get('close_at'))
        if close is None:
            report['failures'].append({'race_id': race_id, 'reason': 'missing deadline'})
            continue
        seconds = close - clock().timestamp()
        if seconds <= 300:
            report['closed'] += 1
            continue
        # Freeze once early and once within 40 minutes of close. Repeated
        # scheduler runs do not repeatedly refetch unchanged early cards.
        near = seconds <= 2400
        cached = False
        for p in (folder / today).glob(f'{race_id}_*.json'):
            try:
                s = json.loads(p.read_text())
                age_to_close = close - datetime.fromisoformat(s['captured_at']).timestamp()
                if (s['preclose_eligible'] and s['coverage'] == 'complete'
                        and abs(s['close_at'] - close) <= 300
                        and (not near or 300 < age_to_close <= 2400)):
                    cached = True
                    break
            except (OSError, ValueError, KeyError):
                pass
        if cached:
            report['cached'] += 1
            continue
        if attempts >= max_fetches:
            report['failures'].append({'race_id': race_id, 'reason': 'fetch_limit'})
            continue
        attempts += 1
        try:
            data, captured = fetcher(item['source_url'])
            s = snapshot(data, item['source_url'], captured, race_id, close)
            path = save_snapshot(s, folder)
            report['fetched'] += 1
            report['races'].append({'race_id': race_id, 'active_entries': s['active_entries'],
                                    'coverage': s['coverage'], 'available_metrics': s['available_metrics'],
                                    'expected_metrics': s['expected_metrics'], 'snapshot': str(path.relative_to(root))})
            if s['coverage'] != 'complete':
                report['failures'].append({'race_id': race_id, 'reason': 'partial_ex'})
        except Exception as error:
            report['failures'].append({'race_id': race_id, 'reason': f'{type(error).__name__}: {error}'[:240]})
        time.sleep(0.3)
    report['finished_at'] = clock().isoformat()
    path = root / 'outputs/ex_data/collection.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    os.replace(tmp, path)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = collect(args.root)
    print(json.dumps(report, ensure_ascii=False))
    raise SystemExit(1 if report['failures'] else 0)


if __name__ == '__main__':
    main()
