"""G1/G2/G3 advisory department. Frozen pre-close, independently scored.

All coefficients are explicit hypotheses, not fitted/calibrated probabilities.
The existing seven departments and risk baseline are not modified.
"""
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from itertools import permutations
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'graded_scenarios_v1'
GRADES = {3: 'G3', 4: 'G2', 5: 'G1'}  # WINTICKET cup enum, not race class
JST = ZoneInfo('Asia/Tokyo')


def number(value, default=0.):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def normalize(values):
    total = sum(values.values())
    if total <= 0:
        raise ValueError('Empty probability mass')
    return {key: value / total for key, value in values.items()}


def grade_of(data):
    cup_id = data.get('schedule', {}).get('cupId')
    cup = next((c for c in data.get('cups', []) if c.get('id') == cup_id), {})
    # Exclude lower-grade supporting races within a graded meeting.
    return GRADES.get(cup.get('grade')) if data.get('race', {}).get('isGradeRace') is True else None


def rank_tickets(entries, quotes):
    """Conditional model + market joint support; never uses win popularity for 3rd."""
    riders = {int(r['car_no']): r for r in entries}
    if len(riders) != len(entries) or len(riders) < 3:
        raise ValueError('Duplicate or insufficient starters')
    keys = list(permutations(sorted(riders), 3))
    market = None
    if quotes:
        if set(quotes) != set(keys) or any(not 1 <= p < 9999.9 for p in quotes.values()):
            raise ValueError('Complete valid trifecta quotes required')
        market = normalize({key: 1 / quotes[key] for key in keys})
    verified = all(r.get('line_verification_status') == 'verified'
                   and number(r.get('line_position')) >= 1
                   and str(r.get('line_id', '')).lower() not in ('', 'nan', 'none') for r in entries)
    leaders = {c for c, r in riders.items() if verified and number(r.get('line_position')) == 1
               and (r.get('style') == '逃' or number(r.get('front_runner_count')) > 0)}
    competing = len({riders[c]['line_id'] for c in leaders}) >= 2
    max_score = max(number(r.get('score')) for r in entries)
    scores = {}
    for c, r in riders.items():
        # Convert cumulative place rates to mutually exclusive position evidence.
        first = min(1., max(0., number(r.get('win_rate'))))
        second = max(0., min(1., number(r.get('place2_rate'))) - first)
        third = max(0., min(1., number(r.get('place3_rate'))) - first - second)
        strength = math.exp(max(-6., (number(r.get('score')) - max_score) / 12))
        scores[c] = [strength * (.08 + first), math.sqrt(strength) * (.08 + second),
                     math.sqrt(strength) * (.08 + third)]
    reasons = {}
    def factor(a, b, c, position):
        if not verified:
            return 1.
        r = riders[c]
        same = r['line_id'] == riders[a]['line_id']
        value = 1.
        if competing and c in leaders:
            value *= .9  # bounded contest hypothesis; never a fixed winning axis
        if same and number(r['line_position']) > number(riders[a]['line_position']):
            value *= 1.15
        if b is not None:
            # A survivor's follower differs from a generic popular third pick.
            follow_b = r['line_id'] == riders[b]['line_id'] and number(r['line_position']) > number(riders[b]['line_position'])
            if follow_b:
                value *= 1.15
        return value
    win = normalize({c: scores[c][0] * (1.1 if competing and number(riders[c].get('line_position')) == 2 else 1.) for c in riders})
    model = {}
    for a in riders:
        second = normalize({b: scores[b][1] * factor(a, None, b, 2) for b in riders if b != a})
        for b in second:
            third = normalize({c: scores[c][2] * factor(a, b, c, 3) for c in riders if c not in (a, b)})
            for c, value in third.items():
                model[a, b, c] = win[a] * second[b] * value
    joint = normalize({k: model[k] ** .35 * market[k] ** .65 for k in keys}) if market else model
    ordered = sorted(keys, key=lambda k: (-joint[k], k))
    ranks = []
    if market:
        for position in range(3):
            weights = {c: sum(p for k, p in market.items() if k[position] == c) for c in riders}
            ranks.append({c: i + 1 for i, c in enumerate(sorted(riders, key=lambda c: (-weights[c], c)))})
    holes = [k for k in ordered if market and quotes[k] >= 100
             and joint[k] / market[k] >= 1.10 and any(ranks[p][k[p]] >= 4 for p in range(3))][:12]
    def pack(k):
        labels = []
        a, b, c = k
        if competing and number(riders[a].get('line_position')) == 2:
            labels.append('先行争いから番手浮上')
        if verified and riders[a]['line_id'] != riders[b]['line_id']:
            labels.append('別ラインの2着')
        if verified and riders[c]['line_id'] in (riders[a]['line_id'], riders[b]['line_id']):
            labels.append('1・2着のラインから3着残り')
        if market:
            labels += [f'{p+1}着支持{ranks[p][k[p]]}位' for p in range(3) if ranks[p][k[p]] >= 4]
        return {'buy': '-'.join(map(str, k)), 'score': joint[k], 'odds': quotes.get(k), 'reasons': labels}
    main = ordered[:12]
    baseline = sorted(keys, key=lambda k: (-market[k], k))[:12] if market else []
    hole_baseline = sorted((k for k in keys if quotes.get(k, 0) >= 100), key=lambda k: quotes[k])[:len(holes)]
    return {'main': [pack(k) for k in main], 'hole': [pack(k) for k in holes],
            'market_baseline': ['-'.join(map(str, k)) for k in baseline],
            'hole_market_baseline': ['-'.join(map(str, k)) for k in hole_baseline],
            'verified_lines': verified, 'competing_leaders': competing,
            'calibrated': False, 'method': VERSION}


def read_json(path, default):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def atomic_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    temp.replace(path)


def load_ledger(path):
    records = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            stamp = datetime.fromisoformat(row['snapshot_at_jst'])
            if stamp.tzinfo is None or stamp.timestamp() >= row['close_at'] or row['grade'] not in GRADES.values():
                raise ValueError('Invalid graded pre-close record')
            rid = row['race_id']
            if rid in records and records[rid] != row:
                old = records[rid]
                if old.get('market_available') or not row.get('market_available') or stamp <= datetime.fromisoformat(old['snapshot_at_jst']):
                    raise ValueError('Conflicting immutable graded records')
            records[rid] = row
    return records


def forecast(schedule, now, fetch_state, saved, clock=None):
    """Fetch each race's own grade and account for pending forecasts.

    Keep an initial unpriced forecast. A single upgrade when the complete market
    arrives freezes normal/hole/comparators together. No subsequent re-ranking.
    """
    from fetch_today_entries import find_query_data, build_odds_rows, _entry_rows_complete
    from race_features import build_entry_rows
    decisions, additions = [], []
    for row in schedule:
        rid = str(row['race_id'])
        decision = {**row, 'race_id': rid, 'grade': saved.get(rid, {}).get('grade'), 'status': 'pending'}
        decisions.append(decision)
        if rid in saved and saved[rid].get('market_available'):
            decision['status'] = 'saved'
            continue
        if number(row.get('close_at')) - now.timestamp() <= 300:
            decision['status'] = 'saved_waiting_odds' if rid in saved else 'closed_without_snapshot'
            continue
        try:
            state = fetch_state(row['source_url'])
            data = find_query_data(state, 'FETCH_KEIRIN_RACE')
            if str(data.get('race', {}).get('id')) != rid:
                raise ValueError('Race identity mismatch')
            grade = grade_of(data)
            decision['grade'] = grade
            if not grade:
                decision['status'] = 'not_target'
                continue
            if data['race'].get('cancel'):
                decision['status'] = 'cancelled'
                continue
            close = number(data['race'].get('closeAt'))
            captured = clock() if clock else datetime.now(JST)
            if close - captured.timestamp() <= 300:
                decision['status'] = 'near_close'
                continue
            entries = build_entry_rows(data, row['date'], row['venue'], int(row['race_no']), rid, row['source_url'], False)
            if not _entry_rows_complete(entries)[0]:
                raise ValueError('Incomplete starters')
            odds_data = find_query_data(state, 'FETCH_KEIRIN_RACE_ODDS')
            if odds_data.get('oddsDelayed') or odds_data.get('finalOdds'):
                raise ValueError('Delayed or final odds are not pre-race evidence')
            odds = build_odds_rows(odds_data, row['date'], row['venue'], row['race_no'], rid, row['source_url'])
            quotes = {}
            for quote in odds:
                if quote['bet_type'] != 'trifecta':
                    continue
                key = tuple(map(int, quote['buy'].split('-')))
                if key in quotes:
                    raise ValueError('Duplicate quote')
                quotes[key] = number(quote['odds_used'], float('nan'))
            # Partial markets must not create distorted popularity ranks.
            expected = set(permutations([int(r['car_no']) for r in entries], 3))
            quote_ready = set(quotes) == expected and all(1 <= p < 9999.9 for p in quotes.values())
            updated = number(odds_data.get('oddsUpdatedAt'))
            if quotes and (updated <= 0 or updated > captured.timestamp() or captured.timestamp() - updated > 300):
                quote_ready = False
            if not quote_ready:
                quotes = {}
                if rid in saved:
                    decision['status'] = 'saved_waiting_odds'
                    continue
            ranked = rank_tickets(entries, quotes)
            record = {**row, 'race_id': rid, 'grade': grade, 'close_at': close,
                      'start_at': number(data['race'].get('startAt')),
                      'snapshot_at_jst': captured.isoformat(timespec='seconds'),
                      'riders': [{'car': str(r['car_no']), 'name': r['player_name']} for r in entries],
                      'cancelled_cars': [str(c) for c in range(1, 10) if str(c) in str(entries[0].get('cancelled_car_numbers', ''))],
                      'quotes_source': row['source_url'], 'quotes_captured_at': captured.isoformat(timespec='seconds'),
                      'market_available': quote_ready,
                      'input_features': [{k: (None if isinstance(r.get(k), float) and not math.isfinite(r[k]) else r.get(k)) for k in ('car_no','score','win_rate','place2_rate','place3_rate','line_verification_status','line_id','line_position','style','front_runner_count')} for r in entries],
                      'market_quotes': {'-'.join(map(str,k)): v for k,v in quotes.items()},
                      **ranked}
            additions.append(record)
            decision['status'] = 'saved' if quote_ready else 'saved_waiting_odds'
        except Exception as exc:
            decision['status'] = 'data_error'
            decision['reason'] = str(exc)
    return additions, decisions


def build_feed(root, records, decisions, now):
    from official_outcomes import normalize_outcome
    from public_live import marks
    latest = {str(r['race_id']): r for r in read_json(root/'outputs/latest_results.json', [])}
    previous = {r['id']: r for r in read_json(root/'outputs/graded_live.json', {}).get('races', [])}
    current_public = {r['id']: r for r in read_json(root/'outputs/public_live.json', {}).get('races', [])}
    coverage = {d['race_id']: d for d in decisions if d.get('grade') in GRADES.values()}
    for rid, r in records.items():
        coverage.setdefault(rid, r)
    races = []
    for rid, row in coverage.items():
        saved = records.get(rid)
        race = {'id': rid, 'date': row['date'], 'venue': row['venue']+' '+row['grade'],
                'number': int(row['race_no']), 'start_at': number(row.get('start_at')),
                'close_at': number(row.get('close_at')), 'riders': saved['riders'] if saved else [],
                'cancelled_cars': saved.get('cancelled_cars', []) if saved else [],
                'actual': [], 'payouts': {}, 'grade': None, 'grade_hole': None,
                'department_status': row.get('status', 'saved')}
        outcome = normalize_outcome(latest.get(rid, {}))
        race['cancelled_cars'] = sorted(set(race['cancelled_cars']) | set(current_public.get(rid, {}).get('cancelled_cars', [])))
        if outcome:
            race.update(actual=outcome['winning_buys'], payouts=outcome['payouts'])
        elif rid not in latest and rid in previous:
            race.update(actual=previous[rid]['actual'], payouts=previous[rid]['payouts'])
        if saved:
            for field, kind in [('grade','main'),('grade_hole','hole')]:
                picks = saved[kind]
                buys = [p['buy'] for p in picks]
                race[field] = {'tickets': buys, 'marks': marks({'top12': buys}),
                               'market_available': saved['market_available'],
                               'snapshot_at': saved['snapshot_at_jst'], 'opinions': [],
                               'ticket_details': picks,
                               'baseline': saved['market_baseline' if kind == 'main' else 'hole_market_baseline'],
                               'note': ('ライン確認済み' if saved['verified_lines'] else 'ライン未確認：隊列補正なし') +
                                       (' ／ オッズ確認済み' if saved['market_available'] else ' ／ オッズ待ち：通常予想のみ事前保存')}
        races.append(race)
    races.sort(key=lambda r: (r['date'], r['start_at'], r['number']))
    return {'schema': 1, 'generated_at': now.isoformat(), 'updated_at': now.isoformat(),
            'schedule_date': now.date().isoformat(), 'races': races,
            'coverage': decisions, 'graded_department': True,
            'note': 'G1・G2・G3専用。100倍以上でも条件を満たさない穴は追加しません。重賞で穴が増えるという仮説・重みは未検証です。'}


def run(root=ROOT, carry=None):
    from fetch_today_entries import http_get, extract_preloaded_state
    now = datetime.now(JST)
    folder = root/'outputs/company'
    folder.mkdir(parents=True, exist_ok=True)
    ledger = folder/'graded_department_ledger.jsonl'
    saved = load_ledger(ledger)
    if carry:
        lines = set(ledger.read_text(encoding='utf-8').splitlines() if ledger.exists() else [])
        if Path(carry).exists():
            lines.update(Path(carry).read_text(encoding='utf-8').splitlines())
        lines.discard('')
        lines = sorted(lines, key=lambda line: json.loads(line)['snapshot_at_jst'])
        candidate = ledger.with_suffix('.merge')
        candidate.write_text(''.join(line+'\n' for line in lines), encoding='utf-8')
        saved = load_ledger(candidate)
        candidate.replace(ledger)
    with (root/'outputs/latest_race_schedule.csv').open(encoding='utf-8-sig') as f:
        schedule = [r for r in csv.DictReader(f) if r['date'] == now.date().isoformat()]
    if not schedule:
        raise ValueError('Current-day schedule missing; do not claim zero graded races')
    # Explicit per-race grade flag from the same day's full entry ingestion
    # avoids refetching every F1/F2 race. Never infer a grade from the venue.
    excluded = set()
    entries_path = root/'data/raw/today_entries.csv'
    if entries_path.exists():
        flags = defaultdict(list)
        with entries_path.open(encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if r.get('date') == now.date().isoformat():
                    flags[r['race_id']].append(r.get('is_grade_race'))
        excluded = {rid for rid, values in flags.items() if values and all(v == '0' for v in values)}
    def fetch(url):
        return extract_preloaded_state(http_get(url, attempts=2))
    additions, decisions = forecast([r for r in schedule if r['race_id'] not in excluded], now, fetch, saved)
    decisions += [{**r, 'grade': None, 'status': 'not_target'} for r in schedule if r['race_id'] in excluded]
    if additions:
        with ledger.open('a', encoding='utf-8') as f:
            for row in additions:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                saved[row['race_id']] = row
    atomic_json(folder/'graded_department_status.json', {'updated_at': now.isoformat(), 'decisions': decisions})
    atomic_json(root/'outputs/graded_live.json', build_feed(root, saved, decisions, now))
    print(f'Graded department: {len(additions)} new; {len(saved)} frozen; {sum(d["status"] == "data_error" for d in decisions)} data errors')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--carry')
    run(carry=parser.parse_args().carry)
