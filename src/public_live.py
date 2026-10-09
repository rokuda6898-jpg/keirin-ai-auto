"""Small public data feed. Refresh data without redeploying the website."""
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from official_outcomes import normalize_outcome

ROOT = Path(__file__).resolve().parents[1]
LABELS = {'data_department':'データ部','pace_department':'展開部','line_department':'ライン部',
          'risk_department':'リスク部','prediction_department':'予想部',
          'strategist_department':'軍師','high_payout_department':'高配当戦略部'}


def read_json(path, default):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def frozen(path):
    rows = {}
    if not path.exists():
        return rows
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            stamp = row.get('snapshot_at_jst', '')
            close = finite(row.get('close_at'))
            when = datetime.fromisoformat(stamp)
            if close is None or when.tzinfo is None or when.timestamp() >= close:
                raise ValueError('Invalid pre-close forecast: '+str(row.get('race_id')))
            rid = str(row['race_id'])
            if rid not in rows or stamp < rows[rid]['snapshot_at_jst']:
                rows[rid] = row
    return rows


def marks(row):
    scores = defaultdict(lambda: [0., 0., 0.])
    probs = row.get('probabilities', [])
    for rank, buy in enumerate(row.get('top12', [])):
        weight = finite(probs[rank]) if rank < len(probs) else None
        weight = weight if weight is not None and weight > 0 else 1/(rank+1)
        for position, car in enumerate(buy.split('-')):
            scores[car][position] += weight
    order = sorted(scores, key=lambda car: (*(-n for n in scores[car]), int(car)))
    return [{'mark': mark, 'car': car} for mark, car in zip('◎○▲△☆', order)]


def forecast_view(row):
    if not row:
        return None
    tickets = row.get('top12', [])
    if len(tickets) > 12 or len(set(tickets)) != len(tickets):
        raise ValueError('Invalid ticket count')
    for buy in tickets:
        parts = buy.split('-')
        if len(parts) != 3 or len(set(parts)) != 3 or any(p not in '123456789' or len(p) != 1 for p in parts):
            raise ValueError('Invalid trifecta')
    return {'tickets': tickets, 'marks': marks(row), 'snapshot_at': row['snapshot_at_jst'],
            'opinions': [{'department': LABELS.get(name,name), 'tickets': buys,
                          'evidence': row.get('department_evidence', {}).get(name, {})}
                         for name, buys in row.get('department_submissions', {}).items()],
            'rule': row.get('strategist', {}).get('rule', '')}


def build(root=ROOT, now=None):
    now = now or datetime.now(ZoneInfo('Asia/Tokyo'))
    out = root/'outputs'
    company = frozen(out/'company/company_decision_ledger.jsonl')
    shadow = frozen(out/'company/fusion_shadow_live_ledger.jsonl')
    schedule = read_csv(out/'latest_race_schedule.csv')
    if not schedule:
        raise ValueError('Schedule is required; refusing an empty feed')
    race_map = {str(row['race_id']): dict(row) for row in schedule}
    for saved in (company, shadow):
        for rid, row in saved.items():
            race_map.setdefault(rid, {key: row.get(key) for key in
                ('race_id','date','venue','race_no','close_at','start_at')})
    results = {str(r['race_id']): r for r in read_json(out/'latest_results.json', [])}
    riders = defaultdict(list)
    previous = read_json(out/'public_live.json', {})
    cancelled = defaultdict(set, {r['id']:set(r.get('cancelled_cars', []))
                                 for r in previous.get('races', [])})
    entry_rows = read_csv(root/'data/raw/today_entries.csv')
    for rid in {str(e['race_id']) for e in entry_rows}:
        cancelled[rid] = set()
    for entry in entry_rows:
        import re
        cancelled[str(entry['race_id'])].update(re.findall(r'[1-9]', entry.get('cancelled_car_numbers', '')))
    for row in read_csv(out/'latest_predictions.csv'):
        riders[str(row['race_id'])].append({'car':str(row['car_no']), 'name':row.get('player_name','')})
    races = []
    for rid, row in race_map.items():
        result = results.get(rid, {})
        outcome = normalize_outcome(result)
        actual = outcome['winning_buys'] if outcome else []
        payouts = outcome['payouts'] if outcome else {}
        if not outcome and rid not in results and shadow.get(rid, {}).get('actual'):
            old = shadow[rid]
            # Ledger actual may contain several tied orders. Its legacy odds
            # is the maximum *hit* payout, never the payout for every winner.
            recovered = normalize_outcome({'official_result_available': bool(old.get('result_source')),
                                           'actual_trifecta': old['actual']})
            if recovered:
                actual = recovered['winning_buys']
                hits, odds = old.get('hit_tickets', []), old.get('hit_odds', [])
                if len(hits) == len(odds):
                    payouts = {buy: finite(price)*100 for buy, price in zip(hits, odds)
                               if buy in actual and finite(price) is not None and finite(price) > 0}
        races.append({'id':rid, 'date':row['date'], 'venue':row['venue'],
                      'number':int(float(row['race_no'])), 'start_at':finite(row.get('start_at')),
                      'close_at':finite(row.get('close_at')), 'riders':riders.get(rid, []),
                      'cancelled_cars':sorted(cancelled.get(rid, set())),
                      'company':forecast_view(company.get(rid)), 'shadow':forecast_view(shadow.get(rid)),
                      'actual':actual, 'payouts':{k:finite(v) for k,v in payouts.items()}})
    races.sort(key=lambda r:(r['date'], r['start_at'] or r['close_at'] or 1e12, r['venue'], r['number']))
    source_stamps = [r.get('result_observed_at_jst') for r in results.values()]
    source_stamps += [r.get('schedule_fetched_at_jst') for r in schedule]
    for name in ('company_decision_predictions.json','fusion_shadow_live_predictions.json'):
        source_stamps.append(read_json(out/'company'/name,{}).get('updated_at_jst'))
    source_times = []
    for stamp in source_stamps:
        if stamp:
            value = datetime.fromisoformat(stamp)
            if value.tzinfo is not None and value <= now:
                source_times.append(value)
    # Republishing an old snapshot must not falsely make its data look fresh.
    updated = max(source_times).isoformat(timespec='seconds') if source_times else None
    payload = {'schema':1,'updated_at':updated,'generated_at':now.isoformat(timespec='seconds'),
               'schedule_date':max(r['date'] for r in schedule), 'races':races,
               'historical_reference':{'shadow_top12_rate':.3877,'scope':'過去一年の検証値。今後の的中率ではありません。'}}
    target = out/'public_live.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(',',':'), allow_nan=False), encoding='utf-8')
    temporary.replace(target)
    print(f'Public feed: {len(races)} races; {target.stat().st_size} bytes')
    return payload


if __name__ == '__main__':
    build()
