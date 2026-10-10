"""Prospective, immutable forecasts from the validated conditional joint model.

The company, risk department and older independent ledgers are read-only here.
Historical research is never copied into this prospective ledger.
"""
import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

import trifecta_chain_research as chain
from fixed_year_study import input_reason
from official_outcomes import normalize_outcome
from public_live import finite, forecast_view, frozen, read_csv, read_json

ROOT = Path(__file__).resolve().parents[1]
LEDGER = 'outputs/company/conditional_live_ledger.jsonl'
FEED = 'outputs/conditional_live.json'
SOURCES = ('trifecta_chain_research.py', 'winner_pair_research.py',
           'fixed_year_features.py', 'fixed_year_study.py', 'common.py')


def source_hashes(root):
    # Git checkouts may use CRLF; mathematical code must remain identical.
    return {n: hashlib.sha256((root/'src'/n).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
            for n in SOURCES}


def load_model(root):
    m = read_json(root/'models/conditional_trifecta_v1.json', {})
    model = root/'models/conditional_trifecta_v1.joblib'
    if (m.get('source_hashes') != source_hashes(root)
            or m.get('model_sha256') != chain.digest(model)):
        raise ValueError('conditional model or scoring dependencies changed')
    if not m.get('historical_rate', 0) > .45:
        raise ValueError('historical 45 percent qualification missing')
    return m, joblib.load(model)


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False,
                               separators=(',', ':')), encoding='utf-8')
    temp.replace(path)


def save_ledger(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.jsonl.tmp')
    temp.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False,
                          separators=(',', ':'))+'\n' for _, r in sorted(rows.items())), encoding='utf-8')
    temp.replace(path)


def now_jst():
    return datetime.now(ZoneInfo('Asia/Tokyo'))


def run(root=ROOT, predict_new=True, clock=now_jst, model_loader=load_model, predictor=chain.predict):
    now = clock()
    if now.tzinfo is None:
        raise ValueError('timezone-aware clock required')
    schedule = read_csv(root/'outputs/latest_race_schedule.csv')
    if not schedule:
        raise ValueError('schedule required: cannot silently drop all races')
    if len({s['race_id'] for s in schedule}) != len(schedule):
        raise ValueError('duplicate scheduled race')
    rows = frozen(root/LEDGER)
    results = {str(r['race_id']): r for r in read_json(root/'outputs/latest_results.json', [])}
    entries_path = root/'data/raw/today_entries.csv'
    entries = pd.read_csv(entries_path, dtype={'race_id': str, 'player_id': str}, low_memory=False) if entries_path.exists() else pd.DataFrame()
    fields = {str(rid): race for rid, race in entries.groupby('race_id', sort=False)} if not entries.empty else {}
    metadata = {str(s['race_id']): dict(s) for s in schedule}
    errors, candidates, closes = {}, [], {}
    for rid, item in metadata.items():
        if rid in rows or not predict_new:
            continue
        close = finite(item.get('close_at'))
        if close is None:
            errors[rid] = '締切時刻を確認できません'
            continue
        if close <= now.timestamp() or normalize_outcome(results.get(rid, {})):
            errors[rid] = '締切前の保存なし・後付け予想はしません'
            continue
        race = fields.get(rid)
        if race is None:
            errors[rid] = '出走表の取得待ち'
            continue
        reason = input_reason(race)
        if reason or not race.date.astype(str).eq(item['date']).all():
            errors[rid] = '出走表の整合性エラー: '+str(reason or 'date mismatch')
            continue
        entry_close = pd.to_numeric(race.get('close_at', pd.Series(dtype=float)), errors='coerce')
        if entry_close.isna().any() or entry_close.empty or entry_close.nunique() != 1:
            errors[rid] = '出走表の締切時刻が不一致'
            continue
        close = min(close, float(entry_close.iloc[0]))
        if close <= now.timestamp():
            errors[rid] = '出走表では締切済み'
            continue
        closes[rid] = close
        candidates.append(race)
    if candidates:
        manifest, bundle = model_loader(root)
        # The scoring API masks outcome fields itself; no result is used to rank.
        for prediction in predictor(pd.concat(candidates, ignore_index=True), bundle):
            rid = prediction['race_id']
            if rid not in closes or rid in rows:
                raise ValueError('unexpected or duplicate predicted race')
            completed = clock()
            if completed.timestamp() >= closes[rid]:
                errors[rid] = '計算中に締切を経過'
                continue
            weight = manifest['chain_weight']
            probabilities = ((1-weight)*np.asarray(prediction['independent'])
                             +weight*np.asarray(prediction['chain']))
            if not np.isfinite(probabilities).all() or not np.isclose(sum(probabilities), 1):
                raise ValueError('invalid joint probabilities')
            order = sorted(range(len(probabilities)), key=lambda i: (-probabilities[i], prediction['keys'][i]))[:12]
            race = fields[rid]
            rows[rid] = {**metadata[rid], 'race_id': rid, 'close_at': closes[rid],
                         'snapshot_at_jst': completed.isoformat(),
                         'top12': [prediction['keys'][i] for i in order],
                         'probabilities': [float(probabilities[i]) for i in order],
                         'model_sha256': manifest['model_sha256'],
                         'training_cutoff_exclusive': bundle['cutoff'],
                         'riders': [{'car': str(int(r.car_no)), 'name': str(getattr(r, 'player_name', ''))}
                                    for r in race.itertuples()],
                         'actual': [], 'payouts': {}, 'cancelled_cars': [],
                         'purchase_authorized': False}
            forecast_view(rows[rid])  # Validate tickets before persistence.
    for rid, row in rows.items():
        if rid in results:
            outcome = normalize_outcome(results[rid])
            row['actual'] = outcome['winning_buys'] if outcome else []
            row['payouts'] = outcome['payouts'] if outcome else {}
            row['result_observed_at_jst'] = results[rid].get('result_observed_at_jst')
        if rid in fields:
            row['cancelled_cars'] = sorted({c for raw in fields[rid].get('cancelled_car_numbers', [])
                                            for c in re.findall('[1-9]', str(raw))})
        metadata.setdefault(rid, row)
    save_ledger(root/LEDGER, rows)
    # The feed always includes every scheduled race, including missing forecasts.
    races = []
    for rid, item in metadata.items():
        row = rows.get(rid)
        outcome = normalize_outcome(results.get(rid, {}))
        view = forecast_view(row)
        if view:
            view['note'] = '条件付き3連単：1着に応じた2着、その1・2着に応じた3着を評価。'
        races.append({'id': rid, 'date': item['date'], 'venue': item.get('venue', ''),
                      'number': int(float(item['race_no'])), 'start_at': finite(item.get('start_at')),
                      'close_at': finite(item.get('close_at')), 'shadow': view,
                      'riders': row.get('riders', []) if row else [],
                      'cancelled_cars': row.get('cancelled_cars', []) if row else [],
                      'actual': row.get('actual', []) if row else outcome['winning_buys'] if outcome else [],
                      'payouts': row.get('payouts', {}) if row else outcome['payouts'] if outcome else {},
                      'forecast_error': errors.get(rid, '')})
    races.sort(key=lambda r: (r['date'], r['start_at'] or r['close_at'] or 1e12, r['venue'], r['number']))
    completed = clock()
    payload = {'schema': 1, 'updated_at': completed.isoformat(), 'generated_at': completed.isoformat(),
               'schedule_date': max(s['date'] for s in schedule), 'races': races,
               'historical_reference': {'hits': 6242, 'races': 13612, 'rate': 6242/13612,
                   'scope': '過去ページを用いた再検証。未来の成績ではありません。'},
               'coverage': {'scheduled': len(schedule), 'saved': sum(s['race_id'] in rows for s in schedule),
                            'errors': errors}}
    atomic_json(root/FEED, payload)
    # Missing upcoming forecasts are visible and also fail the monitoring run.
    missing = [s['race_id'] for s in schedule if (finite(s.get('close_at')) or 0) > completed.timestamp()
               and s['race_id'] not in rows]
    return payload, missing if predict_new else []


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--settle-only', action='store_true')
    args = p.parse_args()
    report, missing = run(predict_new=not args.settle_only)
    print(json.dumps(report['coverage'], ensure_ascii=False))
    if missing:
        raise SystemExit('Upcoming forecasts missing: '+', '.join(missing))
