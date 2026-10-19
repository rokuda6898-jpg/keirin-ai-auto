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
from conditional_input_refresh import valid_race
from nexus_model.forecast_confidence import annotate_conditional_view
from conditional_budget import (freeze as freeze_budget, settle as settle_budget,
                                MIN_BUDGET, BUDGET, UNIT, RANGE_VERSION)
from conditional_budget_odds import capture as capture_budget_odds

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


def saved_outcome_evidence(root, item):
    """Read explicit winners/withdrawals from an already captured same-race page."""
    try:
        rid, day = str(item['race_id']), str(item['date'])
        number = int(float(item['race_no']))
        if (not re.fullmatch(r'\d{12}', rid) or rid[4:] != day.replace('-', '')
                or int(rid[:2]) != number):
            return None
        snap = read_json(root/'data/raw/nexus_page_snapshots'/day/'outcomes'/(rid+'.json'), {})
        if not isinstance(snap, dict):
            return None
        meta, data = snap['meta'], snap['data']
        if not isinstance(meta, dict) or not isinstance(data, dict):
            return None
        race, schedule = data['race'], data['schedule']
        if not isinstance(race, dict) or not isinstance(schedule, dict):
            return None
        if (snap.get('schema') != 1 or snap.get('source_kind') != 'existing_public_page_response'
                or str(meta.get('race_id')) != rid or str(race.get('id')) != rid
                or meta.get('date') != day or str(schedule.get('date')) != rid[4:]
                or meta.get('race_number') != number or race.get('number') != number):
            return None
        captured = datetime.fromisoformat(snap['captured_at'])
        if captured.tzinfo is None:
            return None
        entries = data.get('entries', [])
        declared = race.get('entriesNumber')
        if (type(declared) is not int or not 3 <= declared <= 9
                or not isinstance(entries, list) or len(entries) != declared
                or any(not isinstance(e, dict) or str(e.get('raceId')) != rid
                       or type(e.get('number')) is not int or type(e.get('absent')) is not bool
                       for e in entries)
                or sorted(e['number'] for e in entries) != list(range(1, declared+1))):
            return None
        cancelled = sorted(str(e['number']) for e in entries if e['absent'])
        wins = data.get('trifectaWinningOddsIds') or []
        payouts = {}
        if (not isinstance(wins, list) or any(not isinstance(w, str) for w in wins)
                or len(wins) != len(set(wins))):
            return None
        for quote in data.get('trifecta', []):
            if not isinstance(quote, dict):
                return None
            if quote.get('id') not in wins:
                continue
            key, price = quote.get('key', []), finite(quote.get('payoffUnitPrice'))
            if (str(quote.get('raceId')) != rid or quote.get('absent') is True
                    or not isinstance(key, list) or len(key) != 3 or len(set(key)) != 3
                    or any(type(c) is not int or not 1 <= c <= declared for c in key)
                    or set(map(str, key)) & set(cancelled)
                    or price is None or price <= 0 or not price.is_integer()):
                return None
            buy = '-'.join(map(str, key))
            if buy in payouts:
                return None
            payouts[buy] = int(price)
        outcome = None
        if race.get('decidedAt') and wins and len(payouts) == len(wins):
            outcome = dict(winning_buys=sorted(payouts), payouts=payouts, payout_complete=True,
                           source_url=snap.get('url', ''), result_source='winticket',
                           source_observed_at=captured.isoformat(), decided_at=race['decidedAt'],
                           evidence='existing_public_page_response')
        return dict(cancelled_cars=cancelled, outcome=outcome)
    except (ValueError, TypeError, KeyError, OSError):
        return None


def apply_result_evidence(row, result, evidence=None, csv_cancelled=()):
    """Keep frozen bets intact and never turn refunded combinations into hits."""
    cancelled = set(map(str, row.get('cancelled_cars', []))) | set(map(str, csv_cancelled))
    if evidence:
        cancelled.update(evidence['cancelled_cars'])
    row['cancelled_cars'] = sorted(cancelled)
    previous = row.get('budget_official_outcome') or {}
    trusted_previous = previous if previous.get('evidence') == 'existing_public_page_response' else None
    outcome = (evidence or {}).get('outcome') or trusted_previous or normalize_outcome(result or {})
    if outcome:
        outcome = dict(outcome)
        winners = [t for t in outcome['winning_buys'] if not set(t.split('-')) & cancelled]
        outcome['winning_buys'] = winners
        outcome['payouts'] = {t: outcome['payouts'][t] for t in winners if t in outcome['payouts']}
        outcome['payout_complete'] = bool(winners) and outcome.get('payout_complete', False) and set(outcome['payouts']) == set(winners)
        row['actual'], row['payouts'] = winners, outcome['payouts']
        row['budget_official_outcome'] = outcome
        row['result_observed_at_jst'] = outcome.get('source_observed_at')
    elif result:
        row['actual'], row['payouts'] = [], {}
        row['budget_official_outcome'] = None
    if 'budget' in row:
        row['budget_settlement'] = settle_budget(row['budget'], row['top12'],
                                                row.get('budget_official_outcome'), row['cancelled_cars'])


def run(root=ROOT, predict_new=True, clock=now_jst, model_loader=load_model, predictor=chain.predict,
        odds_loader=capture_budget_odds):
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
    metadata = {str(s['race_id']): dict(s) for s in schedule}
    fields = {}
    # The isolated conditional fetch has precedence only for a complete,
    # current-day race; keep other workflows' inputs untouched.
    for path in (root/'data/raw/today_entries.csv',
                 root/'data/raw/conditional_today_entries.csv'):
        if not path.exists():
            continue
        entries = pd.read_csv(path, dtype={'race_id': str, 'player_id': str}, low_memory=False)
        if entries.empty:
            continue
        for rid, race in entries.groupby('race_id', sort=False):
            rid = str(rid)
            item = metadata.get(rid)
            if item and (path.name != 'conditional_today_entries.csv' or
                         valid_race(race, rid, item['date'], item['close_at'])):
                fields[rid] = race
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
            # Only NEW forecasts receive a quote. Existing forecasts are never
            # retrofitted with later odds, even when the race is still open.
            rows[rid]['budget'] = freeze_budget(rows[rid], odds_loader, clock)
            saved = clock()
            if saved.timestamp() >= closes[rid]:
                del rows[rid]
                errors[rid] = 'オッズ取得中に締切を経過・後付け保存なし'
                continue
            rows[rid]['snapshot_at_jst'] = saved.isoformat()
            rows[rid]['budget_settlement'] = settle_budget(
                rows[rid]['budget'], rows[rid]['top12'], None)
            # Checkpoint before fetching another race; later failures cannot
            # silently replace an earlier allocation with a new market quote.
            save_ledger(root/LEDGER, rows)
    for rid, row in rows.items():
        csv_cancelled = {c for raw in fields[rid].get('cancelled_car_numbers', [])
                         for c in re.findall('[1-9]', str(raw))} if rid in fields else set()
        apply_result_evidence(row, results.get(rid), saved_outcome_evidence(root, row), csv_cancelled)
        metadata.setdefault(rid, row)
    save_ledger(root/LEDGER, rows)
    # The feed always includes every scheduled race, including missing forecasts.
    races = []
    for rid, item in metadata.items():
        row = rows.get(rid)
        outcome = normalize_outcome(results.get(rid, {}))
        view = forecast_view(row)
        if view:
            annotate_conditional_view(view, row)
            view['note'] = '条件付き3連単：1着に応じた2着、その1・2着に応じた3着を評価。'
        cancelled = row.get('cancelled_cars', []) if row else []
        cancelled_set = set(map(str, cancelled))
        reported_buys = row.get('actual', []) if row else outcome['winning_buys'] if outcome else []
        reported_payouts = row.get('payouts', {}) if row else outcome['payouts'] if outcome else {}
        # Some providers list refunded tickets beside the winning order. A
        # withdrawn rider cannot be part of the official finishing order.
        returned_buys = [buy for buy in reported_buys if cancelled_set.intersection(buy.split('-'))]
        actual = [buy for buy in reported_buys if buy not in returned_buys]
        races.append({'id': rid, 'date': item['date'], 'venue': item.get('venue', ''),
                      'number': int(float(item['race_no'])), 'start_at': finite(item.get('start_at')),
                      'close_at': finite(item.get('close_at')), 'shadow': view,
                      'riders': row.get('riders', []) if row else [],
                      'cancelled_cars': cancelled, 'returned_buys': returned_buys,
                      'actual': actual,
                      'payouts': {buy: reported_payouts[buy] for buy in actual if buy in reported_payouts},
                      'forecast_error': errors.get(rid, ''),
                      'result_observed_at': row.get('result_observed_at_jst') if row else None,
                      'budget': row.get('budget') if row else None,
                      'budget_settlement': row.get('budget_settlement') if row else None})
    races.sort(key=lambda r: (r['date'], r['start_at'] or r['close_at'] or 1e12, r['venue'], r['number']))
    completed = clock()
    payload = {'schema': 1, 'updated_at': completed.isoformat(), 'generated_at': completed.isoformat(),
               'schedule_date': max(s['date'] for s in schedule), 'races': races,
               'model': {'id': 'conditional_trifecta_v1',
                   'name': '条件付き3連単・上位12点', 'ticket_count': 12,
                   'selection_evidence': 'model_selection.json'},
               'budget_policy': {'budget_yen': BUDGET, 'minimum_budget_yen': MIN_BUDGET,
                   'maximum_budget_yen': BUDGET, 'unit_yen': UNIT, 'version': RANGE_VERSION,
                   'budget_selection': 'smallest_feasible_no_loss',
                   'mode': 'simulation', 'odds_policy': 'freeze_at_prediction',
                   'settlement_policy': 'official_payout_only', 'backfill': False,
                   'unpriced_retry': 'pre_close_only_30_minutes',
                   'legacy_allocations': 'preserve_saved_v1'},
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

