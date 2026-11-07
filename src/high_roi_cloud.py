"""Publish a distinct provisional policy from intact, pre-close NEXUS evidence.

This does not reproduce the unavailable ROI9891 model or reuse its performance.
Forecasts are frozen once created. Publication receipts and official settlements
are stored outside the forecast, and only confirmed pre-close receipts qualify
for this policy's prospective performance.
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from nexus_input_snapshots import canonical, read_json, save_json, timestamp
from nexus_model import optimizer, replay_evidence
from nexus_model.forecast_confidence import from_joint
from secondary_results_sync import official_outcome, settlement, summary

ROOT = Path(__file__).resolve().parents[1]
JST = ZoneInfo('Asia/Tokyo')
MODEL_ID = 'high_roi_cloud_20261107_v1'
OUTPUT = Path('outputs/high_roi_cloud_live.json')
BANDS = ('MAIN', 'MEDIUM', 'HIGH')
METHOD = {
    'id': 'saved-joint-sqrt-odds-v1',
    'name': '高配当・再構築版（検証中）',
    'description': '保存済みNEXUS同時確率と締切前オッズを使う新方式。旧ROI9891とは別方式。',
    'formula': 'q * sqrt(min(odds, 500))',
    'odds_clip': 500,
    'empirically_validated': False,
    'historical_performance_inherited': False,
    'probability_only_fallback': '有効なオッズで全選択買い目を構成できない場合は同時確率のみで選択',
}


def now_jst():
    return datetime.now(JST)


def digest(value):
    return hashlib.sha256(canonical(value).encode('utf-8')).hexdigest()


def input_digest(snapshot):
    data, meta = snapshot['data'], snapshot['meta']
    model_data = {k: data[k] for k in ('players', 'entries', 'records', 'linePrediction') if k in data}
    model_data['race'] = {k: data['race'].get(k) for k in ('id', 'distance', 'startAt', 'closeAt')}
    model_data['schedule'] = {k: data['schedule'].get(k) for k in ('date', 'cupId', 'index', 'day')}
    model_meta = {k: v for k, v in meta.items() if k not in ('source_url', 'session')}
    return digest([model_meta, model_data])


def _numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def _epoch(value):
    if _numeric(value):
        return float(value)
    return timestamp(value).timestamp()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_source(row, snapshot, manifest, manifest_hash, now, *, max_age=1800, min_lead=300):
    """Check provenance and timing before either quote scoring or optimization."""
    _require(now.tzinfo is not None, 'timezone-aware clock required')
    today = now.astimezone(JST).date().isoformat()
    meta, data = snapshot['meta'], snapshot['data']
    rid = row['race_id']
    _require(isinstance(rid, str) and re.fullmatch(r'\d{12}', rid), 'invalid race identity')
    _require(row['date'] == meta['date'] == today and rid[-8:] == today.replace('-', ''), 'race date is not today')
    _require(meta['race_id'] == rid == str(data['race']['id']), 'snapshot race mismatch')
    _require(snapshot.get('source_kind') == 'existing_public_page_response', 'unsupported snapshot source')
    captured = timestamp(snapshot['captured_at'])
    close = _epoch(meta['close_at'])
    _require(_numeric(meta['close_at']) and _numeric(meta['start_at']), 'numeric source schedule required')
    _require(close == _epoch(row['close_at']) == _epoch(data['race']['closeAt']), 'close time mismatch')
    _require(meta['start_at'] >= close, 'invalid start time')
    _require(close - now.timestamp() > max(300, min_lead), 'inside publication deadline')
    age = (now - captured).total_seconds()
    _require(captured.astimezone(JST).date().isoformat() == today and 0 <= age <= max_age, 'snapshot not fresh')
    _require(not meta.get('cancel') and not row.get('cancelled'), 'cancelled race')
    _require(not row.get('result_complete') and not row.get('winners') and not row.get('payouts'), 'source row already has results')
    _require(not data.get('results') and not data.get('trifectaWinningOddsIds')
             and not data.get('finalOdds') and not data['race'].get('decidedAt'), 'post-result or final-odds snapshot')
    hashed = input_digest(snapshot)
    _require(snapshot.get('input_sha256') == hashed, 'snapshot input hash mismatch')
    _require(re.fullmatch(r'[a-f0-9]{64}', str(snapshot.get('sha256', ''))), 'raw source hash missing')
    pred = row['prediction']
    evidence = pred['replay_evidence']
    replay_evidence.verify(evidence)
    _require(evidence.get('evidence_kind') == 'prospective_preclose_inference', 'unbound inference evidence')
    _require(pred.get('model_id') == evidence.get('model_id') == manifest['id'], 'base model mismatch')
    _require(evidence.get('model_manifest_sha256') == manifest_hash, 'base manifest hash mismatch')
    _require(evidence.get('race_id') == rid and evidence.get('date') == today, 'evidence race mismatch')
    _require(_epoch(evidence['close_at']) == close, 'evidence deadline mismatch')
    _require(pred.get('input_sha256') == evidence.get('input_sha256') == hashed, 'saved inference input mismatch')
    _require(pred.get('source_sha256') == evidence.get('source_sha256'), 'saved inference source mismatch')
    inference_at = timestamp(pred['captured_at'])
    input_at = timestamp(pred['input_captured_at'])
    _require(input_at <= inference_at <= now and inference_at.timestamp() < close, 'inference is not pre-close')
    _require(evidence.get('snapshot_at') == pred['captured_at']
             and evidence.get('input_captured_at') == pred['input_captured_at'], 'inference timestamp mismatch')
    history = pred['history_through']
    _require(history == evidence.get('feature_history_through') and date.fromisoformat(history) < date.fromisoformat(today), 'history cutoff mismatch')
    cutoff = (date.fromisoformat(manifest['train'][1]) + timedelta(days=1)).isoformat()
    _require(evidence.get('training_cutoff_exclusive') == cutoff and cutoff <= today, 'training cutoff mismatch')
    cars, q = evidence['cars'], evidence['q']
    _require(isinstance(cars, list) and 3 <= len(cars) <= 9 and len(set(cars)) == len(cars)
             and all(type(c) is int and 1 <= c <= 9 for c in cars), 'invalid active cars')
    active = {x['number'] for x in data['entries'] if not x.get('absent')}
    _require(set(cars) == active, 'active entry mismatch')
    _require(evidence.get('q_order') == 'itertools.permutations(cars,3)', 'unknown q ordering')
    triples = list(itertools.permutations(cars, 3))
    _require(len(q) == len(triples) and all(_numeric(v) and 0 <= v <= 1 for v in q)
             and abs(math.fsum(q) - 1) <= 1e-8, 'invalid complete joint probabilities')
    bands = [pred['band_probabilities'][b] for b in BANDS]
    _require(all(_numeric(v) and 0 <= v <= 1 for v in bands)
             and abs(math.fsum(bands) - 1) <= 1e-8, 'invalid band probabilities')
    return cars, q, triples, bands


def valid_quotes(snapshot, triples):
    """Ignore absent, non-quoted/capped, duplicate, or settled quote entries."""
    allowed, found, duplicates = set(triples), {}, set()
    rid = snapshot['meta']['race_id']
    for entry in snapshot['data'].get('trifecta', []) or []:
        try:
            key = tuple(entry['key'])
            value = entry['odds']
            valid = (key in allowed and str(entry.get('raceId')) == rid
                     and not entry.get('absent') and entry.get('unitPrice') == 100
                     and not entry.get('payoffUnitPrice')
                     and _numeric(value) and 1 <= value < 9999.9)
            if not valid:
                continue
            if key in found:
                duplicates.add(key)
            found[key] = float(value)
        except (TypeError, KeyError):
            continue
    return {key: value for key, value in found.items() if key not in duplicates}


def build_forecast(row, snapshot, manifest, manifest_hash, *, clock=now_jst, max_age=1800, min_lead=300):
    now = clock()
    cars, q, triples, bands = validate_source(row, snapshot, manifest, manifest_hash, now,
                                             max_age=max_age, min_lead=min_lead)
    category = BANDS[max(range(3), key=lambda i: bands[i])]
    cap = 18 if snapshot['meta'].get('graded') is True else (10 if category == 'HIGH' else 8)
    _require(len(triples) >= cap, 'too few active combinations for required budget')
    quotes = valid_quotes(snapshot, triples)
    mode, plan = 'probability_only', None
    if len(quotes) >= cap:
        weights = [p * math.sqrt(min(quotes.get(t, 0), 500)) for t, p in zip(triples, q)]
        total = math.fsum(weights)
        if total > 0:
            candidate = optimizer.select(cars, [w / total for w in weights], cap)
            if all(tuple(t) in quotes for t in candidate['tickets']):
                plan, mode = candidate, 'odds_smoothed'
    if plan is None:
        plan = optimizer.select(cars, q, cap)
    tickets = ['-'.join(map(str, t)) for t in plan['tickets']]
    _require(len(tickets) == len(set(tickets)) == cap and 1 <= len(plan['blocks']) <= 4, 'invalid optimized budget')
    expanded = []
    for block in plan['blocks']:
        part = [t for t in itertools.product(*(block[k] for k in ('first', 'second', 'third'))) if len(set(t)) == 3]
        _require(block['count'] == len(part), 'invalid formation block count')
        expanded.extend('-'.join(map(str, t)) for t in part)
    _require(len(expanded) == cap and set(expanded) == set(tickets), 'incomplete formation expansion')
    saved = clock()
    # Optimization may cross the deadline; never stamp the earlier input time.
    _require(_epoch(row['close_at']) - saved.timestamp() > max(300, min_lead), 'deadline passed during optimization')
    _require(saved >= now, 'clock moved backwards')
    first = {str(c): math.fsum(p for t, p in zip(triples, q) if t[0] == c) for c in cars}
    ordered = sorted(cars, key=lambda c: (-first[str(c)], c))
    marks = [{'car': c, 'mark': ('◎', '○', '▲', '△', '☆')[i] if i < 5 else '印なし'}
             for i, c in enumerate(ordered)]
    pred = row['prediction']
    forecast = {
        'model_id': MODEL_ID, 'reconstruction': True, 'category': category,
        'band_probabilities': bands, 'count': cap, 'unit_yen': 100, 'total_yen': cap * 100,
        'blocks': plan['blocks'], 'tickets': tickets, 'marks': marks,
        'unmarked': ordered[5:], 'first_probabilities': first,
        'captured_at': snapshot['captured_at'], 'saved_at': saved.isoformat(),
        'history_through': pred['history_through'], 'source_inference_at': pred['captured_at'],
        'source_sha256': snapshot['sha256'], 'input_sha256': snapshot['input_sha256'],
        'source_url': snapshot.get('url', snapshot['meta'].get('source_url')),
        'source_mode': mode, 'method': copy.deepcopy(METHOD),
        'quote_count': len(quotes), 'quote_total': len(triples),
        'selected_odds_count': cap if mode == 'odds_smoothed' else 0,
        'base_model_id': manifest['id'], 'base_manifest_sha256': manifest_hash,
        'base_evidence_sha256': pred['replay_evidence']['evidence_sha256'],
        'replay_evidence': copy.deepcopy(pred['replay_evidence']),
        'training_cutoff_exclusive': pred['replay_evidence']['training_cutoff_exclusive'],
        'confidence': from_joint(cars, q, tickets, annotated_at=saved.isoformat()),
        'empirical_return_validation': False,
    }
    # Quotes appear only when every selected ticket was actually odds-scored.
    if mode == 'odds_smoothed':
        forecast['odds'] = {'-'.join(map(str, t)): quotes[tuple(t)] for t in plan['tickets']}
        forecast['odds_at'] = snapshot['captured_at']
    result = schedule_row(row, snapshot, saved)
    result.update(status='forecast', forecast=forecast,
                  forecast_sha256=digest(forecast), performance_eligible=False)
    return result


def schedule_row(row, snapshot, now, reason=None):
    """Keep the full day's schedule visible without inventing a late forecast."""
    meta = (snapshot or {}).get('meta', {})
    rid = row['race_id']
    result = {k: meta.get(k, row.get(k)) for k in (
        'race_id', 'date', 'venue', 'race_number', 'race_class', 'race_type', 'graded')}
    for key, alias in (('race_number', 'race_no'), ('race_class', 'class_name'), ('race_type', 'stage')):
        if result.get(key) is None:
            result[key] = row.get(alias)
    result['close_at'] = int(_epoch(meta.get('close_at', row['close_at'])))
    result['start_at'] = int(_epoch(meta.get('start_at', row.get('start_at', row['close_at']))))
    result['venue_id'] = str(meta.get('venue_id') or row.get('venue_id') or rid[2:4])
    result['session'] = row.get('session') or meta.get('session')
    result['source_url'] = row.get('source_url') or meta.get('source_url') or (snapshot or {}).get('url')
    result['lineup'] = copy.deepcopy(row.get('lines', []))
    riders = copy.deepcopy(row.get('riders', []))
    data = (snapshot or {}).get('data', {})
    records = {str(record.get('playerId')): record for record in data.get('records', [])}
    players = {str(player.get('id')): player for player in data.get('players', [])}
    for rider in riders:
        pid = str(rider.get('pid', ''))
        record, player = records.get(pid, {}), players.get(pid, {})
        for key, value in [('score', record.get('racePoint')), ('style', record.get('style')),
                           ('prefecture', player.get('prefecture'))]:
            if value is not None:
                rider[key] = value
    result['riders'] = riders
    result.update(forecast=None, performance_eligible=False,
                  status='closed' if result['close_at'] <= now.timestamp() else 'unavailable')
    if reason:
        result['unavailable_reason'] = reason
    return result


def _load_official(root):
    records = {}
    paths = sorted((root / 'outputs/official_results_archive').glob('????-??-??.json'))
    paths.append(root / 'outputs/latest_results.json')
    for path in paths:
        for row in read_json(path, []) or []:
            if not isinstance(row, dict):
                continue
            rid = str(row.get('race_id', ''))
            if not re.fullmatch(r'\d{12}', rid):
                continue
            if path.name != 'latest_results.json' and rid[-8:] != path.stem.replace('-', ''):
                raise ValueError('official archive race date mismatch')
            outcome = official_outcome(row, 'trifecta')
            if outcome is None or not outcome.get('observed_at'):
                continue
            try:
                at = timestamp(outcome['observed_at'])
            except (ValueError, TypeError):
                continue
            prior = records.get(rid)
            if prior is None or at >= timestamp(prior['observed_at']):
                records[rid] = outcome
    return records


def run(root=ROOT, *, clock=now_jst, race_date=None, confirm_existing=False, max_age=1800, min_lead=300):
    root = Path(root)
    now = clock()
    today = now.astimezone(JST).date().isoformat()
    _require(race_date is None or race_date == today, '--date must be the actual current JST date')
    path = root / OUTPUT
    feed = read_json(path)
    if feed is None:
        feed = {'schema': 1, 'model_id': MODEL_ID, 'today': today, 'generated_at': now.isoformat(),
                'reconstruction': True, 'method': copy.deepcopy(METHOD), 'races': []}
    _require(feed.get('model_id') == MODEL_ID and feed.get('schema') == 1, 'unexpected existing cloud feed')
    existing = {}
    for row in feed['races']:
        rid, pred = row['race_id'], row.get('forecast')
        _require(rid not in existing, 'duplicate existing race')
        existing[rid] = row
        if pred is None:
            _require(not row.get('performance_eligible') and not row.get('first_published_at'),
                     'unforecast schedule row cannot have publication credit')
            continue
        _require(isinstance(pred, dict) and pred.get('model_id') == MODEL_ID, 'foreign existing forecast')
        _require(row.get('forecast_sha256') == digest(pred), 'frozen forecast hash mismatch')
        if confirm_existing and not row.get('first_published_at') and timestamp(pred['saved_at']) <= now:
            if now.timestamp() < _epoch(row['close_at']):
                row['first_published_at'] = now.isoformat()
                row['performance_eligible'] = True
        if row.get('performance_eligible'):
            _require(row.get('first_published_at') and timestamp(pred['saved_at']) <= timestamp(row['first_published_at'])
                     and _epoch(row['first_published_at']) < _epoch(row['close_at']), 'invalid pre-close publication receipt')
    manifest_bytes = (root / 'models/nexus53/manifest.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    source = read_json(root / 'outputs/nexus_live.json', {})
    candidates = [row for row in source.get('races', []) if row.get('date') == today]
    failures, generated_ids = {}, set()
    candidate_map = {row['race_id']: row for row in candidates}
    _require(len(candidate_map) == len(candidates), 'duplicate source schedule row')
    for row in candidates:
        rid = row['race_id']
        if rid in existing and existing[rid].get('forecast') is not None:
            continue
        snapshot_path = root / 'data/raw/nexus_page_snapshots' / today / 'preclose' / (rid + '.json')
        snapshot = None
        try:
            snapshot = read_json(snapshot_path)
            _require(snapshot is not None, 'pre-close snapshot missing')
            new = build_forecast(row, snapshot, manifest, manifest_hash, clock=clock,
                                 max_age=max_age, min_lead=min_lead)
            generated_ids.add(rid)
        except (ValueError, KeyError, TypeError, AssertionError) as error:
            failures[rid] = str(error)
            new = schedule_row(row, snapshot, clock(), str(error))
        existing[rid] = new
    feed['races'] = sorted(existing.values(), key=lambda row: (row['date'], row['close_at'], row['race_id']))
    official = _load_official(root)
    for row in feed['races']:
        if row.get('forecast') is None:
            continue
        outcome = official.get(row['race_id'])
        if outcome is None or _epoch(outcome['observed_at']) > clock().timestamp():
            continue
        # Official cancellations may occur before close; normal results may not.
        if not outcome['void'] and _epoch(outcome['observed_at']) < _epoch(row['close_at']):
            continue
        prior = row.get('settlement')
        if prior and (prior.get('status') == 'void' and not outcome['void']):
            continue
        if prior and prior.get('settled_at') and _epoch(prior['settled_at']) > _epoch(outcome['observed_at']):
            continue
        settled = settlement(row, outcome)
        if settled is not None:
            settled['settled_at'] = outcome['observed_at']
            row['settlement'] = settled
    # Recheck after result loading, immediately before summaries and writing.
    # Persisted forecasts are never rewritten, even after close.
    final_now = clock()
    expired = {rid for rid in generated_ids
               if _epoch(existing[rid]['close_at']) - final_now.timestamp() <= max(300, min_lead)}
    for rid in expired:
        failures[rid] = 'deadline passed before feed write'
        row = existing[rid]
        row.update(forecast=None, performance_eligible=False,
                   status='closed' if row['close_at'] <= final_now.timestamp() else 'unavailable',
                   unavailable_reason=failures[rid])
        row.pop('forecast_sha256', None)
        row.pop('settlement', None)
    generated_ids -= expired
    if generated_ids:
        feed['generated_at'] = max((existing[rid]['forecast']['saved_at'] for rid in generated_ids), key=_epoch)
    feed['today'] = today
    feed['updated_at'] = clock().isoformat()
    today_rows = [row for row in feed['races'] if row['date'] == today]
    forecast_rows = [row for row in today_rows if row.get('forecast') is not None]
    histories = [row['forecast']['history_through'] for row in forecast_rows]
    feed['history_through'] = min(histories) if histories else None
    feed['coverage'] = {'date': today, 'source_races': len(candidates), 'forecast_count': len(forecast_rows),
                        'created': len(generated_ids), 'skipped': failures}
    feed['health'] = {'stage': 'ready' if forecast_rows else 'waiting',
                      'message': '再構築版。保存NEXUS確率に基づく暫定方式で、回収性能は未検証。',
                      'history_through': feed['history_through']}
    settled_at = [row['settlement']['settled_at'] for row in feed['races']
                  if row.get('settlement', {}).get('settled_at')]
    feed['results_updated_at'] = max(settled_at, key=_epoch) if settled_at else None
    feed['observed'] = summary(feed)
    save_json(path, feed)
    return feed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', help='must equal the actual current JST date')
    parser.add_argument('--confirm-existing-publication', action='store_true',
                        help='confirm only forecasts already fetched from the published repository')
    args = parser.parse_args()
    feed = run(race_date=args.date, confirm_existing=args.confirm_existing_publication)
    print(json.dumps({'model_id': MODEL_ID, 'today': feed['today'], 'coverage': feed['coverage'],
                      'observed': feed['observed']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
