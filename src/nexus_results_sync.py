"""Synchronize official results into frozen Nexus forecasts without prediction.

The latest-results feed supplies outcomes only. Its timestamps never replace
the forecast/odds publication timestamps, and existing refund/source receipts
remain attached to the original race record.
"""

import argparse
from copy import deepcopy
import csv
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import tempfile

from official_outcomes import normalize_outcome
from nexus_model.publication import settle_prediction


ROOT = Path(__file__).resolve().parents[1]
JST = timezone(timedelta(hours=9))


def _timestamp(value):
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, timezone.utc)
        parsed = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=JST)
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def _latest_stamp(values):
    valid = [(stamp, value) for value in values if (stamp := _timestamp(value)) is not None]
    return max(valid, key=lambda item: item[0])[1] if valid else None


def _read(path):
    with path.open(encoding='utf-8-sig') as handle:
        return json.load(handle)


def _whole_yen(value):
    try:
        amount = Fraction(str(value))
        return amount.numerator if amount > 0 and amount.denominator == 1 else None
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


def _outcome(result):
    outcome = normalize_outcome(result)
    if outcome is None:
        return None
    raw = result.get('payouts_trifecta_json', {})
    try:
        raw = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        raw = {}
    raw = raw if isinstance(raw, dict) else {}
    payouts = {}
    for ticket in outcome['payouts']:
        if ticket in raw:
            value = _whole_yen(raw[ticket])
        else:
            # The normalizer permits a single official legacy odds value.
            # Convert its decimal text exactly, without float multiplication.
            try:
                value = _whole_yen(Fraction(str(result.get('actual_trifecta_odds'))) * 100)
            except (ValueError, TypeError, ZeroDivisionError):
                value = None
        if value is not None:
            payouts[ticket] = value
    return dict(outcome, payouts=payouts,
                payout_complete=set(payouts) == set(outcome['winning_buys']))


def _date(row):
    value = str(row.get('date', ''))
    try:
        return datetime.strptime(value, '%Y-%m-%d').date().isoformat()
    except ValueError:
        race_id = str(row.get('race_id', ''))
        if len(race_id) == 12 and race_id.isdigit():
            try:
                return datetime.strptime(race_id[-8:], '%Y%m%d').date().isoformat()
            except ValueError:
                pass
    return None


def _schedule(outputs):
    path = outputs / 'latest_race_schedule.csv'
    if not path.exists():
        return {}
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return {str(row['race_id']): row for row in csv.DictReader(handle) if row.get('race_id')}


def _new_race(result, schedule):
    race_id = str(result.get('race_id', ''))
    meta = schedule.get(race_id, {})
    day = _date(meta) or _date(result)
    if not day:
        return None
    row = dict(race_id=race_id, date=day)
    for key in ('venue', 'source_url'):
        if meta.get(key) or result.get(key):
            row[key] = meta.get(key) or result[key]
    number = meta.get('race_no') or result.get('race_no')
    if not number and len(race_id) == 12 and race_id.isdigit():
        number = race_id[:2]
        row['venue_code'] = race_id[2:4]
    if number:
        row['race_no'] = int(number)
    for key in ('start_at', 'close_at'):
        value = meta.get(key)
        if value:
            try:
                row[key] = datetime.fromtimestamp(float(value), JST).isoformat()
            except (ValueError, OverflowError, OSError):
                pass
    return row


def _settle_saved(row):
    prediction = row.get('prediction')
    if not prediction:
        row.pop('settlement', None)
        return
    captured = _timestamp(prediction.get('captured_at'))
    close = _timestamp(row.get('close_at'))
    if captured is None or close is None or captured >= close:
        row.pop('settlement', None)
        row['settlement_unavailable_reason'] = '締切前の予想保存時刻を確認できません'
        return
    saved = prediction
    # Preserve legacy snapshots with missing optional metadata, but never score
    # an allocation known to have been saved or quoted after the cutoff.
    if any((stamp := _timestamp(prediction.get(key))) is not None and stamp >= close
           for key in ('odds_at', 'allocation_saved_at', 'auto_saved_at')):
        saved = dict(prediction, allocations={}, auto_budget=None)
    row['settlement'] = settle_prediction(
        saved, row.get('payouts', {}), row.get('refunded', ()),
        bool(row.get('refund_all') or row.get('cancelled')),
    )
    row.pop('settlement_unavailable_reason', None)


def _complete(row):
    if row.get('refund_all') or row.get('cancelled'):
        return True
    if 'result_complete' in row:
        return bool(row['result_complete'])
    winners, payouts = row.get('winners', []), row.get('payouts', {})
    return bool(winners) and set(winners) == set(payouts) and all(_whole_yen(value) for value in payouts.values())


def _merge_race(live_row, ledger_row):
    merged = dict(live_row, **deepcopy(ledger_row))
    if not (live_row.get('winners') or live_row.get('refund_all') or live_row.get('cancelled')):
        return merged
    if ledger_row.get('refund_all') or ledger_row.get('cancelled'):
        return merged
    live_explicit = live_row.get('result_sync_source') == 'nexus_page_snapshot' and _complete(live_row)
    ledger_explicit = ledger_row.get('result_sync_source') == 'nexus_page_snapshot' and _complete(ledger_row)
    if ledger_explicit and not live_explicit and not (live_row.get('refund_all') or live_row.get('cancelled')):
        return merged
    live_stamp = _timestamp(_latest_stamp([live_row.get('results_updated_at'), live_row.get('result_captured_at')]))
    ledger_stamp = _timestamp(_latest_stamp([ledger_row.get('results_updated_at'), ledger_row.get('result_captured_at')]))
    newer = live_stamp is not None and (ledger_stamp is None or live_stamp > ledger_stamp)
    prefer_live = ((live_explicit and not ledger_explicit) or
                   live_row.get('refund_all') or live_row.get('cancelled') or
                   (_complete(live_row) and not _complete(ledger_row)) or
                   (newer and not (_complete(ledger_row) and not _complete(live_row))))
    if prefer_live:
        # A feed and ledger can be read between separate publication writes.
        # Keep the ledger's frozen forecast, but not its older outcome.
        fields = {'winners', 'payouts', 'result_complete', 'payout_pending', 'settlement'}
        for key in fields:
            merged.pop(key, None)
        for key, value in live_row.items():
            if key in fields or key.startswith(('result_', 'results_')) or key in ('refunded', 'refund_all', 'cancelled'):
                merged[key] = deepcopy(value)
        merged['closed'] = True
        if _complete(merged):
            _settle_saved(merged)
        else:
            merged.pop('settlement', None)
    return merged


def _apply_result(row, outcome):
    if row.get('result_sync_source') == 'nexus_page_snapshot' and _complete(row):
        # Compact feeds can list 100-yen refunds as winning payouts. They lack
        # the explicit winning IDs / absence flags that certified this result.
        _settle_saved(row)
        return
    incoming_stamp = _timestamp(outcome.get('source_observed_at'))
    existing_stamp = _timestamp(_latest_stamp([
        row.get('results_updated_at'), row.get('result_captured_at'),
    ]))
    if incoming_stamp is not None and existing_stamp is not None and incoming_stamp < existing_stamp:
        return
    if row.get('refund_all') or row.get('cancelled'):
        # The compact result feed carries no refund details. Existing official
        # cancellation/refund evidence takes precedence over this feed.
        row['result_complete'] = True
        row['closed'] = True
        _settle_saved(row)
        return
    if _complete(row) and (not outcome['payout_complete'] or (existing_stamp is not None and incoming_stamp is None)):
        return
    row.update(winners=outcome['winning_buys'], payouts=outcome['payouts'],
               result_complete=outcome['payout_complete'],
               payout_pending=not outcome['payout_complete'], closed=True)
    observed = outcome.get('source_observed_at')
    if observed and _timestamp(observed):
        row['results_updated_at'] = observed
        row.setdefault('result_captured_at', observed)
    row['result_sync_source'] = 'latest_results.json'
    if outcome.get('result_source'):
        row.setdefault('result_source', outcome['result_source'])
    if outcome.get('source_url'):
        row.setdefault('result_source_url', outcome['source_url'])
    if outcome['payout_complete']:
        _settle_saved(row)
    else:
        # A known finish without every winning payout is not a settled bet.
        row.pop('settlement', None)
        row.pop('settlement_unavailable_reason', None)


def _apply_saved_outcome(outputs, row):
    """Prefer a complete, identity-checked original result over compact rows."""
    race_id, day = str(row.get('race_id', '')), _date(row)
    if (not day or len(race_id) != 12 or not race_id.isdigit()
            or race_id[4:] != day.replace('-', '') or row.get('refund_all') or row.get('cancelled')):
        return False
    path = outputs.parent / 'data/raw/nexus_page_snapshots' / day / 'outcomes' / (race_id + '.json')
    if not path.exists():
        return False
    snapshot = _read(path)
    if not isinstance(snapshot, dict):
        return False
    data, meta = snapshot.get('data', {}), snapshot.get('meta', {})
    race, schedule = data.get('race', {}), data.get('schedule', {})
    captured = _timestamp(snapshot.get('captured_at'))
    decided = _timestamp(race.get('decidedAt')) if race.get('decidedAt') else None
    if (snapshot.get('schema') != 1 or snapshot.get('source_kind') != 'existing_public_page_response'
            or str(meta.get('race_id')) != race_id or str(race.get('id')) != race_id
            or meta.get('date') != day or str(schedule.get('date')) != race_id[4:]
            or str(meta.get('venue_id')).zfill(2) != race_id[2:4]
            or meta.get('race_number') != int(race_id[:2]) or race.get('number') != int(race_id[:2])
            or not captured or not decided or captured < decided
            or data.get('trifectaPayoffStatus') != 1):
        return False
    old_capture = _timestamp(row.get('result_sync_captured_at'))
    if row.get('result_sync_source') == 'nexus_page_snapshot' and old_capture and captured < old_capture:
        return False
    entries, quotes = data.get('entries', []), data.get('trifecta', [])
    count = race.get('entriesNumber')
    if (type(count) is not int or not 3 <= count <= 9 or len(entries) != count
            or any(not isinstance(e, dict) or type(e.get('number')) is not int
                   or type(e.get('absent')) is not bool or str(e.get('raceId')) != race_id for e in entries)
            or sorted(e['number'] for e in entries) != list(range(1, count + 1))
            or len(quotes) != count * (count - 1) * (count - 2)):
        return False
    keys, ids = set(), set()
    for quote in quotes:
        key = quote.get('key', [])
        if (not isinstance(key, list) or len(key) != 3 or any(type(c) is not int or not 1 <= c <= count for c in key)
                or len(set(key)) != 3 or tuple(key) in keys or quote.get('id') in ids
                or type(quote.get('absent')) is not bool or str(quote.get('raceId')) != race_id):
            return False
        keys.add(tuple(key))
        ids.add(quote.get('id'))
    winning_ids = data.get('trifectaWinningOddsIds')
    if not isinstance(winning_ids, list) or not winning_ids or not set(winning_ids).issubset(ids):
        return False
    from nexus_model.source import outcomes
    payouts, refunded, refund_all, complete = outcomes(data)
    if (not complete or refund_all or set(payouts).intersection(refunded)
            or any(_whole_yen(value) is None for value in payouts.values())):
        return False
    row.update(winners=sorted(payouts), payouts={k: _whole_yen(v) for k, v in payouts.items()},
               refunded=sorted(refunded), result_complete=True, payout_pending=False, closed=True,
               result_sync_source='nexus_page_snapshot',
               result_sync_captured_at=snapshot['captured_at'])
    # Keep original result/source receipts intact; record correction provenance
    # separately, using only the observed capture time (never the repair time).
    if snapshot.get('sha256'):
        row['result_sync_source_sha256'] = snapshot['sha256']
    if snapshot.get('url'):
        row['result_sync_source_url'] = snapshot['url']
    row.setdefault('result_captured_at', snapshot['captured_at'])
    row['results_updated_at'] = _latest_stamp([row.get('results_updated_at'), snapshot['captured_at']])
    _settle_saved(row)
    return True


def _write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.' + path.name + '.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def sync_latest_results(outputs_dir=None):
    """Return a change summary; missing feeds are a harmless no-op.

    Corrupt JSON and I/O failures propagate to the caller. No source request,
    forecast inference, odds update or allocation calculation is performed.
    """
    outputs = Path(outputs_dir) if outputs_dir is not None else ROOT / 'outputs'
    summary = dict(changed=False, updated_races=0, ledger_files=[], live_changed=False,
                   official_results=0, complete_results=0, skipped_races=[], reason=None)
    results_path, live_path = outputs / 'latest_results.json', outputs / 'nexus_live.json'
    if not results_path.exists() or not live_path.exists():
        summary['reason'] = 'missing_results' if not results_path.exists() else 'missing_nexus_feed'
        return summary
    results, original_live = _read(results_path), _read(live_path)
    if not isinstance(results, list) or not isinstance(original_live, dict) or not isinstance(original_live.get('races'), list):
        raise ValueError('latest_results must be a list and nexus_live must contain a races list')
    live = deepcopy(original_live)
    schedule = _schedule(outputs)
    rows = {str(row['race_id']): deepcopy(row) for row in live['races']}
    order = list(rows)
    dates = {day for row in [*rows.values(), *results] if isinstance(row, dict) and (day := _date(row))}
    originals = {}
    for day in sorted(dates):
        path = outputs / 'nexus_forecasts' / (day + '.json')
        originals[day] = _read(path) if path.exists() else None
        if originals[day] is not None and not isinstance(originals[day], list):
            raise ValueError(f'{path.name} must contain a race list')
        for ledger_row in originals[day] or []:
            race_id = str(ledger_row['race_id'])
            if race_id not in rows:
                order.append(race_id)
            # The ledger owns frozen forecasts; retain other live-only fields.
            rows[race_id] = _merge_race(rows.get(race_id, {}), ledger_row)
    before_results = deepcopy(rows)
    stamps = [live.get('results_updated_at'), *(row.get('results_updated_at') for row in rows.values())]
    # Old-to-new processing makes duplicate source records deterministic.
    results = sorted((row for row in results if isinstance(row, dict)),
                     key=lambda row: _timestamp(row.get('result_observed_at_jst')) or datetime.min.replace(tzinfo=timezone.utc))
    for result in results:
        race_id = str(result.get('race_id', ''))
        outcome = _outcome(result)
        if race_id in rows:
            stamps.append(result.get('result_observed_at_jst'))
        if not outcome:
            continue
        summary['official_results'] += 1
        summary['complete_results'] += int(outcome['payout_complete'])
        if race_id not in rows:
            row = _new_race(result, schedule)
            if row is None:
                summary['skipped_races'].append(race_id)
                continue
            rows[race_id] = row
            order.append(race_id)
        stamps.append(result.get('result_observed_at_jst'))
        _apply_result(rows[race_id], outcome)
    summary['explicit_source_results'] = 0
    for row in rows.values():
        if _apply_saved_outcome(outputs, row):
            summary['explicit_source_results'] += 1
            stamps.append(row.get('results_updated_at'))
    summary['updated_races'] = sum(row != before_results.get(race_id) for race_id, row in rows.items())
    live['races'] = [rows[race_id] for race_id in order]
    updated = _latest_stamp(stamps)
    if updated:
        live['results_updated_at'] = updated
    grouped = {}
    for row in live['races']:
        day = _date(row)
        if day:
            grouped.setdefault(day, []).append(row)
    changes = []
    for day, day_rows in sorted(grouped.items()):
        if day_rows != originals.get(day):
            path = outputs / 'nexus_forecasts' / (day + '.json')
            changes.append((path, day_rows))
            summary['ledger_files'].append(str(path.relative_to(outputs)))
    if live != original_live:
        changes.append((live_path, live))
        summary['live_changed'] = True
    # Validate every output before replacing any file, then publish live last.
    encoded = [(path, json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False))
               for path, value in changes]
    for path, content in encoded:
        _write(path, content)
    summary['changed'] = bool(changes)
    summary['results_updated_at'] = live.get('results_updated_at')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outputs-dir', type=Path, default=ROOT / 'outputs')
    args = parser.parse_args()
    print(json.dumps(sync_latest_results(args.outputs_dir), ensure_ascii=False))


if __name__ == '__main__':
    main()
