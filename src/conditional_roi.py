"""Prospective eight-ticket challenger from the immutable conditional top12.

Only existing saved inputs are read. Historical research is never imported,
and neither the original forecast ledger nor its budgets are written here.
"""
import argparse
import copy
import hashlib
import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from conditional_budget import (BUDGET, MIN_BUDGET, UNIT, VERSION, RANGE_VERSION,
                                allocate, allocate_range, check_saved_plan,
                                fingerprint, number, settle, valid_tickets)
from nexus_input_snapshots import ROOT, JST, canonical, now_jst, read_json, save_json, timestamp
from nexus_model.forecast_confidence import from_saved
from nexus_results_sync import read_saved_page_outcome
from official_outcomes import normalize_outcome, ticket, ticket_return
from public_live import forecast_view, read_csv

FEED = Path('outputs/conditional_roi.json')
LEDGER = Path('outputs/company/conditional_live_ledger.jsonl')
POLICY_ID = 'conditional-roi-eight-v1'
SELECTED_RANKS = (1, 2, 3, 4, 9, 10, 11, 12)
MIN_LEAD = 60


def _now(clock):
    value = clock()
    if value.tzinfo is None:
        raise ValueError('timezone-aware clock required')
    return value.astimezone(JST)


def _finite(value):
    if isinstance(value, bool):
        raise ValueError('invalid timestamp')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('invalid timestamp')
    return value


def _hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _truth(value):
    return value is True or str(value).lower() in ('true', '1')


def _refund(value):
    return isinstance(value, dict) and (
        any(_truth(value.get(k)) for k in ('cancel', 'cancelled', 'refund_all'))
        or any(value.get(k) not in (None, '', '[]', [], {}, False)
               for k in ('cancelled_cars', 'cancelled_car_numbers', 'refunded', 'returned_buys')))


def _known(value):
    return isinstance(value, dict) and bool(
        _truth(value.get('official_result_available')) or value.get('actual')
        or value.get('payouts') or value.get('budget_official_outcome') or value.get('decided_at')
        or value.get('actual_trifecta') or value.get('actual_trifecta_buys') not in (None, '', '[]', []))


def _metadata(value):
    rid = str(value.get('race_id', value.get('id', '')))
    day = value.get('date')
    num = _finite(value.get('race_no', value.get('number')))
    close = _finite(value.get('close_at'))
    venue = value.get('venue')
    if (not re.fullmatch(r'\d{12}', rid) or not isinstance(day, str)
            or rid[4:] != day.replace('-', '') or num != int(num) or int(rid[:2]) != num
            or not isinstance(venue, str) or not venue
            or datetime.fromtimestamp(close, JST).date().isoformat() != day):
        raise ValueError('レースID・日付・会場・締切が不一致')
    start = value.get('start_at')
    return dict(id=rid, date=day, venue=venue, number=int(num), close_at=close,
                start_at=_finite(start) if start not in (None, '') else None)


def _same(meta, value):
    try:
        other = _metadata(value)
        return all(other[k] == meta[k] for k in ('id', 'date', 'venue', 'number', 'close_at'))
    except (ValueError, TypeError, OverflowError, OSError):
        return False


def _original_identity(original, meta, current):
    if not _same(meta, original):
        raise ValueError('元12点と開催情報が不一致')
    tickets = original.get('top12')
    probabilities = original.get('probabilities')
    if (not valid_tickets(tickets) or len(tickets) != 12
            or not isinstance(probabilities, list) or len(probabilities) != 12):
        raise ValueError('元12点と保存確率が不足')
    q = [number(value) for value in probabilities]
    if any(isinstance(v, bool) for v in probabilities) or sum(q) > 1 or any(a < b for a, b in zip(q, q[1:])):
        raise ValueError('保存確率の合計・順位が不正')
    saved = timestamp(original.get('snapshot_at_jst'))
    if (saved.timestamp() >= meta['close_at'] or saved > current
            or saved.astimezone(JST).date().isoformat() != meta['date']):
        raise ValueError('元予想の締切前保存を確認できません')
    model = original.get('model_sha256')
    cutoff = original.get('training_cutoff_exclusive')
    if (not re.fullmatch(r'[a-f0-9]{64}', str(model or '')) or not isinstance(cutoff, str)
            or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', cutoff) or cutoff > meta['date']):
        raise ValueError('元モデルの識別情報・学習期限が不正')
    return dict(id=meta['id'], date=meta['date'], venue=meta['venue'], number=meta['number'],
                close_at=meta['close_at'], snapshot_at_jst=original['snapshot_at_jst'],
                tickets=copy.deepcopy(tickets), probabilities=copy.deepcopy(probabilities),
                model_sha256=model, training_cutoff_exclusive=cutoff,
                source_url=original.get('source_url', ''))


def _stored_identity(row):
    return dict(id=row['id'], date=row['date'], venue=row['venue'], number=row['number'],
                close_at=row['close_at'], snapshot_at_jst=row['original_snapshot_at_jst'],
                tickets=row['original_tickets'], probabilities=row['original_probabilities'],
                model_sha256=row['original_model_sha256'],
                training_cutoff_exclusive=row['original_training_cutoff_exclusive'],
                source_url=row['original_source_url'])


def _frozen_signature(row):
    return _hash(dict(original_fingerprint=row['original_fingerprint'],
                      snapshot_at_jst=row['snapshot_at_jst'], tickets=row['shadow']['tickets'],
                      probabilities=row['probabilities']))


def _validate_saved(row, activated=None):
    if not row.get('shadow'):
        return
    _metadata(row)
    original = _stored_identity(row)
    selected = [original['tickets'][i - 1] for i in SELECTED_RANKS]
    q = [original['probabilities'][i - 1] for i in SELECTED_RANKS]
    saved = timestamp(row['snapshot_at_jst'])
    if (row.get('forecast_status') != 'saved' or len(original['tickets']) != 12
            or not valid_tickets(original['tickets']) or not valid_tickets(selected)
            or row['shadow']['tickets'] != selected or row['probabilities'] != q
            or row['shadow'].get('snapshot_at') != row['snapshot_at_jst']
            or row['original_fingerprint'] != _hash(original)
            or row.get('forecast_fingerprint') != _frozen_signature(row)
            or row['close_at'] - saved.timestamp() < MIN_LEAD
            or saved < timestamp(row['original_snapshot_at_jst'])
            or (activated is not None and saved.timestamp() < activated)
            or saved.astimezone(JST).date().isoformat() != row['date']):
        raise ValueError('保存済み8点予想の整合性エラー: ' + row['id'])
    plan = row.get('budget') or {}
    if plan.get('status') == 'allocated' and not check_saved_plan(plan, selected):
        raise ValueError('保存済み8点配分の整合性エラー: ' + row['id'])
    if plan:
        allocated_at = timestamp(plan.get('allocation_frozen_at_jst'))
        initial = row.get('initial_budget')
        if (plan.get('status') not in ('allocated', 'infeasible') or allocated_at < saved
                or row['close_at'] - allocated_at.timestamp() < MIN_LEAD
                or row.get('budget_fingerprint') != _hash(plan)
                or row.get('initial_budget_fingerprint') != _hash(initial)
                or plan.get('original_budget_sha256') != _hash(initial)):
            raise ValueError('保存済み配分の時刻・内容が変更されています: ' + row['id'])
        quotes = _verified_quote(initial, original, allocated_at)
        if (set(plan.get('snapshot_odds', {})) != set(selected)
                or any(number(plan['snapshot_odds'][t]) != number(quotes[t]) for t in selected)
                or plan.get('odds_captured_at_jst') != initial['odds_captured_at_jst']):
            raise ValueError('保存済み8点配分のオッズが元記録と不一致: ' + row['id'])


def _new_forecast(row, original, identity, current):
    tickets = [identity['tickets'][i - 1] for i in SELECTED_RANKS]
    probabilities = [identity['probabilities'][i - 1] for i in SELECTED_RANKS]
    stamp = current.isoformat()
    for key, value in identity.items():
        if key in ('tickets', 'probabilities', 'snapshot_at_jst', 'model_sha256', 'training_cutoff_exclusive', 'source_url'):
            row['original_' + key] = copy.deepcopy(value)
    row.update(original_fingerprint=_hash(identity), snapshot_at_jst=stamp,
               probabilities=probabilities, selected_ranks=list(SELECTED_RANKS),
               forecast_status='saved', forecast_error='', budget=None, initial_budget=None,
               budget_status='waiting', budget_error='保存済みオッズを待っています',
               riders=copy.deepcopy(original.get('riders') or []))
    row.pop('budget_fingerprint', None)
    row.pop('initial_budget_fingerprint', None)
    view_input = dict(top12=tickets, probabilities=probabilities,
                      snapshot_at_jst=stamp, close_at=row['close_at'])
    row['shadow'] = forecast_view(view_input)
    row['shadow'].update(confidence=from_saved(view_input, annotated_at=stamp),
                         rule=POLICY_ID, note='回収重視8点：元の確率順位1〜4位と9〜12位。')
    row['forecast_fingerprint'] = _frozen_signature(row)


def _source_url(plan, identity):
    parsed = urlsplit(str(plan.get('source_url', '')))
    rid = identity['id']
    if parsed.scheme != 'https' or parsed.username or parsed.password:
        return False
    if plan.get('provider') == 'winticket':
        match = re.fullmatch(r'/keirin/[^/]+/racecard/(\d{10})/(\d+)/(\d+)/?', parsed.path)
        source_day = ((datetime.strptime(match[1][:8], '%Y%m%d') + timedelta(days=int(match[2]) - 1)).date().isoformat()
                      if match and int(match[2]) >= 1 else None)
        return bool(parsed.hostname == 'www.winticket.jp' and match
                    and match[1][-2:] == rid[2:4] and int(match[3]) == identity['number']
                    and source_day == identity['date']
                    and plan['source_url'] == identity['source_url'])
    if plan.get('provider') == 'oddspark':
        query = parse_qs(parsed.query)
        expected = dict(betType='9', joCode=rid[2:4], kaisaiBi=rid[4:], raceNo=str(identity['number']))
        return parsed.hostname == 'www.oddspark.com' and parsed.path == '/keirin/Odds.do' and all(query.get(k) == [v] for k, v in expected.items())
    return False


def _verified_quote(plan, identity, current):
    if not isinstance(plan, dict) or plan.get('status') not in ('allocated', 'infeasible'):
        raise ValueError('元予想の保存済みオッズを待っています')
    tickets = identity['tickets']
    captured, frozen = timestamp(plan['odds_captured_at_jst']), timestamp(plan['allocation_frozen_at_jst'])
    if (plan.get('version') not in (VERSION, RANGE_VERSION) or plan.get('unit_yen') != UNIT
            or plan.get('ticket_sha256') != fingerprint(tickets)
            or not _source_url(plan, identity)
            or not 0 <= (frozen - captured).total_seconds() <= 180
            or any(t > current or t.timestamp() >= identity['close_at']
                   or t.astimezone(JST).date().isoformat() != identity['date'] for t in (captured, frozen))):
        raise ValueError('オッズの出典・元予想・締切前時刻が不一致')
    odds = plan.get('snapshot_odds')
    if not isinstance(odds, dict) or set(odds) != set(tickets):
        raise ValueError('元12点のオッズが不足')
    quotes = {t: str(number(odds[t])) for t in tickets}
    if any(number(v) >= number('9999.9') for v in quotes.values()):
        raise ValueError('上限表示オッズは配分に利用しません')
    if plan['status'] == 'allocated' and not check_saved_plan(plan, tickets):
        raise ValueError('元配分の購入額・保存内容が不正')
    if (plan['status'] == 'infeasible' and (plan.get('allocations') != []
            or plan.get('total_stake_yen') != 0
            or type(plan.get('budget_yen')) is not int
            or not MIN_BUDGET <= plan['budget_yen'] <= BUDGET or plan['budget_yen'] % UNIT
            or (plan['version'] == VERSION and plan['budget_yen'] != BUDGET))):
        raise ValueError('配分不成立の元記録に購入額があります')
    return quotes


def _attempt_allocation(row, original, source, current, clock):
    if row.get('budget') and row['budget'].get('status') in ('allocated', 'infeasible'):
        return
    if row['close_at'] - current.timestamp() < MIN_LEAD:
        row.update(budget_status='expired', budget_error='締切前の配分保存なし・後付け配分はしません')
        return
    if row.get('refund_pending') or row.get('official_outcome') or _known(original) or _known(source):
        row.update(budget_status='excluded', budget_error='結果・返還情報があるため新規配分を作成しません')
        return
    identity = _stored_identity(row)
    try:
        live_identity = _original_identity(original, row, current)
        if _hash(live_identity) != row['original_fingerprint']:
            raise ValueError('保存後に元12点が変わったため配分を保留')
        plans = [original.get('budget')]
        if (_same(row, source) and (source.get('shadow') or {}).get('tickets') == identity['tickets']
                and source['shadow'].get('snapshot_at') == identity['snapshot_at_jst']):
            plans.append(source.get('budget'))
        accepted, errors = None, []
        for candidate in plans:
            try:
                quotes = _verified_quote(candidate, identity, current)
                accepted = candidate
                break
            except (ValueError, TypeError, KeyError) as error:
                errors.append(str(error))
        if accepted is None:
            raise ValueError(errors[0] if errors else '保存済みオッズを待っています')
        tickets = row['shadow']['tickets']
        same_budget = accepted['status'] == 'allocated'
        plan = (allocate(tickets, quotes, budget=accepted['budget_yen']) if same_budget
                else allocate_range(tickets, quotes))
        saved = _now(clock)
        if (row['close_at'] - saved.timestamp() < MIN_LEAD or saved.date().isoformat() != row['date']
                or saved < current or saved < timestamp(row['snapshot_at_jst'])):
            raise ValueError('計算中に締切が近づいたため配分を保存しません')
        plan.update(version=accepted['version'] if same_budget else RANGE_VERSION,
                    budget_selection='same_as_original' if same_budget else 'smallest_feasible_no_loss',
                    comparison_policy_id=POLICY_ID, source_prediction_snapshot_at_jst=identity['snapshot_at_jst'],
                    odds_captured_at_jst=accepted['odds_captured_at_jst'],
                    allocation_frozen_at_jst=saved.isoformat(), allocation_attempted_at_jst=current.isoformat(),
                    source_url=accepted['source_url'], provider=accepted['provider'],
                    source_race_id=row['id'], original_budget_sha256=_hash(accepted))
        if plan['status'] == 'allocated' and not check_saved_plan(plan, tickets):
            raise ValueError('8点配分の保存内容が不正')
        row.update(budget=plan, initial_budget=copy.deepcopy(accepted),
                   budget_fingerprint=_hash(plan), initial_budget_fingerprint=_hash(accepted),
                   budget_status=plan['status'], budget_error='')
    except (ValueError, TypeError, KeyError) as error:
        row.update(budget_status='waiting', budget_error=str(error)[:240])


def _flat(tickets, outcome, refund=False):
    empty = dict(mode='simulation', status='not_predicted', invested_yen=0,
                 payout_yen=None, profit_yen=None, roi=None, hit=None)
    if not tickets:
        return empty
    empty.update(invested_yen=UNIT * len(tickets), status='pending')
    if refund:
        return {**empty, 'status': 'refund_pending'}
    if not outcome:
        return empty
    if not outcome.get('payout_complete'):
        return {**empty, 'status': 'payout_pending'}
    try:
        wins = outcome['winning_buys']
        if not wins or any(not ticket(t) or number(outcome['payouts'][t]) % 1 for t in wins):
            raise ValueError('invalid official payout')
        hit, paid = ticket_return([dict(buy=t, stake_yen=UNIT) for t in tickets], outcome)
        paid = int(paid)
    except (ValueError, TypeError, KeyError, OverflowError):
        return {**empty, 'status': 'payout_pending'}
    return {**empty, 'status': 'settled', 'hit': hit, 'payout_yen': paid,
            'profit_yen': paid - empty['invested_yen'], 'roi': paid / empty['invested_yen']}


def _result_matches(row, result):
    if not isinstance(result, dict) or str(result.get('race_id', result.get('id', ''))) != row['id']:
        return False
    for key, alias in (('date', 'date'), ('venue', 'venue'), ('number', 'race_no')):
        value = result.get(alias, result.get(key))
        if value not in (None, ''):
            try:
                matches = _finite(value) == row[key] if key == 'number' else str(value) == str(row[key])
                if not matches:
                    return False
            except (ValueError, TypeError):
                return False
    return True


def _observable_outcome(row, outcome, current):
    if not isinstance(outcome, dict):
        return False
    try:
        observed = timestamp(outcome.get('source_observed_at')).timestamp()
        if not row['close_at'] < observed <= current.timestamp():
            return False
        decided = outcome.get('decided_at')
        if decided not in (None, '', 0, '0', False):
            if isinstance(decided, str):
                try:
                    decided = timestamp(decided).timestamp()
                except ValueError:
                    decided = _finite(decided)
            else:
                decided = _finite(decided)
            if not row['close_at'] < decided <= observed:
                return False
    except (ValueError, TypeError, OverflowError):
        return False
    return True


def _settlements(row, original, source, result, evidence=None, nexus=None, *, current):
    inputs = [v for v in (original, source, result, nexus) if _result_matches(row, v)]
    if any(_refund(v) for v in inputs) or _refund(evidence):
        row['refund_pending'] = True
    cancelled = set(map(str, row.get('cancelled_cars', [])))
    for value in inputs + ([evidence] if evidence else []):
        cancelled.update(re.findall(r'[1-9]', str(value.get('cancelled_cars', []))))
    row['cancelled_cars'] = sorted(cancelled)
    outcomes = [row.get('official_outcome'), (evidence or {}).get('outcome')]
    for value in inputs:
        outcomes.extend((value.get('budget_official_outcome'), normalize_outcome(value)))
    def priority(outcome):
        try:
            observed = timestamp(outcome.get('source_observed_at', '')).timestamp()
        except (ValueError, TypeError):
            observed = float('-inf')
        return (bool(outcome.get('payout_complete')),
                outcome.get('evidence') == 'existing_public_page_response', observed)
    outcomes = [v for v in outcomes if _observable_outcome(row, v, current)]
    if outcomes:
        outcome = copy.deepcopy(max(outcomes, key=priority))
        winners = [t for t in outcome.get('winning_buys', []) if ticket(t) and not cancelled.intersection(t.split('-'))]
        outcome['winning_buys'] = winners
        outcome['payouts'] = {t: outcome.get('payouts', {})[t] for t in winners if t in outcome.get('payouts', {})}
        outcome['payout_complete'] = bool(winners) and outcome.get('payout_complete', False) and set(outcome['payouts']) == set(winners)
        row['official_outcome'] = outcome
        row.update(actual=winners, payouts=outcome['payouts'], result_observed_at=outcome.get('source_observed_at'))
    elif row.get('official_outcome'):
        row.pop('official_outcome', None)
        row.update(actual=[], payouts={}, result_observed_at=None)
    outcome = row.get('official_outcome')
    new_tickets = (row.get('shadow') or {}).get('tickets', [])
    old_tickets = row.get('original_tickets', []) if new_tickets else []
    row['flat_settlement'] = _flat(new_tickets, outcome, row.get('refund_pending', False))
    row['initial_flat_settlement'] = _flat(old_tickets, outcome, row.get('refund_pending', False))
    for plan_key, tickets, key in (('budget', new_tickets, 'budget_settlement'),
                                    ('initial_budget', old_tickets, 'initial_budget_settlement')):
        plan = row.get(plan_key)
        settled = settle(plan, tickets, outcome)
        if row.get('refund_pending') and plan and plan.get('status') == 'allocated':
            settled = settle(plan, tickets, None)
            settled['status'] = 'refund_pending'
        row[key] = settled


def _metrics(rows, key):
    values = [r[key] for r in rows]
    invested = sum(v['invested_yen'] for v in values)
    paid = sum(v['payout_yen'] for v in values)
    hits = sum(bool(v['hit']) for v in values)
    return dict(races=len(values), hits=hits, hit_rate=hits / len(values) if values else None,
                invested_yen=invested, payout_yen=paid, profit_yen=paid - invested,
                roi=paid / invested if invested else None)


def _write(root, payload, current):
    races = payload['races']
    races.sort(key=lambda r: (r['date'], r.get('start_at') or r.get('close_at') or 1e12, r['id']))
    comparison = dict(scope='prospectively_saved_policy_only', registered=sum(bool(r.get('shadow')) for r in races))
    for label, old, new in (('flat_100', 'initial_flat_settlement', 'flat_settlement'),
                            ('allocation', 'initial_budget_settlement', 'budget_settlement')):
        paired = [r for r in races if r.get('shadow') and all(r.get(k, {}).get('status') == 'settled' for k in (old, new))
                  and (label != 'allocation' or r[old]['invested_yen'] == r[new]['invested_yen'])]
        comparison[label] = dict(paired_races=len(paired), initial=_metrics(paired, old), roi_eight=_metrics(paired, new))
    payload.update(updated_at=current.isoformat(), generated_at=current.isoformat(), comparison=comparison)
    payload['coverage'] = dict(races=len(races), saved=comparison['registered'],
                              missing=sum(not r.get('shadow') for r in races),
                              allocated=sum((r.get('budget') or {}).get('status') == 'allocated' for r in races))
    save_json(Path(root) / FEED, payload)


def run(root=ROOT, clock=now_jst):
    root, current = Path(root), _now(clock)
    baseline = read_json(root / FEED, {})
    if baseline and (baseline.get('schema') != 1 or baseline.get('policy', {}).get('id') != POLICY_ID):
        raise ValueError('unexpected conditional ROI policy')
    payload = copy.deepcopy(baseline) or dict(schema=1, policy=dict(id=POLICY_ID,
        activated_at=current.isoformat(), selected_ranks=list(SELECTED_RANKS), minimum_lead_seconds=MIN_LEAD,
        mode='simulation', backfill=False, allocation='same_original_budget_else_smallest_feasible',
        quote_policy='existing_verified_preclose_quote_once'), races=[])
    payload['model'] = dict(id='conditional_trifecta_v1', name='条件付き3連単・回収重視8点', ticket_count=8, policy_id=POLICY_ID)
    originals, rows = {}, {r['id']: r for r in payload['races']}
    if len(rows) != len(payload['races']):
        raise ValueError('duplicate saved ROI race')
    activated = timestamp(payload['policy']['activated_at']).timestamp()
    for row in rows.values():
        _validate_saved(row, activated)
    path = root / LEDGER
    for line in path.read_text(encoding='utf-8-sig').splitlines() if path.exists() else []:
        if line.strip():
            value = json.loads(line)
            rid = str(value['race_id'])
            if rid in originals:
                raise ValueError('duplicate original conditional forecast')
            originals[rid] = value
    feed = read_json(root / 'outputs/conditional_live.json', {})
    sources = {str(r['id']): r for r in feed.get('races', [])}
    if len(sources) != len(feed.get('races', [])):
        raise ValueError('duplicate source race')
    metadata, invalid = {}, {}
    for value in list(originals.values()) + list(sources.values()) + read_csv(root / 'outputs/latest_race_schedule.csv'):
        rid = str(value.get('race_id', value.get('id', '')))
        try:
            meta = _metadata(value)
            if rid in metadata and not _same(metadata[rid], value):
                invalid[rid] = '元予想・公開データ・開催情報の一致を確認できません'
            metadata[rid] = meta
        except (ValueError, TypeError, OverflowError, OSError) as error:
            invalid[rid] = str(error)
            if rid and rid not in metadata:
                def optional(value):
                    try:
                        return _finite(value)
                    except (ValueError, TypeError):
                        return None
                num = optional(value.get('race_no', value.get('number')))
                metadata[rid] = dict(id=rid, date=str(value.get('date') or current.date().isoformat()),
                                     venue=str(value.get('venue') or ''), number=int(num) if num is not None else 0,
                                     close_at=optional(value.get('close_at')), start_at=optional(value.get('start_at')))
    for rid, row in rows.items():
        metadata.setdefault(rid, {k: row.get(k) for k in ('id', 'date', 'venue', 'number', 'start_at', 'close_at')})
    results = {str(r['race_id']): r for r in read_json(root / 'outputs/latest_results.json', [])}
    nexus = {str(r.get('race_id')): r for r in read_json(root / 'outputs/nexus_live.json', {}).get('races', [])}
    for rid, meta in metadata.items():
        original, source, result = originals.get(rid, {}), sources.get(rid, {}), results.get(rid, {})
        row = rows.setdefault(rid, {**meta, 'shadow': None, 'riders': copy.deepcopy(source.get('riders') or original.get('riders') or []),
                                    'budget': None, 'initial_budget': None, 'actual': [], 'payouts': {}, 'cancelled_cars': []})
        if row.get('shadow') and not _same(row, meta):
            invalid[rid] = '保存済み予想から開催情報が変わっています'
        evidence = read_saved_page_outcome(root, dict(race_id=rid, date=row['date'], race_no=row['number'], close_at=row['close_at']))
        if not row.get('shadow'):
            row.update(meta)
            current = _now(clock)
            status, error = 'missing', '元12点の締切前予想を待っています'
            if rid in invalid:
                status, error = 'invalid', invalid[rid]
            elif row['close_at'] <= activated:
                status, error = 'pre_policy', '導入前のレース・8点予想は後付けしません'
            elif row['close_at'] - current.timestamp() < MIN_LEAD:
                status, error = 'expired', '締切前の8点保存なし・後付け予想はしません'
            elif row['date'] != current.date().isoformat():
                status, error = 'missing', '対象日の元予想を待っています'
            elif evidence or _known(original) or _known(result) or _known(source) or any(_refund(v) for v in (original, result, source, nexus.get(rid, {}))):
                status, error = 'invalid', '結果・返還情報があるため新しい8点を作成しません'
            elif original:
                try:
                    identity = _original_identity(original, row, current)
                    _new_forecast(row, original, identity, current)
                    status, error = 'saved', ''
                except (ValueError, TypeError, KeyError, OverflowError) as exc:
                    status, error = 'invalid', str(exc)[:240]
            row.update(forecast_status=status, forecast_error=error)
        _settlements(row, original, source, result, evidence, nexus.get(rid), current=_now(clock))
        if row.get('shadow') and rid not in invalid:
            _attempt_allocation(row, original, source, _now(clock), clock)
            _settlements(row, original, source, result, evidence, nexus.get(rid), current=_now(clock))
        elif row.get('shadow') and rid in invalid:
            row['budget_error'] = invalid[rid]
    payload['races'] = list(rows.values())
    payload['schedule_date'] = max((r['date'] for r in metadata.values()), default=current.date().isoformat())
    _write(root, payload, _now(clock))
    gate_publication(root, baseline, clock=clock)
    return read_json(root / FEED)


def gate_publication(root, baseline, clock=now_jst, lead_seconds=60):
    """Discard private new forecasts/plans that can no longer publish in time."""
    root, current = Path(root), _now(clock)
    payload = read_json(root / FEED)
    if not payload:
        return []
    before = {r['id']: r for r in (baseline or {}).get('races', [])}
    dropped = []
    for row in payload['races']:
        old = before.get(row['id'], {})
        if row.get('close_at') is None:
            continue
        if row['close_at'] - current.timestamp() >= max(1, lead_seconds):
            continue
        if row.get('shadow') and not old.get('shadow'):
            row.update(shadow=None, snapshot_at_jst=None, probabilities=[], budget=None, initial_budget=None,
                       forecast_status='expired', forecast_error='公開前に締切が近づいたため新規8点を破棄',
                       budget_status='expired', budget_error='公開できる時間がありません')
            row.pop('forecast_fingerprint', None)
            row.pop('budget_fingerprint', None)
            row.pop('initial_budget_fingerprint', None)
            dropped.append(row['id'])
        elif row.get('budget') and not old.get('budget'):
            row.update(budget=None, initial_budget=copy.deepcopy(old.get('initial_budget')),
                       budget_status='expired', budget_error='公開前に締切が近づいたため新規配分を破棄')
            row.pop('budget_fingerprint', None)
            row.pop('initial_budget_fingerprint', None)
            dropped.append(row['id'])
        if row['id'] in dropped:
            _settlements(row, {}, {}, {}, current=current)
    if dropped:
        _write(root, payload, current)
    return dropped


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    report = run(parser.parse_args().root)
    print(json.dumps(report['coverage'], ensure_ascii=False))
