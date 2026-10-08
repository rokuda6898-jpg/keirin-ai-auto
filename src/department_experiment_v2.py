"""Prospective v2 experiment. Reporting uses only the standard library.

Old predictions are never regenerated. Fixed-six comparisons and matched-risk
comparisons are frozen before close; insufficient candidate sets remain absent.
"""
import hashlib
import html
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from itertools import permutations
from pathlib import Path
from zoneinfo import ZoneInfo

from official_outcomes import normalize_outcome, ticket_return

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'department_positions_v2'
LEDGER = 'annual_position_v2_ledger.jsonl'
ATTEMPTS = 'annual_position_v2_attempts.jsonl'
OUTCOMES = 'annual_position_v2_outcomes.jsonl'
PRODUCER = 'predict_canonical_v2'
DEPARTMENTS = ('data_department', 'pace_department', 'line_department')
VARIANTS = ('baseline', 'preserve', 'conditional', 'combined', 'context', 'market')
GROUPS = ('本線', '穴')
RULES = {'fixed_tickets_per_group': 6, 'stake_per_ticket': 100, 'maximum_tickets': 12,
         'minimum_hole_odds': 100, 'minimum_main_ev': 1.10, 'minimum_hole_ev': 1.25,
         'quote_age_seconds': 300, 'capture_minutes': [5, 40], 'evaluation_days': 56,
         'minimum_paired_races': 500, 'minimum_racing_days': 28, 'family_alpha': .05,
         'primary_tests': 12, 'resamples': 12000, 'automatic_promotion': False}
SOURCES = ('annual_knowledge.py', 'betting_logic.py', 'high_payout_strategy.py',
           'department_ticket_v2.py', 'department_context.py', 'department_experiment_v2.py',
           'official_outcomes.py', 'position_market.py', 'quote_quality.py', 'predict.py')
LABELS = {'baseline': '共通選別・従来評価', 'preserve': '①部署評価を保持',
          'conditional': '②条件付き市場評価', 'combined': '①＋②',
          'context': '①＋②＋専門展開', 'market': '市場支持のみ', 'risk': 'リスク部・現行'}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def code_provenance():
    files = {name: hashlib.sha256((ROOT / 'src' / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
             for name in SOURCES}
    return {'files': files, 'sha256': digest(files)}


def seal(row):
    row = {k: v for k, v in row.items() if k != 'record_sha256'}
    return {**row, 'record_sha256': digest(row)}


def read_lines(path):
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            try:
                yield json.loads(line)
            except (ValueError, TypeError):
                yield None


def append(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(seal(row), ensure_ascii=False, allow_nan=False) + '\n')


def frame_records(frame):
    """Preserve binary float values through JSON; pandas.to_json rounds prices."""
    def clean(value):
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if hasattr(value, 'item'):
            return clean(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return value
    return clean(frame.to_dict(orient='records'))


def quote_view(race, market, now):
    cars = sorted(int(c) for c in race.car_no)
    expected = {'-'.join(map(str, p)) for p in permutations(cars, 3)}
    raw = frame_records(market)
    observed = [r for r in raw if str(r.get('race_id')) == str(race.iloc[0].race_id)
                and r.get('bet_type') == 'trifecta']
    count = Counter(str(r.get('buy')) for r in observed)
    prices, stamps, reasons = {}, {}, Counter()
    for r in observed:
        buy = str(r.get('buy'))
        try:
            price = float(r.get('odds_used'))
            stamp = datetime.fromisoformat(str(r.get('odds_captured_at_jst')))
            if buy not in expected or count[buy] != 1:
                reasons['duplicate_or_unexpected'] += 1
            elif not math.isfinite(price) or not 1 <= price < 9999.9:
                reasons['missing_or_capped_price'] += 1
            elif (stamp.tzinfo is None or not 0 <= now.timestamp() - stamp.timestamp() <= 300
                  or stamp.timestamp() >= float(race.iloc[0].close_at)):
                reasons['invalid_quote_time'] += 1
            elif str(r.get('odds_verification_status', '')).startswith('excluded'):
                reasons['excluded_source'] += 1
            else:
                prices[buy], stamps[buy] = price, stamp.isoformat()
        except (ValueError, TypeError, OverflowError):
            reasons['invalid_or_missing_quote'] += 1
    complete = set(prices) == expected and len(set(stamps.values())) == 1
    return prices, stamps, {'expected': len(expected), 'observed': len(observed), 'usable': len(prices),
                            'complete_single_snapshot': complete, 'rejected': dict(reasons)}, raw


def make_record(race, inputs, market, now, knowledge, provenance):
    from betting_logic import score_riders, select_race
    from department_ticket_v2 import candidate_portfolios, take
    from department_context import context_for
    if provenance.get('producer') != PRODUCER:
        return None, 'noncanonical_producer'
    if now.tzinfo is None or not 300 < float(race.iloc[0].close_at) - now.timestamp() <= 2400:
        return None, 'outside_40_to_5_minutes'
    prices, stamps, quality, raw = quote_view(race, market, now)
    portfolios, risk_tickets, legacy, contexts = {}, [], {}, {}
    for department, temp in inputs.items():
        default = score_riders(temp, market)
        frame, policy = select_race(default, market)
        tickets = [{'buy': str(t.buy), 'group': str(t.ticket_group), 'stake_yen': 100,
                    'odds': float(t.odds_used), 'prob': float(t.prob), 'ev': float(t.ev)}
                   for t in frame[frame.is_selected].itertuples()]
        if department == 'risk_department':
            risk_tickets = tickets
            continue
        legacy[department] = tickets
        preserved = score_riders(temp, market, preserve_position_scores=True)
        pack = lambda f: [{'car_no': int(r.car_no), 'first': float(r.score_first),
                           'second': float(r.score_second), 'third': float(r.score_third)} for r in f.itertuples()]
        base_scores, preserved_scores = pack(default), pack(preserved)
        context, contexts[department] = context_for(department, temp, knowledge.get('profiles', {}))
        for variant in VARIANTS:
            use_preserve = variant in ('preserve', 'combined', 'context')
            conditional = variant in ('conditional', 'combined', 'context')
            result = candidate_portfolios(preserved_scores if use_preserve else base_scores, prices, policy,
                        conditional=conditional, market_only=variant == 'market',
                        context=context if variant == 'context' else None)
            if (conditional or variant == 'market') and not quality['complete_single_snapshot']:
                result = {'available': False, 'reason': 'complete_single_market_required', 'pools': {g: [] for g in GROUPS}}
            if variant == 'context' and context is None:
                result = {'available': False, 'reason': contexts[department]['status'], 'pools': {g: [] for g in GROUPS}}
            portfolios[department + ':' + variant] = result
    views = {name: {g: {} for g in GROUPS} for name in ('fixed_six', 'risk_matched')}
    for group in GROUPS:
        risk = [t for t in risk_tickets if t['group'] == group]
        eligible_risk = bool(risk) and all(prices.get(t['buy']) == t['odds'] for t in risk)
        views['risk_matched'][group]['risk'] = risk if eligible_risk else None
        for key, portfolio in portfolios.items():
            views['fixed_six'][group][key] = take(portfolio, group, RULES['fixed_tickets_per_group'])
            views['risk_matched'][group][key] = take(portfolio, group, len(risk)) if eligible_risk else None
    inputs_json = {d: frame_records(f) for d, f in inputs.items()}
    evidence = {'inputs': inputs_json, 'quotes': raw, 'usable_quotes': prices, 'quote_times': stamps,
                'knowledge_cutoff': knowledge['window_end_exclusive'], 'knowledge_sha256': knowledge.get('fingerprint')}
    row = {'version': VERSION, 'producer': PRODUCER, 'race_id': str(race.iloc[0].race_id),
           'date': str(race.iloc[0].get('date', now.date().isoformat())),
           'venue': str(race.iloc[0].get('venue', '')), 'race_no': int(race.iloc[0].get('race_no', 0)),
           'close_at': float(race.iloc[0].close_at), 'snapshot_at': now.isoformat(),
           'window_start': now.isoformat(),
           'strata': {'field_size': len(race), 'race_class': str(race.iloc[0].get('race_class', 'unknown')),
                      'race_type': str(race.iloc[0].get('race_type', 'unknown'))},
           'rules': RULES, 'code': code_provenance(), 'models': provenance['models'],
           'model_source': provenance.get('model_source'), 'position_source': provenance.get('position_source'),
           'evidence': evidence, 'input_sha256': digest(evidence), 'quote_quality': quality,
           'risk_tickets': risk_tickets, 'legacy_department_tickets': legacy,
           'arms': portfolios, 'views': views, 'context_evidence': contexts,
           'purchase_authorized': False}
    return seal(row), 'eligible' if quality['complete_single_snapshot'] else 'partial_market_recorded'


def record_attempt(race, reason, now, output_dir=ROOT / 'outputs', **details):
    append(Path(output_dir) / 'company' / ATTEMPTS, {'version': VERSION, 'race_id': str(race.iloc[0].race_id),
        'at': now.isoformat(), 'reason': reason, 'field_size': len(race),
        'race_class': str(race.iloc[0].get('race_class', 'unknown')), **details})


def append_record(row, completed, output_dir=ROOT / 'outputs'):
    if row is None or not 300 < row['close_at'] - completed.timestamp() <= 2400:
        return False
    if any(not 0 <= completed.timestamp() - datetime.fromisoformat(t).timestamp() <= 300
           for t in row['evidence']['quote_times'].values()):
        return False
    path = Path(output_dir) / 'company' / LEDGER
    starts = [r['window_start'] for r in read_lines(path) if valid_record(r)
              and r['code']['sha256'] == row['code']['sha256']]
    row = seal({**row, 'snapshot_at': completed.isoformat(),
                'window_start': min(starts) if starts else completed.isoformat()})
    if not valid_record(row):
        raise ValueError('invalid v2 capture')
    append(path, row)
    return True


def valid_record(row):
    try:
        if not isinstance(row, dict) or row['record_sha256'] != seal(row)['record_sha256']:
            return False
        if row['version'] != VERSION or row['producer'] != PRODUCER or row['rules'] != RULES:
            return False
        if row['input_sha256'] != digest(row['evidence']) or row['code']['sha256'] != digest(row['code']['files']):
            return False
        if set(row['code']['files']) != set(SOURCES) or not row['models']:
            return False
        if any(len(str(v)) != 64 or any(c not in '0123456789abcdef' for c in str(v))
               for v in [*row['code']['files'].values(), *row['models'].values()]):
            return False
        if set(row['evidence']['inputs']) != set(DEPARTMENTS) | {'risk_department'}:
            return False
        cars = {int(r['car_no']) for r in row['evidence']['inputs']['risk_department']}
        if len(cars) < 3 or not cars <= set(range(1, 10)):
            return False
        for riders in row['evidence']['inputs'].values():
            if len(riders) != len(cars) or {int(r['car_no']) for r in riders} != cars:
                return False
        stamp = datetime.fromisoformat(row['snapshot_at'])
        window = datetime.fromisoformat(row['window_start'])
        if window.tzinfo is None or window > stamp:
            return False
        if stamp.tzinfo is None or not 300 < row['close_at'] - stamp.timestamp() <= 2400:
            return False
        if row['evidence']['knowledge_cutoff'] > stamp.date().isoformat():
            return False
        prices = row['evidence']['usable_quotes']
        for buy, price in prices.items():
            parts = [int(x) for x in buy.split('-')]
            qt = datetime.fromisoformat(row['evidence']['quote_times'][buy])
            if (len(parts) != 3 or len(set(parts)) != 3 or not set(parts) <= cars or
                    not math.isfinite(price) or not 1 <= price < 9999.9 or qt.tzinfo is None or
                    not 0 <= stamp.timestamp() - qt.timestamp() <= 300):
                return False
        expected = {d + ':' + v for d in DEPARTMENTS for v in VARIANTS}
        if set(row['arms']) != expected or set(row['views']) != {'fixed_six', 'risk_matched'}:
            return False
        for view, groups in row['views'].items():
            if set(groups) != set(GROUPS):
                return False
            for group, arms in groups.items():
                if set(arms) != expected | ({'risk'} if view == 'risk_matched' else set()):
                    return False
                risk = [t for t in row['risk_tickets'] if t['group'] == group]
                count = 6 if view == 'fixed_six' else len(risk)
                for arm, tickets in arms.items():
                    if tickets is None:
                        continue
                    if not 0 < len(tickets) == count <= 12 or len({t['buy'] for t in tickets}) != count:
                        return False
                    for t in tickets:
                        if (t['stake_yen'] != 100 or t['group'] != group or t['buy'] not in prices
                                or t['odds'] != prices[t['buy']] or (t['odds'] >= 100) != (group == '穴')
                                or not math.isfinite(t['prob']) or not 0 <= t['prob'] <= 1):
                            return False
                    if arm == 'risk':
                        if tickets != risk:
                            return False
                    elif tickets != row['arms'][arm]['pools'][group][:count]:
                        return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError):
        return False


def latest_records(folder):
    latest, invalid = {}, 0
    for row in read_lines(folder / LEDGER):
        if not valid_record(row):
            invalid += 1
            continue
        key = (row['code']['sha256'], row['race_id'])
        if key not in latest or row['snapshot_at'] > latest[key]['snapshot_at']:
            latest[key] = row
    return list(latest.values()), invalid


def outcomes_for(folder, race_ids):
    outcomes, conflicts = {}, []
    for item in read_lines(folder / OUTCOMES):
        if not item or item.get('record_sha256') != seal(item)['record_sha256']:
            raise ValueError('corrupt official outcome evidence')
        rid = item['race_id']
        if rid not in outcomes or item['observed_at'] > outcomes[rid]['observed_at']:
            outcomes[rid] = item
    path = folder.parent / 'latest_results.json'
    for result in json.loads(path.read_text(encoding='utf-8')) if path.exists() else []:
        rid = str(result.get('race_id'))
        outcome = normalize_outcome(result)
        if rid not in race_ids or outcome is None:
            continue
        previous = outcomes.get(rid)
        observed = outcome['source_observed_at']
        if previous:
            old = previous['outcome']
            if outcome == old:
                continue
            # Re-reading a stale result is not a new official observation.
            if not observed or observed <= old.get('source_observed_at', ''):
                if outcome['winning_buys'] != old['winning_buys'] or outcome['payouts'] != old['payouts']:
                    conflicts.append(rid)
                continue
            if old['payout_complete'] and not outcome['payout_complete'] and old['winning_buys'] == outcome['winning_buys']:
                continue
        item = {'race_id': rid, 'outcome': outcome, 'observed_at': observed or datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                'source_timestamp_verified': bool(observed)}
        append(folder / OUTCOMES, item)
        outcomes[rid] = item
    return {rid: item['outcome'] for rid, item in outcomes.items()}, conflicts


def metrics(samples, view, group, arm):
    hits = stake = returned = 0
    complete = 0
    cash = []
    for row in samples:
        bets = row['views'][view][group][arm]
        hit, value = ticket_return(bets, row['outcome'])
        hits += hit
        if value is not None:
            complete += 1
            stake += 100 * len(bets)
            returned += value
            cash.append(value)
    largest = max(cash, default=0)
    return {'races': len(samples), 'hits': hits, 'hit_rate': hits / len(samples) if samples else None,
            'payout_complete_races': complete, 'stake_yen': stake, 'return_yen': returned,
            'return_rate': returned / stake if stake else None,
            'largest_hit_return_share': largest / returned if returned else None,
            'return_rate_without_largest_hit': (returned - largest) / stake if stake else None}


def evaluation(samples, view, group, left, right, end, now, primary):
    result = {'status': 'exploratory' if not primary else 'collecting', 'end_at': end.isoformat(),
              'minimum_paired_races': 500, 'minimum_racing_days': 28, 'automatic_promotion': False}
    if not primary or now < end:
        return result
    days = defaultdict(list)
    for row in samples:
        if datetime.fromisoformat(row['snapshot_at']) >= end:
            continue
        lh, lv = ticket_return(row['views'][view][group][left], row['outcome'])
        rh, rv = ticket_return(row['views'][view][group][right], row['outcome'])
        days[row['date']].append((int(rh) - int(lh), None if lv is None or rv is None else (rv - lv) / 600))
    if sum(map(len, days.values())) < 500 or len(days) < 28:
        return {**result, 'status': 'insufficient_at_fixed_deadline'}
    # Day-cluster bootstrap; one predeclared end, no repeated significance looks.
    blocks = [(len(rows), sum(x for x, _ in rows), sum(y is not None for _, y in rows),
               sum(y for _, y in rows if y is not None)) for rows in days.values()]
    rng = random.Random(20261008)
    hit_deltas, roi_deltas = [], []
    for _ in range(RULES['resamples']):
        drawn = rng.choices(blocks, k=len(blocks))
        hit_deltas.append(sum(b[1] for b in drawn) / sum(b[0] for b in drawn))
        n = sum(b[2] for b in drawn)
        if n:
            roi_deltas.append(sum(b[3] for b in drawn) / n)
    alpha = RULES['family_alpha'] / (2 * RULES['primary_tests'])
    def interval(values):
        values.sort()
        return [values[int(len(values) * alpha / 2)], values[min(len(values)-1, int(len(values)*(1-alpha/2)))]] if values else None
    hit_ci, roi_ci = interval(hit_deltas), interval(roi_deltas)
    payout_races = sum(b[2] for b in blocks)
    payout_days = sum(b[2] > 0 for b in blocks)
    if payout_races < 500 or payout_days < 28:
        roi_ci = None
    return {**result, 'status': 'fixed_deadline_evaluated', 'method': 'day_cluster_percentile_bootstrap',
            'family_correction': '24 primary endpoint tests (12 hit rate, 12 ROI)',
            'payout_complete_races': payout_races, 'payout_complete_days': payout_days,
            'hit_rate_difference_interval': hit_ci, 'return_rate_difference_interval': roi_ci,
            'hit_improvement_supported': hit_ci[0] > 0, 'roi_improvement_supported': bool(roi_ci and roi_ci[0] > 0)}


def comparisons(rows, outcomes, now):
    report = {}
    if not rows:
        return report
    start = min(datetime.fromisoformat(r['window_start']) for r in rows)
    end = start + timedelta(days=RULES['evaluation_days'])
    for dept in DEPARTMENTS:
        for group in GROUPS:
            pairs = [('fixed_six', 'baseline', v) for v in VARIANTS if v != 'baseline']
            pairs += [('fixed_six', 'preserve', 'combined'), ('fixed_six', 'combined', 'context'),
                      ('fixed_six', 'market', 'combined')]
            pairs += [('risk_matched', 'risk', v) for v in VARIANTS]
            for view, left, right in pairs:
                lkey = 'risk' if left == 'risk' else dept + ':' + left
                rkey = dept + ':' + right
                exclusions = Counter()
                matched = []
                for row in rows:
                    if datetime.fromisoformat(row['snapshot_at']) >= end:
                        exclusions['after_fixed_evaluation_window'] += 1
                        continue
                    arms = row['views'][view][group]
                    if arms.get(lkey) is None or arms.get(rkey) is None:
                        exclusions['missing_or_insufficient_preclose_candidates'] += 1
                    else:
                        matched.append(row)
                samples = [{**r, 'outcome': outcomes[r['race_id']]} for r in matched if r['race_id'] in outcomes]
                primary = view == 'fixed_six' and left == 'baseline' and right in ('preserve', 'conditional')
                key = f'{dept}:{group}:{view}:{left}:{right}'
                report[key] = {'department': dept, 'group': group, 'view': view, 'left': left, 'right': right,
                    'primary': primary, 'preclose_matched': len(matched), 'settled_pairs': len(samples),
                    'race_ids': [r['race_id'] for r in samples], 'exclusions': dict(exclusions),
                    'arms': {left: metrics(samples, view, group, lkey), right: metrics(samples, view, group, rkey)},
                    'evaluation': evaluation(samples, view, group, lkey, rkey, end, now, primary)}
    return report


def build_report(output_dir=ROOT / 'outputs', now=None):
    now = now or datetime.now(ZoneInfo('Asia/Tokyo'))
    folder = Path(output_dir) / 'company'
    folder.mkdir(parents=True, exist_ok=True)
    rows, invalid = latest_records(folder)
    outcomes, conflicts = outcomes_for(folder, {r['race_id'] for r in rows})
    cohorts = defaultdict(list)
    for row in rows:
        cohorts[row['code']['sha256']].append(row)
    attempts, bad_attempts = {}, 0
    for row in read_lines(folder / ATTEMPTS):
        if not row or row.get('record_sha256') != seal(row)['record_sha256']:
            bad_attempts += 1
            continue
        key = row['race_id']
        if key not in attempts or row['at'] > attempts[key]['at']:
            attempts[key] = row
    groups = {}
    for code, cohort in cohorts.items():
        pairs = comparisons(cohort, outcomes, now)
        risk = {}
        for group in GROUPS:
            settled = [{**r, 'outcome': outcomes[r['race_id']]} for r in cohort if r['race_id'] in outcomes
                       and r['views']['risk_matched'][group]['risk']]
            risk[group] = metrics(settled, 'risk_matched', group, 'risk')
        total_return = sum(m['return_yen'] for m in risk.values())
        groups[code] = {'frozen_races': len(cohort), 'settled_races': sum(r['race_id'] in outcomes for r in cohort),
                       'comparisons': pairs, 'risk_dependency': risk,
                       'risk_hole_return_share': risk['穴']['return_yen'] / total_return if total_return else None,
                       'probability_calibration': calibration(cohort, outcomes),
                       'quote_coverage': dict(Counter('complete' if r['quote_quality']['complete_single_snapshot'] else 'partial' for r in cohort)),
                       'field_sizes': dict(Counter(str(r['strata']['field_size']) for r in cohort)),
                       'model_versions': dict(Counter(digest(r['models']) for r in cohort))}
    report = {'version': VERSION, 'updated_at': now.isoformat(), 'rules': RULES,
              'frozen_races': len({r['race_id'] for r in rows}), 'invalid_records': invalid,
              'invalid_attempts': bad_attempts, 'outcome_conflicts': sorted(set(conflicts)),
              'capture_attempts': len(attempts), 'capture_reasons': dict(Counter(r['reason'] for r in attempts.values())),
              'attempts_by_field_size': {str(size): dict(Counter(r['reason'] for r in attempts.values() if r['field_size'] == size))
                                         for size in sorted({r['field_size'] for r in attempts.values()})},
              'cohorts': groups, 'automatic_promotion': False, 'purchase_authorized': False,
              'status': 'collecting_prospective_evidence',
              'limitations': ['Probabilities and thresholds remain uncalibrated hypotheses.',
                             'Each pair has its own cohort; do not rank departments across different race sets.',
                             'Unavailable markets/context are excluded explicitly; no invented odds or padding.',
                             '56-day fixed window; insufficient evidence does not extend the test until it wins.']}
    (folder / 'annual_position_v2_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    render(report, folder)
    return report


def calibration(rows, outcomes):
    """Proper scores on the complete distribution, never only selected tickets."""
    by = defaultdict(list)
    bins = defaultdict(lambda: defaultdict(lambda: [0, 0.0, 0]))
    for row in rows:
        outcome = outcomes.get(row['race_id'])
        if not outcome or len(outcome['winning_buys']) != 1:
            continue  # tie settlement remains valid; ordinary probability scoring is separate
        winner = outcome['winning_buys'][0]
        for arm, portfolio in row['arms'].items():
            probs = portfolio.get('distribution')
            if not probs:
                continue
            by[arm].append((-math.log(max(probs.get(winner, 0), 1e-15)),
                            sum((p - int(buy == winner)) ** 2 for buy, p in probs.items())))
            for buy, p in probs.items():
                key = 'under_0.1%' if p < .001 else '0.1_to_1%' if p < .01 else '1_to_5%' if p < .05 else '5%_plus'
                value = bins[arm][key]
                value[0] += 1
                value[1] += p
                value[2] += buy == winner
    return {arm: {'races': len(values), 'mean_log_loss': sum(v[0] for v in values) / len(values),
                   'mean_brier': sum(v[1] for v in values) / len(values),
                   'bins': {key: {'tickets': n, 'mean_prediction': p / n, 'observed_hit_rate': h / n}
                            for key, (n, p, h) in bins[arm].items()},
                   'calibration_passed': False, 'note': 'Descriptive prospective audit, not automatic certification'}
            for arm, values in by.items()}


def render(report, folder):
    def rate(v):
        return '未集計' if v is None else f'{v*100:.1f}%'
    parts = []
    statuses = {'collecting': '事前に定めた期間で検証中', 'exploratory': '参考比較・採用判定の対象外',
                'insufficient_at_fixed_deadline': '期間終了・件数不足のため改善未確認',
                'fixed_deadline_evaluated': '固定期間の評価完了（的中率と回収率は別判定）'}
    def interval(values):
        return '判定に必要な記録が不足' if values is None else f'{values[0]*100:+.2f}〜{values[1]*100:+.2f}ポイント'
    names = dict(zip(DEPARTMENTS, ('データ部', '展開部', 'ライン部')))
    for code, cohort in report['cohorts'].items():
        parts.append(f'<h2>実装版 {code[:12]} ／ 事前保存 {cohort["frozen_races"]}・結果照合 {cohort["settled_races"]}</h2>')
        for pair in cohort['comparisons'].values():
            title = f'{names[pair["department"]]}・{pair["group"]}：{LABELS[pair["left"]]} → {LABELS[pair["right"]]}'
            mode = '各6点の比較' if pair['view'] == 'fixed_six' else 'リスク部の現行点数と比較'
            rows = ''.join(f'<tr><td>{LABELS[key]}</td><td>{m["hits"]}/{m["races"]}</td><td>{rate(m["hit_rate"])}</td>'
                           f'<td>{m["stake_yen"]:,}円</td><td>{rate(m["return_rate"])}</td></tr>' for key, m in pair['arms'].items())
            parts.append(f'<details><summary>{title} ／ 事前比較成立 {pair["preclose_matched"]}レース</summary>'
                         f'<p>{mode}・1点100円。候補不足は補充しません。保存{cohort["frozen_races"]}レース中、比較成立{pair["preclose_matched"]}レース。</p>'
                         '<table><tr><th>方式</th><th>的中/照合</th><th>的中率</th><th>払戻確認済み投資額</th><th>回収率</th></tr>'
                         + rows + f'</table><p>判定：{statuses[pair["evaluation"]["status"]]} ／ 固定終了日 {pair["evaluation"]["end_at"][:10]}</p>'
                         + (f'<p>的中率差の補正済み区間：{interval(pair["evaluation"].get("hit_rate_difference_interval"))}。'
                            f'回収率差：{interval(pair["evaluation"].get("return_rate_difference_interval"))}。</p>'
                            if pair['evaluation']['status'] == 'fixed_deadline_evaluated' else '') + '</details>')
        parts.append(f'<p>リスク部の払戻全体に占める穴の割合：{rate(cohort["risk_hole_return_share"])}</p>')
        for group, risk in cohort['risk_dependency'].items():
            parts.append(f'<p>リスク部・{group}参考集計：最大1的中の払戻寄与 {rate(risk["largest_hit_return_share"])}、'
                         f'その払戻を除いた回収率 {rate(risk["return_rate_without_largest_hit"])}</p>')
    page = '<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' +\
        '<title>部署別比較 v2</title><style>body{font-family:system-ui;background:#f4f7fb;color:#172b45;padding:20px}main{max-width:1100px;margin:auto}' +\
        'p{line-height:1.8}details{background:white;padding:16px;margin:12px 0;border-radius:10px;overflow:auto}summary{cursor:pointer}td,th{padding:10px;border-bottom:1px solid #ddd}</style><main>' +\
        '<p><a href="annual_position_experiment_report.html">旧v1の記録</a> ／ <a href="annual_department_report.html">部署別予想</a></p>' +\
        '<h1>部署別比較 v2</h1><p>リスク部の現行買い目を保存。①評価保持、②条件付き評価、専門展開、市場支持のみを区別します。' +\
        '同じレース・同じ時点・同じ点数と金額の二者比較です。各最大12点、穴は取得時100倍以上。過去の予想は追加しません。</p>' +\
        '<p>尺度だけで候補が変わらない共通選別を使用。条件付き評価は候補選別にも反映。市場情報の再混合はしません。' +\
        '予想生成は通常予測の経路に統一し、実装・モデル・入力の版を保存します。</p>' +\
        f'<p>事前保存 {report["frozen_races"]}レース ／ 記録不整合 {report["invalid_records"]}件。改善効果は未確認・自動採用なし。</p>' +\
        '<p>①②の主要比較は各6点、開始から56日で区切り、各500レース・28開催日以上を必要とします。的中率と回収率は別に判定し、' +\
        '日単位の再標本化と多重比較補正を使います。不足時は改善未確認です。</p>' +\
        f'<p>取得・見送り理由：{html.escape(str(report["capture_reasons"]))}。オッズ欠損でも①の記録は保存し、②と市場比較だけを成立不可にします。</p>' +\
        ''.join(parts) + '<p>部署間・比較間で対象レースが違う成績を単純に順位付けしません。確率・EV・展開仮説は未校正です。</p>' +\
        '<p><a href="annual_position_v2_report.json">詳細・除外理由・実装版</a></p></main></html>'
    (folder / 'annual_position_v2_report.html').write_text(page, encoding='utf-8')
