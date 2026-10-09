"""Prospective equation laboratory, isolated from the CEO and purchase paths.

Only canonical, already frozen v2 inputs may produce a new pre-close record.
Reports never replay a closed race. Every model is fitted on prior JST days.
"""
import argparse
import hashlib
import html
import json
import math
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from department_experiment_v2 import (LEDGER as INPUTS, OUTCOMES, append, digest,
                                     read_lines, seal, valid_record, outcomes_for)
from equation_models import METHODS, LABELS, CONFIG, fit, predict
from equation_preview import provisional_distribution, standalone_tickets
from official_outcomes import ticket_return

ROOT = Path(__file__).resolve().parents[1]
JST = ZoneInfo('Asia/Tokyo')
VERSION = 'equation_lab_v1'
LEDGER = 'annual_equation_ledger.jsonl'
MODELS = 'annual_equation_models.jsonl'
ATTEMPTS = 'annual_equation_attempts.jsonl'
PATHS = 'annual_equation_observed_paths.jsonl'
GROUPS = ('本線', '穴')
RULES = {'stake_yen': 100, 'maximum_tickets': 12, 'minimum_hole_odds': 100,
         'main_min_ev': 1.10, 'hole_min_ev': 1.25, 'hole_min_probability': .001,
         'snapshot_age_seconds': 300, 'capture_minutes': [5, 40],
         'evaluation_days': 56, 'minimum_pairs': 500, 'minimum_days': 28,
         'family_alpha': .05, 'primary_endpoints': 12, 'bootstrap_resamples': 12000,
         'automatic_promotion': False, 'ceo_integration': False}


def clock():
    return datetime.now(JST)


def code_id():
    names = ('equation_lab.py', 'equation_models.py', 'equation_preview.py',
             'department_experiment_v2.py', 'official_outcomes.py')
    return digest({'rules': RULES, 'training': CONFIG, 'sources': {
        n: hashlib.sha256(Path(__file__).with_name(n).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for n in names}})


def checked(path):
    for row in read_lines(path):
        if not isinstance(row, dict) or row.get('record_sha256') != seal(row)['record_sha256']:
            raise ValueError(f'corrupt equation evidence: {path.name}')
        yield row


def source_ready(row):
    if not valid_record(row) or not row['quote_quality']['complete_single_snapshot']:
        return False
    prices = row['evidence']['usable_quotes']
    n = len(row['evidence']['inputs']['risk_department'])
    if len(prices) != n * (n - 1) * (n - 2):
        return False
    for riders in row['evidence']['inputs'].values():
        if any(r.get('finish_pos') is not None or r.get('official_finish_pos') is not None
               or r.get('result_available') not in (0, None, False) for r in riders):
            return False
    return True


def prior_samples(folder, cutoff):
    """As-of labels, one complete source per race; never use target-day labels."""
    cutoff = datetime.combine(cutoff, datetime.min.time(), JST)
    lower = cutoff - timedelta(days=CONFIG['training_lookback_days'])
    latest, bad = {}, 0
    for row in read_lines(folder / INPUTS):
        if not row or not valid_record(row):
            bad += 1
            continue
        stamp = datetime.fromisoformat(row['snapshot_at'])
        if not lower <= stamp < cutoff or row['close_at'] >= cutoff.timestamp() or not source_ready(row):
            continue
        rid = row['race_id']
        if rid not in latest or stamp > datetime.fromisoformat(latest[rid]['snapshot_at']):
            latest[rid] = row
    results, conflicts = {}, set()
    for item in checked(folder / OUTCOMES):
        stamp = datetime.fromisoformat(item['observed_at'])
        rid = item['race_id']
        if stamp.tzinfo is None or stamp >= cutoff or rid not in latest or stamp.timestamp() < latest[rid]['close_at']:
            continue
        outcome = item['outcome']
        key = digest({k: outcome[k] for k in ('winning_buys', 'payouts', 'payout_complete')})
        if rid in results and results[rid][0] != key:
            conflicts.add(rid)
        results[rid] = (key, outcome, item['record_sha256'])
    samples = []
    for rid, source in latest.items():
        if rid not in results or rid in conflicts:
            continue
        _, out, out_hash = results[rid]
        # Ties are settled in evaluation, but not reduced to an invented order in training.
        if len(out['winning_buys']) != 1:
            continue
        actual = out['winning_buys'][0]
        if actual not in source['evidence']['usable_quotes']:
            continue
        samples.append({'race_id': rid, 'date': datetime.fromisoformat(source['snapshot_at']).astimezone(JST).date().isoformat(),
            'riders': source['evidence']['inputs']['risk_department'],
            'quotes': source['evidence']['usable_quotes'], 'actual': actual,
            'input_record': source['record_sha256'], 'outcome_record': out_hash,
            'close_at': source['close_at']})
    samples = sorted(samples, key=lambda r: (r['date'], r['race_id']))[-CONFIG['maximum_training_races']:]
    paths, by_id, duplicate_paths = [], {}, set()
    for item in read_lines(folder / PATHS):
        if not item:
            continue
        rid = item.get('race_id')
        if rid in by_id and digest(by_id[rid]) != digest(item):
            duplicate_paths.add(rid)
        by_id[rid] = item
    sample_map = {s['race_id']: s for s in samples}
    for rid, item in by_id.items():
        try:
            stamp = datetime.fromisoformat(item['observed_at'])
            if (rid not in sample_map or rid in duplicate_paths or item.get('verified') is not True
                    or item.get('schema') != 'observed_top3_stages_v1'
                    or not str(item.get('source_url', '')).startswith('https://')
                    or stamp.tzinfo is None or not sample_map[rid]['close_at'] < stamp.timestamp() < cutoff.timestamp()):
                continue
            cars = {int(r['car_no']) for r in sample_map[rid]['riders']}
            orders = item.get('orders', {})
            stages = [orders.get(k) for k in ('start','bell','back','finish')]
            if any(not isinstance(s,list) or len(s)!=3 or len(set(s))!=3 or not set(s)<=cars for s in stages):
                continue
            if '-'.join(map(str,stages[-1])) != sample_map[rid]['actual']:
                continue
            paths.append(item)
        except (KeyError, ValueError, TypeError):
            continue
    return samples, paths, {'invalid_inputs': bad, 'conflicting_results': sorted(conflicts),
                            'conflicting_paths': sorted(duplicate_paths)}


def prepare_models(output_dir, now=None):
    now = now or clock()
    started = time.monotonic()
    folder = Path(output_dir) / 'company'
    day, code = now.astimezone(JST).date().isoformat(), code_id()
    existing = [m for m in checked(folder / MODELS) if m['cutoff_exclusive'] == day and m['code'] == code]
    if existing:
        # The first frozen fit of the day is reused; late arrivals cannot tune it.
        return min(existing, key=lambda m: (m['created_at'], m['record_sha256']))
    samples, paths, diagnostics = prior_samples(folder, now.astimezone(JST).date())
    fitted = fit(samples, paths)
    model = seal({'version': VERSION, 'code': code, 'cutoff_exclusive': day,
        'created_at': (now + timedelta(seconds=time.monotonic()-started)).isoformat(), 'training': fitted,
        'training_manifest': [{k: s[k] for k in ('race_id', 'date', 'input_record', 'outcome_record')} for s in samples],
        'observed_path_hashes': [digest(p) for p in paths], 'diagnostics': diagnostics,
        'purchase_authorized': False, 'ceo_integration': False})
    append(folder / MODELS, model)
    return model


def choose(distribution, prices, group, count):
    if distribution is None or not 0 < count <= 12:
        return None
    if set(distribution) != set(prices) or abs(sum(distribution.values()) - 1) > 1e-8:
        raise ValueError('invalid full distribution')
    tickets = []
    for buy, p in distribution.items():
        if not math.isfinite(p) or not 0 < p <= 1:
            raise ValueError('invalid probability')
        price = prices[buy]
        if ('穴' if price >= 100 else '本線') != group:
            continue
        ev = p * price
        if ev < (RULES['hole_min_ev'] if group == '穴' else RULES['main_min_ev']):
            continue
        if group == '穴' and p < RULES['hole_min_probability']:
            continue
        tickets.append({'buy': buy, 'prob': p, 'odds': price, 'ev': ev,
                        'group': group, 'stake_yen': 100})
    tickets.sort(key=lambda t: (-t['prob'], t['buy']))
    return tickets[:count] if len(tickets) >= count else None


def independent_views(distributions, riders, prices):
    """One independently ranked ticket list per equation, never pooled or voted."""
    results = {}
    for method in METHODS:
        trained = distributions[method] is not None
        probabilities = (distributions[method] if trained else
                         provisional_distribution(method, riders, prices))
        results[method] = {
            'basis': 'trained_shadow' if trained else 'untrained_preclose_proxy',
            'picks': standalone_tickets(probabilities, prices),
        }
    return results


def capture(input_hash, source_time, output_dir=ROOT / 'outputs', now=None):
    """Called only after v2 append succeeds; includes elapsed fit/inference time."""
    now = now or clock()
    started = time.monotonic()
    folder = Path(output_dir) / 'company'
    sources = [r for r in read_lines(folder / INPUTS) if r and r.get('input_sha256') == input_hash
               and r.get('snapshot_at') == source_time]
    if len(sources) != 1 or not source_ready(sources[0]):
        return 'complete_canonical_input_required'
    source = sources[0]
    if now.tzinfo is None or not 300 < source['close_at'] - now.timestamp() <= 2400:
        return 'outside_capture_window'
    models = prepare_models(output_dir, now)
    distributions, reasons, views = {}, {}, {}
    prices = source['evidence']['usable_quotes']
    for method in METHODS:
        model = models['training']['models'][method]
        distributions[method] = predict(method, model, source['evidence']['inputs']['risk_department'], prices)
        reasons[method] = models['training']['reasons'][method]
        if model and distributions[method] is None:
            reasons[method] = 'no_observed_paths_for_field_size'
    standalone = independent_views(distributions, source['evidence']['inputs']['risk_department'], prices)
    for group in GROUPS:
        risk = [t for t in source['risk_tickets'] if t['group'] == group]
        valid_risk = (0 < len(risk) <= 12 and all(t['odds'] == prices.get(t['buy']) and t['stake_yen'] == 100 for t in risk))
        views[group] = {'risk': risk if valid_risk else None}
        for method in METHODS:
            views[group][method] = choose(distributions[method], prices, group, len(risk)) if valid_risk else None
    completed = now + timedelta(seconds=time.monotonic() - started)
    if (completed.astimezone(JST).date().isoformat() != models['cutoff_exclusive']
            or not 300 < source['close_at'] - completed.timestamp() <= 2400) or any(
        not 0 <= completed.timestamp() - datetime.fromisoformat(t).timestamp() <= 300
        for t in source['evidence']['quote_times'].values()):
        return 'expired_during_equation_calculation'
    row = seal({'version': VERSION, 'race_id': source['race_id'], 'venue': source['venue'],
        'race_no': source['race_no'], 'date': completed.astimezone(JST).date().isoformat(),
        'snapshot_at': completed.isoformat(), 'close_at': source['close_at'],
        'source_record': source['record_sha256'], 'source_snapshot_at': source['snapshot_at'],
        'code': models['code'], 'cohort': digest([models['code'], source['code']['sha256']]),
        'model_record': models['record_sha256'], 'rules': RULES,
        'distributions': distributions, 'views': views, 'reasons': reasons,
        'standalone': standalone,
        'purchase_authorized': False, 'ceo_integration': False})
    # Keep waiting states as evidence of availability, never as zero-hit forecasts.
    append(folder / LEDGER, row)
    return 'captured_shadow' if any(distributions.values()) else 'collecting_training_data'


def record_attempt(race_id, reason, output_dir, now=None):
    append(Path(output_dir) / 'company' / ATTEMPTS,
           {'race_id': str(race_id), 'at': (now or clock()).isoformat(), 'reason': reason, 'code': code_id()})


def validated_rows(folder):
    models = {m['record_sha256']: m for m in checked(folder / MODELS)}
    sources = {r['record_sha256']: r for r in read_lines(folder / INPUTS) if r and valid_record(r)}
    result = []
    for row in checked(folder / LEDGER):
        source, model = sources.get(row['source_record']), models.get(row['model_record'])
        if source is None or model is None or not source_ready(source):
            raise ValueError('missing frozen equation provenance')
        stamp, created = datetime.fromisoformat(row['snapshot_at']), datetime.fromisoformat(model['created_at'])
        if (row['version'] != VERSION or row['rules'] != RULES or row['purchase_authorized'] is not False
                or row['ceo_integration'] is not False or row['race_id'] != source['race_id']
                or row['close_at'] != source['close_at'] or row['code'] != model['code']
                or row['source_snapshot_at'] != source['snapshot_at']
                or row['date'] != stamp.astimezone(JST).date().isoformat()
                or row['cohort'] != digest([row['code'], source['code']['sha256']])
                or stamp.tzinfo is None or created.tzinfo is None or created > stamp
                or datetime.fromisoformat(source['snapshot_at']) > stamp
                or not 300 < row['close_at'] - stamp.timestamp() <= 2400
                or model['cutoff_exclusive'] != stamp.astimezone(JST).date().isoformat()
                or any(s['date'] >= model['cutoff_exclusive'] for s in model['training_manifest'])
                or any(not 0 <= stamp.timestamp() - datetime.fromisoformat(t).timestamp() <= 300
                       for t in source['evidence']['quote_times'].values())):
            raise ValueError('invalid equation provenance or time')
        for method in METHODS:
            expected_distribution = predict(method, model['training']['models'][method],
                source['evidence']['inputs']['risk_department'], source['evidence']['usable_quotes'])
            actual = row['distributions'][method]
            if expected_distribution is None:
                if actual is not None:
                    raise ValueError('untrained equation invented a distribution')
            elif actual is None or set(actual) != set(expected_distribution) or any(
                abs(actual[k]-p) > 1e-12 for k,p in expected_distribution.items()):
                raise ValueError('distribution does not match frozen equation model')
        prices = source['evidence']['usable_quotes']
        if 'standalone' in row and row['standalone'] != independent_views(
                row['distributions'], source['evidence']['inputs']['risk_department'], prices):
            raise ValueError('independent equation picks disagree with frozen input')
        for group in GROUPS:
            risk = [t for t in source['risk_tickets'] if t['group'] == group]
            expected = risk if 0 < len(risk) <= 12 and all(t['odds'] == prices.get(t['buy']) and t['stake_yen'] == 100 for t in risk) else None
            if row['views'][group]['risk'] != expected:
                raise ValueError('risk reference was changed')
            for method in METHODS:
                selected = choose(row['distributions'][method], prices, group, len(risk)) if expected else None
                if selected != row['views'][group][method]:
                    raise ValueError('equation selection or budget mismatch')
        result.append(row)
    return result, models


def performance(samples, method, group):
    cash, hits, stake = [], 0, 0
    for row in samples:
        tickets = row['views'][group][method]
        hit, returned = ticket_return(tickets, row['outcome'])
        hits += hit; stake += len(tickets) * 100; cash.append(returned)
    total = sum(cash)
    largest = max(cash, default=0)
    return {'races': len(samples), 'hits': hits, 'hit_rate': hits / len(samples) if samples else None,
            'stake_yen': stake, 'return_yen': total, 'return_rate': total / stake if stake else None,
            'largest_hit_share': largest / total if total else None,
            'return_rate_without_largest_hit': (total-largest) / stake if stake else None}


def evaluate(samples, method, group, start, now):
    end = start + timedelta(days=RULES['evaluation_days'])
    result = {'end_at': end.isoformat(), 'status': 'collecting', 'automatic_promotion': False,
              'ceo_integration': False, 'hit_improvement_supported': False}
    if now < end:
        return result
    days = defaultdict(list)
    for r in samples:
        a, av = ticket_return(r['views'][group][method], r['outcome'])
        b, bv = ticket_return(r['views'][group]['risk'], r['outcome'])
        cost = len(r['views'][group]['risk']) * 100
        days[r['date']].append((int(a)-int(b), av-bv, cost))
    if len(samples) < RULES['minimum_pairs'] or len(days) < RULES['minimum_days']:
        return {**result, 'status': 'insufficient_at_fixed_deadline'}
    blocks = [(len(v), sum(x[0] for x in v), sum(x[1] for x in v), sum(x[2] for x in v)) for v in days.values()]
    rng = random.Random(20261009)
    hit, roi = [], []
    for _ in range(RULES['bootstrap_resamples']):
        draw = rng.choices(blocks, k=len(blocks))
        hit.append(sum(b[1] for b in draw)/sum(b[0] for b in draw))
        roi.append(sum(b[2] for b in draw)/sum(b[3] for b in draw))
    alpha = RULES['family_alpha']/RULES['primary_endpoints']
    def ci(values):
        values.sort()
        return [values[int(len(values)*alpha/2)], values[min(len(values)-1,int(len(values)*(1-alpha/2)))]]
    hc, rc = ci(hit), ci(roi)
    return {**result, 'status': 'review_evidence_only', 'hit_difference_interval': hc,
            'return_difference_interval': rc, 'hit_improvement_supported': hc[0] > 0,
            'method': 'day_cluster_bootstrap_12_endpoint_family', 'review_required': True}


def build_report(output_dir=ROOT / 'outputs', now=None):
    now = now or clock()
    folder = Path(output_dir) / 'company'
    folder.mkdir(parents=True, exist_ok=True)
    rows, models = validated_rows(folder)
    outcomes, conflicts = outcomes_for(folder, {r['race_id'] for r in rows})
    if conflicts:
        raise ValueError('conflicting official results in equation lab')
    attempts = list(checked(folder / ATTEMPTS))
    cohorts = defaultdict(list)
    for row in rows:
        cohorts[row['cohort']].append(row)
    comparisons, head_to_head = [], []
    for cohort, values in cohorts.items():
        for method in METHODS:
            for group in GROUPS:
                eligible = [r for r in values if r['views'][group]['risk'] and r['views'][group][method]]
                if not eligible:
                    continue
                start = min(datetime.fromisoformat(r['snapshot_at']) for r in eligible)
                end = start + timedelta(days=RULES['evaluation_days'])
                latest = {}
                for row in eligible:
                    if datetime.fromisoformat(row['snapshot_at']) >= end:
                        continue
                    rid = row['race_id']
                    if rid not in latest or row['snapshot_at'] > latest[rid]['snapshot_at']:
                        latest[rid] = row
                samples = [{**r, 'outcome': outcomes[rid]} for rid, r in latest.items()
                           if rid in outcomes and outcomes[rid]['payout_complete'] and r['close_at'] < now.timestamp()]
                comparisons.append({'cohort': cohort, 'method': method, 'group': group,
                    'preclose_matched': len(latest), 'race_ids': [r['race_id'] for r in samples],
                    'challenger': performance(samples, method, group), 'risk': performance(samples, 'risk', group),
                    'rescued': sum(ticket_return(r['views'][group][method],r['outcome'])[0] and not ticket_return(r['views'][group]['risk'],r['outcome'])[0] for r in samples),
                    'lost': sum(ticket_return(r['views'][group]['risk'],r['outcome'])[0] and not ticket_return(r['views'][group][method],r['outcome'])[0] for r in samples),
                    'evaluation': evaluate(samples, method, group, start, now)})
        # Descriptive direct comparisons use the SAME input snapshot for both
        # equations, never rankings of their different eligibility cohorts.
        for i, left in enumerate(METHODS):
            for right in METHODS[i+1:]:
                for group in GROUPS:
                    matched = {}
                    for row in values:
                        if not row['views'][group][left] or not row['views'][group][right]:
                            continue
                        rid = row['race_id']
                        if rid not in matched or row['snapshot_at'] > matched[rid]['snapshot_at']:
                            matched[rid] = row
                    pairs = [{**r,'outcome':outcomes[rid]} for rid,r in matched.items()
                             if rid in outcomes and outcomes[rid]['payout_complete'] and r['close_at'] < now.timestamp()]
                    if pairs:
                        head_to_head.append({'cohort':cohort,'left':left,'right':right,'group':group,
                            'status':'descriptive_only_not_a_promotion_test',
                            'race_ids':[r['race_id'] for r in pairs],
                            'left_metrics':performance(pairs,left,group),'right_metrics':performance(pairs,right,group)})
    # Only immutable pre-close forecasts, never predictions regenerated after results.
    standalone_latest = {}
    today = now.astimezone(JST).date().isoformat()
    for row in rows:
        if row['date'] != today or not row.get('standalone'):
            continue
        rid = row['race_id']
        if rid not in standalone_latest or row['snapshot_at'] > standalone_latest[rid]['snapshot_at']:
            standalone_latest[rid] = {
                'race_id': rid, 'venue': row['venue'], 'race_no': row['race_no'],
                'snapshot_at': row['snapshot_at'], 'close_at': row['close_at'],
                'methods': row['standalone'],
            }
    standalone_by_race = sorted(standalone_latest.values(),
        key=lambda x: (x['close_at'], x['venue'], x['race_no']))
    latest_model = max(models.values(), key=lambda m: m['created_at']) if models else None
    samples, paths, diagnostics = prior_samples(folder, now.astimezone(JST).date())
    report = {'version': VERSION, 'updated_at': now.isoformat(), 'rules': RULES, 'training_rules': CONFIG,
        'automatic_promotion': False, 'ceo_integration': False, 'purchase_authorized': False,
        'methods': LABELS, 'prior_complete_races': len(samples), 'prior_days': len({s['date'] for s in samples}),
        'verified_path_annotations': len(paths), 'latest_model': latest_model,
        'recorded_races': len({r['race_id'] for r in rows}), 'snapshots': len(rows),
        'attempt_reasons': dict(Counter(r['reason'] for r in attempts)),
        'availability': {m: dict(Counter(r['reasons'][m] for r in rows)) for m in METHODS},
        'comparisons': comparisons, 'head_to_head': head_to_head, 'diagnostics': diagnostics,
        'standalone_by_race': standalone_by_race}
    (folder/'annual_equation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    def pct(v): return '—' if v is None else f'{v*100:.1f}%'
    table = []
    status_labels = {'collecting':'比較を収集中','insufficient_at_fixed_deadline':'期限内の証拠不足','review_evidence_only':'資料を確認・社長未採用'}
    for r in comparisons:
        a, b = r['challenger'], r['risk']
        table.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in [LABELS[r['method']],r['group'],a['races'],f"{a['hits']} / {b['hits']}",pct(a['hit_rate']),pct(b['hit_rate']),pct(a['return_rate']),pct(b['return_rate']),status_labels[r['evaluation']['status']]])+'</tr>')
    page = '<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>方程式の比較研究</title><style>body{font:16px/1.7 system-ui;background:#f4f6fa;color:#172235;margin:0}main{max-width:1100px;margin:auto;padding:28px}section{background:white;padding:20px;border-radius:12px;margin:18px 0}table{border-collapse:collapse;min-width:850px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}.scroll{overflow:auto}a{color:#2459ae}</style><main><h1>３つの方程式を、同じ条件で検証</h1><p><b>研究用です。社長の最終判断・購入判断には未組み込みです。</b>リスク部の現行買い目を比較基準として保持します。学習・的中率改善・本番採用は別の段階です。</p>'
    page += f'<section><h2>準備状況</h2><p>前日までの完全な保存入力と結果：{len(samples)}レース・{len({s["date"] for s in samples})}日。途中隊列の確認済み記録：{len(paths)}件。学習開始条件：50レース以上・7日以上。</p><p>新しい式の事前記録：{report["recorded_races"]}レース。データ不足の式には予想を捏造せず、待機理由を保存します。</p></section>'
    page += '<section><h2>比較する式</h2><p><b>① 市場の見落とし</b>：P(t) ∝ 市場支持(t) × exp(学習した選手・組み合わせ補正)。</p><p><b>② 先着関係</b>：各選手の先着確率から、上位３人の順番と残りの選手への優位性をまとめ、全３連単で正規化。</p><p><b>③ 途中隊列の遷移</b>：序盤→打鐘→最終バック→確定着順の、観測された上位３人の隊列変化を学習。精密な速度や残脚を再現するモデルではありません。途中隊列の記録が不足している間は未学習です。</p></section>'
    page += '<section id="standalone"><h2>方程式ごとの単独買い目（本線・穴）</h2><p>３つの式を独立に計算し、各式の本線・穴を別々に最大12点掲載。リスク部や他の式の買い目で並びを変更しません。<b>未学習</b>は事前特徴量から作る固定式の暫定参考値で、学習済み方程式の予測ではありません。隊列の暫定値は実際の打鐘・バック記録を用いた遷移予測ではありません。的中率・期待値の優位性は未検証。実購入・社長予想には未採用です。</p>'
    for race in standalone_by_race:
        header = f"{race['venue']} {race['race_no']}R｜保存 {race['snapshot_at'][11:16]}"
        page += '<details><summary>'+html.escape(header)+'</summary>'
        for method in METHODS:
            item = race['methods'][method]
            basis = '学習済み・実験用' if item['basis'] == 'trained_shadow' else '未学習・暫定参考'
            page += '<h3>'+html.escape(LABELS[method])+' ／ '+basis+'</h3>'
            for group in GROUPS:
                items = item['picks'][group]
                page += '<p><b>'+group+' '+str(len(items))+'点</b>：'
                page += ('、'.join(html.escape(t['buy'])+' ('+f"{t['odds']:.1f}"+'倍)' for t in items)
                         if items else '該当するオッズ帯の買い目なし')
                page += '</p>'
        page += '</details>'
    if not standalone_by_race:
        page += '<p>保存済みの発走前独立予想はまだありません。次回の適格な発走前スナップショットから掲載します。締切後の後付け生成はしません。</p>'
    page += '</section>'
    page += '<section><h2>同時点・同点数・同金額の成績</h2><p>本線・穴を別々に、リスク部と同じ点数・１点100円で比較。各最大12点、穴100倍以上。候補不足は比較から除外します。各行の対象レースが異なるため、行同士の単純な順位付けはしません。</p><div class="scroll"><table><tr><th>式</th><th>区分</th><th>比較R</th><th>的中：式／リスク</th><th>式の的中率</th><th>リスク的中率</th><th>式の回収率</th><th>リスク回収率</th><th>状態</th></tr>'+(''.join(table) or '<tr><td colspan="9">比較に必要な事前記録を収集中です。改善はまだ確認していません。</td></tr>')+'</table></div></section>'
    page += '<section><h2>社長への採用条件</h2><p>式と条件を固定した56日間で評価し、比較500レース・28日未満なら証拠不足です。複数の式を試した影響を補正し、的中率・回収率・最大払戻への依存を別々に確認します。期限を延ばして当たりを待つ運用や、自動採用はありません。十分な証拠が揃った後に、社長への組み込みを別の変更として審査します。</p></section><p><a href="annual_equation_report.json">詳細・不足理由・学習履歴</a> ／ <a href="annual_position_v2_report.html">２・３着改善の別実験</a> ／ <a href="operations.html">会社の運営状況</a></p></main></html>'
    (folder/'annual_equation_report.html').write_text(page,encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['report', 'train'])
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs')
    args = parser.parse_args()
    if args.command == 'train':
        model = prepare_models(args.output_dir)
        print(json.dumps(model['training']['reasons'], ensure_ascii=False))
    report = build_report(args.output_dir)
    print(json.dumps({k:report[k] for k in ('prior_complete_races','prior_days','recorded_races','verified_path_annotations')},ensure_ascii=False))
