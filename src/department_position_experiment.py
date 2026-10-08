"""Prospective factorial comparison, with the unmodified risk portfolio as control."""
import hashlib
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from betting_logic import score_riders, select_race
from common import OUTPUT_DIR
from market_axis_shadow import inspect_quotes

VERSION = 'department_positions_v1'
LEDGER = 'annual_position_experiment_ledger.jsonl'
DEPARTMENTS = ('data_department', 'pace_department', 'line_department')
VARIANTS = {'baseline': (False, False), 'preserve_only': (True, False),
            'conditional_only': (False, True), 'combined': (True, True)}
LABELS = {'risk': 'リスク部・現行', 'baseline': '各部署・現行',
          'preserve_only': '①2・3着評価を保持', 'conditional_only': '②順位別市場・条件付き評価',
          'combined': '①＋②'}
POLICY = 'latest_complete_bundle_40_to_5_minutes_before_close'


def load_json(path, default):
    # Corrupt prior evidence must not be silently replaced by an empty history.
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def portfolios(temp, market, preserve=False, conditional=False):
    riders = score_riders(temp, market, preserve_position_scores=preserve)
    frame, plan = select_race(riders, market, conditional_positions=conditional)
    tickets = [{'buy': str(t.buy), 'group': str(t.ticket_group), 'stake_yen': 100,
                'odds': float(t.odds_used), 'prob': float(t.prob), 'ev': float(t.ev)}
               for t in frame[frame.is_selected].itertuples()]
    return {'tickets': tickets, 'position_score_source': str(riders.position_score_source.iloc[0]),
            'probability_method': plan['probability_method'],
            'scores': [{'car_no': int(r.car_no), 'first': float(r.score_first),
                        'second': float(r.score_second), 'third': float(r.score_third)}
                       for r in riders.itertuples()]}


def make_bundle(race, department_inputs, market, now, cutoff):
    """No historical replay path. All hypotheses consume this one input bundle."""
    close = float(race.iloc[0]['close_at'])
    if now.tzinfo is None or not math.isfinite(close) or not 300 < close - now.timestamp() <= 2400:
        return None, 'outside_40_to_5_minute_window'
    _, reason, _ = inspect_quotes(race, market, now, close)
    if reason != 'eligible':
        return None, reason
    stamps = set(market.loc[market.bet_type.eq('trifecta'), 'odds_captured_at_jst'].astype(str))
    if len(stamps) != 1:
        return None, 'mixed_quote_snapshots'
    if set(department_inputs) != set(DEPARTMENTS) | {'risk_department'}:
        return None, 'incomplete_departments'
    arms = {'risk': portfolios(department_inputs['risk_department'], market)}
    for department in DEPARTMENTS:
        for variant, (preserve, conditional) in VARIANTS.items():
            arms[department + ':' + variant] = portfolios(department_inputs[department], market, preserve, conditional)
    quotes = [{'buy': str(r.buy), 'odds': float(r.odds_used)}
              for r in market[market.bet_type.eq('trifecta')].sort_values('buy').itertuples()]
    # Save the exact inputs, including department evaluations, for provenance.
    inputs = {d: json.loads(f.to_json(orient='records')) for d, f in department_inputs.items()}
    evidence = {'inputs': inputs, 'quotes': quotes, 'reference_cutoff_exclusive': cutoff}
    fingerprint = hashlib.sha256(json.dumps(evidence, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return {'version': VERSION, 'race_id': str(race.iloc[0]['race_id']),
            'venue': str(race.iloc[0].get('venue', '')), 'race_no': int(race.iloc[0].get('race_no', 0)),
            'snapshot_at': now.isoformat(), 'close_at': close,
            'quote_snapshot_at': next(iter(stamps)), 'snapshot_policy': POLICY,
            'input_sha256': fingerprint, 'evidence': evidence, 'arms': arms,
            'purchase_authorized': False}, 'eligible'


def append_bundle(bundle, completed_at, output_dir=OUTPUT_DIR):
    # The completion time, not the time computation began, governs eligibility.
    if bundle is None or not 300 < bundle['close_at'] - completed_at.timestamp() <= 2400:
        return False
    quote = datetime.fromisoformat(bundle['quote_snapshot_at'])
    if quote.tzinfo is None or not 0 <= completed_at.timestamp() - quote.timestamp() <= 300:
        return False
    bundle = {**bundle, 'snapshot_at': completed_at.isoformat()}
    folder = Path(output_dir) / 'company'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / LEDGER).open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(bundle, ensure_ascii=False, allow_nan=False) + '\n')
    return True


def valid_bundle(row):
    try:
        stamp = datetime.fromisoformat(row['snapshot_at'])
        quote = datetime.fromisoformat(row['quote_snapshot_at'])
        if (row['version'] != VERSION or row['snapshot_policy'] != POLICY
                or stamp.tzinfo is None or quote.tzinfo is None
                or not 300 < float(row['close_at']) - stamp.timestamp() <= 2400
                or not 0 <= stamp.timestamp() - quote.timestamp() <= 300):
            return False
        expected = {'risk'} | {d + ':' + v for d in DEPARTMENTS for v in VARIANTS}
        if set(row['arms']) != expected:
            return False
        for arm in row['arms'].values():
            seen = set()
            for t in arm['tickets']:
                cars = str(t['buy']).split('-')
                if (len(cars) != 3 or len(set(cars)) != 3 or
                        not all(c.isdigit() and 1 <= int(c) <= 9 for c in cars) or
                        t['buy'] in seen or t['group'] not in ('本線', '穴') or t['stake_yen'] != 100):
                    return False
                price = float(t['odds'])
                if not math.isfinite(price) or not 1 <= price < 9999.9 or (price >= 100) != (t['group'] == '穴'):
                    return False
                seen.add(t['buy'])
            if any(sum(t['group'] == g for t in arm['tickets']) > 12 for g in ('本線', '穴')):
                return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def latest_bundles(path):
    latest, invalid = {}, 0
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                invalid += 1
                continue
            if not valid_bundle(row):
                invalid += 1
                continue
            rid = str(row['race_id'])
            if rid not in latest or datetime.fromisoformat(row['snapshot_at']) > datetime.fromisoformat(latest[rid]['snapshot_at']):
                latest[rid] = row
    return latest, invalid


def performance(samples, arm, group):
    stake = returned = hits = missing_payout = 0
    returns = []
    for row in samples:
        tickets = [t for t in row['arms'][arm]['tickets'] if t['group'] == group]
        cost = sum(t['stake_yen'] for t in tickets)
        hit = any(t['buy'] == row['actual'] for t in tickets)
        hits += hit
        payout = row.get('payout_per_100yen')
        if hit and payout is None:
            missing_payout += 1
        value = float(payout) if hit and payout is not None else 0
        if payout is not None:  # identical payout-complete cohort for every arm
            stake += cost
            returned += value
            returns.append((value, cost))
    largest = max((v for v, _ in returns), default=0)
    return {'races': len(samples), 'hits': hits, 'hit_rate': hits / len(samples) if samples else None,
            'payout_complete_races': len(returns), 'stake_yen': stake, 'return_yen': returned,
            'return_rate': returned / stake if stake else None, 'missing_payout_hits': missing_payout,
            'largest_hit_return_share': largest / returned if returned else None,
            'return_rate_without_largest_hit': (returned - largest) / stake if stake else None}


def summarize(rows):
    comparisons = {}
    for department in DEPARTMENTS:
        keys = ['risk'] + [department + ':' + v for v in VARIANTS]
        for group in ('本線', '穴'):
            samples, exclusions = [], Counter()
            for row in rows:
                counts = [sum(t['group'] == group for t in row['arms'][a]['tickets']) for a in keys]
                if len(set(counts)) != 1:
                    exclusions['unequal_ticket_counts'] += 1
                elif counts[0] == 0:
                    exclusions['all_skipped'] += 1
                else:
                    samples.append(row)
            metrics = {a: performance(samples, a, group) for a in keys}
            differences = {}
            # Paired changes isolate preservation, conditional positions and interaction.
            for label, left, right in [('preservation', 'baseline', 'preserve_only'),
                                       ('conditional', 'baseline', 'conditional_only'),
                                       ('conditional_after_preservation', 'preserve_only', 'combined')]:
                l, r = metrics[department + ':' + left], metrics[department + ':' + right]
                differences[label] = {m: r[m] - l[m] if r[m] is not None and l[m] is not None else None
                                      for m in ('hit_rate', 'return_rate')}
            comparisons[department + ':' + group] = {
                'paired_races': len(samples), 'race_ids': [r['race_id'] for r in samples],
                'excluded': dict(exclusions), 'arms': metrics, 'paired_differences': differences,
                'risk_only_hits': sum(any(t['group'] == group and t['buy'] == r['actual'] for t in r['arms']['risk']['tickets'])
                                      and not any(t['group'] == group and t['buy'] == r['actual']
                                                  for a in keys[1:] for t in r['arms'][a]['tickets']) for r in samples)}
    # Descriptive risk-only view includes count-mismatched races, explicitly
    # separate from the fair comparison. Quantify dependence on isolated wins.
    risk = {}
    for group in ('本線', '穴'):
        samples = [r for r in rows if any(t['group'] == group for t in r['arms']['risk']['tickets'])]
        risk[group] = performance(samples, 'risk', group)
    total = sum(m['return_yen'] for m in risk.values())
    return comparisons, {'groups': risk, 'hole_return_share': risk['穴']['return_yen'] / total if total else None}


def build_report(output_dir=OUTPUT_DIR):
    folder = Path(output_dir) / 'company'
    folder.mkdir(parents=True, exist_ok=True)
    latest, invalid = latest_bundles(folder / LEDGER)
    result_path = folder / 'annual_position_experiment_results.json'
    outcomes = load_json(result_path, {})
    outcome_ledger = folder / 'annual_position_experiment_outcomes.jsonl'
    observations = {}
    if outcome_ledger.exists():
        for line in outcome_ledger.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            rid = item['race_id']
            if rid not in observations or item['observed_at'] > observations[rid]['observed_at']:
                observations[rid] = item
        outcomes.update({rid: item['outcome'] for rid, item in observations.items()})
    additions = []
    for result in load_json(Path(output_dir) / 'latest_results.json', []):
        rid = str(result.get('race_id', ''))
        if rid not in latest or str(result.get('official_result_available')).lower() not in ('true', '1'):
            continue
        actual = str(result.get('actual_trifecta', ''))
        cars = actual.split('-')
        if len(cars) != 3 or len(set(cars)) != 3 or not all(c.isdigit() and 1 <= int(c) <= 9 for c in cars):
            continue
        try:
            payout = float(result.get('actual_trifecta_odds')) * 100
            if not math.isfinite(payout) or payout <= 0:
                payout = None
        except (ValueError, TypeError):
            payout = None
        if payout is None and outcomes.get(rid, {}).get('actual') == actual:
            payout = outcomes[rid].get('payout_per_100yen')
        outcome = {'actual': actual, 'payout_per_100yen': payout}
        if outcomes.get(rid) != outcome:
            additions.append({'race_id': rid, 'outcome': outcome,
                              'observed_at': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat()})
        outcomes[rid] = outcome
    if additions:
        with outcome_ledger.open('a', encoding='utf-8') as handle:
            for item in additions:
                handle.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + '\n')
    result_path.write_text(json.dumps(outcomes, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    rows = [{**row, **outcomes[rid]} for rid, row in sorted(latest.items()) if rid in outcomes]
    comparisons, risk = summarize(rows)
    report = {'version': VERSION, 'snapshot_policy': POLICY,
              'updated_at_jst': datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
              'frozen_races': len(latest), 'settled_races': len(rows), 'invalid_ledger_rows': invalid,
              'comparisons': comparisons, 'risk_dependency': risk,
              'automatic_promotion': False, 'purchase_authorized': False,
              'status': 'collecting_prospective_evidence',
              'limitations': ['Uncalibrated hypotheses; no demonstrated improvement.',
                             'Same race/time and equal nonzero ticket counts per group required.',
                             'Legacy first/latest forecasts excluded; no retrospective new predictions.']}
    (folder / 'annual_position_experiment_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    render_report(report, folder)
    return report


def render_report(report, folder):
    def rate(v):
        return '未集計' if v is None else f'{v * 100:.1f}%'
    parts = []
    for key, comp in report['comparisons'].items():
        department, group = key.split(':')
        dept = {'data_department': 'データ部', 'pace_department': '展開部', 'line_department': 'ライン部'}[department]
        body = ''.join('<tr><td>' + LABELS[a.split(':')[-1]] + '</td>' +
                       f'<td>{m["races"]}</td><td>{m["hits"]}</td><td>{rate(m["hit_rate"])}</td>' +
                       f'<td>{m["stake_yen"]:,}円</td><td>{rate(m["return_rate"])}</td></tr>'
                       for a, m in comp['arms'].items())
        parts.append(f'<section><h2>{dept}・{group}</h2><div class="scroll"><table><tr><th>方式</th>'
                     '<th>同条件レース</th><th>的中</th><th>的中率</th><th>照合済み投資額</th><th>回収率</th></tr>' + body +
                     '</table></div><p>点数不一致による除外：' + str(comp['excluded'].get('unequal_ticket_counts', 0)) +
                     'レース。候補不足を追加で埋めません。</p></section>')
    dependency = report['risk_dependency']
    risk = dependency['groups']['穴']
    page = '<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">' +\
        '<title>部署別2・3着評価の比較</title><style>body{font-family:system-ui;background:#f4f7fb;color:#172b45;padding:20px}' +\
        'main{max-width:1050px;margin:auto}section{background:white;padding:20px;margin:16px 0;border-radius:12px}' +\
        'p{line-height:1.8}table{border-collapse:collapse;width:100%}th,td{padding:10px;text-align:left;border-bottom:1px solid #ddd}.scroll{overflow:auto}</style><main>' +\
        '<p><a href="annual_department_report.html">部署別予想へ</a> ／ <a href="operations.html">運用状況へ</a></p>' +\
        '<h1>部署別2・3着評価の比較</h1><p>リスク部の現行ロジック・買い目を固定した比較です。' +\
        '①部署評価の保持、②順位別の市場支持と1・2着に応じた3着評価、それぞれ単独と併用で検証します。</p>' +\
        '<p>締切40分〜5分前に全案を同時保存した最新記録のみ。同じレース・時点・本線と穴それぞれ同じ点数・1点100円。' +\
        '各最大12点、穴は取得時100倍以上。旧集計と過去の再予想を混ぜません。</p>' +\
        f'<p>事前保存 {report["frozen_races"]}レース ／ 公式結果照合 {report["settled_races"]}レース。改善効果は未確認・自動採用なし。</p>' +\
        ''.join(parts) + '<section><h2>リスク部の高配当への依存</h2><p>以下は点数不一致も含む参考集計です。</p>' +\
        f'<p>払戻総額に占める穴：{rate(dependency["hole_return_share"])}。穴の最大1的中が穴払戻額に占める割合：{rate(risk["largest_hit_return_share"])}。' +\
        f'その1件の払戻を除いた穴回収率（投資額据置）：{rate(risk["return_rate_without_largest_hit"])}</p></section>' +\
        '<p>的中率は公式着順、回収率は公式払戻まで確認できた共通レースで集計。市場支持と暫定確率は的中確率の保証ではありません。</p>' +\
        '<p><a href="annual_position_experiment_report.json">詳細データ</a></p></main></html>'
    (folder / 'annual_position_experiment_report.html').write_text(page, encoding='utf-8')
