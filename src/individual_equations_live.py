"""Independent prospective equation portfolios; never a purchase executor.

One immutable record freezes all available methods of a family at the same
instant. Results live separately. No inference is performed by report/settle.
"""
import argparse
import hashlib
import html
import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import pandas as pd

import fixed_year_features as features
import fixed_year_models as experts
import fixed_year_study as study
import fusion_equations as fusion
import fusion_input_repair as repair
import fusion_shadow_live as shadow
import fusion_stage_context as stage
import fusion_stage_study as stage_study
from equation_models import LABELS as MARKET_LABELS
from official_outcomes import normalize_outcome, ticket

ROOT = Path(__file__).resolve().parents[1]
JST = ZoneInfo('Asia/Tokyo')
FOLDER = ROOT / 'outputs/company/individual_equations'
STAGE_FOLDER = ROOT / 'models/individual_stage'
LABELS = {**study.NAMES, 'first_anchor': '1着補正付き統合式（38.77%案）',
          'context_joint': '出走人数・クラス別統合式',
          'stage_global': '着順別統合式', 'stage_context': '条件・着順別統合式',
          **{'market_' + k: v for k, v in MARKET_LABELS.items()}}
LABELS['risk'] = 'リスク評価式（再学習入力の参考予想）'
VERSION = 'individual_equations_v1'
STAKE = 100


def clock():
    return datetime.now(JST)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def read_json(path, default):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def ledger(folder):
    records = read_json(folder / 'forecasts.json', [])
    seen = set()
    for row in records:
        value = {k: v for k, v in row.items() if k != 'sha256'}
        key = (row['family'], row['race_id'])
        if digest(value) != row['sha256'] or key in seen:
            raise ValueError('corrupt or duplicate independent forecast evidence')
        seen.add(key)
    return records


def ranked(distribution):
    if not distribution or any(not ticket(k) for k in distribution) or any(not math.isfinite(p) or p <= 0 for p in distribution.values()):
        raise ValueError('invalid independent probability distribution')
    if abs(sum(distribution.values()) - 1) > 1e-8:
        raise ValueError('independent probabilities are not normalized')
    return [{'buy': buy, 'probability': float(distribution[buy]), 'stake_yen': STAKE}
            for buy in sorted(distribution, key=lambda b: (-distribution[b],
                tuple(map(int, b.split('-')))))[:12]]


def freeze(records, family, meta, methods, model_ids, now):
    """The actual completion clock, not a supplied historical --asof, gates writes."""
    if any(r['race_id'] == meta['race_id'] and r['family'] == family for r in records):
        return False
    if now.tzinfo is None or meta['date'] != now.astimezone(JST).date().isoformat() or not math.isfinite(meta['close_at']) or now.timestamp() >= meta['close_at']:
        return False
    if any(m['training_cutoff_exclusive'] > now.astimezone(JST).date().isoformat()
           for m in model_ids.values()):
        raise ValueError('model training overlaps forecast date')
    sizes = {len(v['tickets']) for v in methods.values()}
    if len(sizes) != 1 or not sizes or not 1 <= next(iter(sizes)) <= 12:
        raise ValueError('all methods must use the same nonempty ticket budget')
    for method in methods.values():
        tickets = method['tickets']
        if len({t['buy'] for t in tickets}) != len(tickets) or any(t['stake_yen'] != STAKE or not ticket(t['buy']) for t in tickets):
            raise ValueError('invalid independent ticket stakes or duplicates')
    row = {'version': VERSION, **meta, 'family': family,
           'snapshot_at_jst': now.isoformat(), 'methods': methods, 'models': model_ids,
           'source_code_sha256': hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
           'purchase_authorized': False}
    row['sha256'] = digest(row)
    records.append(row)
    return True


def load_stage(asof):
    model, manifest = stage_study.load_stage(STAGE_FOLDER)
    path = STAGE_FOLDER / 'frozen_model.joblib'
    if study.digest_file(path) != manifest['source_model_sha256']:
        raise ValueError('stage base model fingerprint mismatch')
    if manifest['training_cutoff_exclusive'] > asof:
        raise ValueError('stage training overlaps forecast date')
    return model, manifest, joblib.load(path)


def race_distributions(race, bundle, old_race, old_bundle, stage_model):
    packet = experts.make_packet(race, bundle['profiles'])
    values = experts.prediction(bundle['equations'], packet)
    values['first_anchor'] = repair.anchor_first(values['fusion'], values['original'])
    old_packet = experts.make_packet(old_race, old_bundle['profiles'])
    available = experts.components(old_bundle['equations'], old_packet)
    keys, matrix = stage.expert_matrix(old_bundle['equations']['pool'], available)
    baseline = fusion.temperature(fusion.mix(old_bundle['equations']['pool'], available),
                                  old_bundle['equations']['temperature'])
    context = stage.context_key(len(old_race), str(old_race.iloc[0].get('race_class', 'missing')))
    for name, distribution in stage.predict(stage_model, keys, matrix, context,
            [baseline[k] for k in keys]).items():
        values[name] = dict(zip(keys, distribution))
    if any(not fusion.valid_distribution(p, fusion.keys(packet)) for p in values.values()):
        raise ValueError('equation fields or normalized probabilities disagree')
    return {name: {'basis': 'trained', 'tickets': ranked(p)} for name, p in values.items()}


def capture_market(records, folder, decided):
    # Only immutable, validated canonical observations from the existing lab.
    # A new top-12 decision is frozen NOW, never attached to a past timestamp.
    import equation_lab
    source_folder = folder.parent
    rows, _ = equation_lab.validated_rows(source_folder)
    count = 0
    for row in sorted(rows, key=lambda r: r['snapshot_at'], reverse=True):
        now = clock()
        if row['race_id'] in decided or row['date'] != now.date().isoformat() or now.timestamp() >= row['close_at']:
            continue
        if not row.get('standalone'):
            continue
        methods = {}
        for name, item in row['standalone'].items():
            # Each odds band already preserves its own top 12, so its union
            # contains every possible member of the overall top 12.
            pool = [t for group in item['picks'].values() for t in group]
            picks = sorted(pool, key=lambda t: (-t['prob'], tuple(map(int, t['buy'].split('-')))))[:12]
            methods['market_' + name] = {'basis': 'trained' if item['basis'] == 'trained_shadow' else 'provisional',
                'tickets': [{'buy': t['buy'], 'probability': t['prob'], 'odds_at_capture': t['odds'],
                             'stake_yen': STAKE} for t in picks]}
        meta = {k: row[k] for k in ('race_id', 'date', 'venue', 'race_no', 'close_at')}
        meta.update(source_record=row['record_sha256'], source_snapshot_at=row['snapshot_at'])
        count += freeze(records, 'market_equations', meta, methods,
                        {'market': {'training_cutoff_exclusive': row['date'],
                                    'model_record': row['model_record']}}, clock())
    return count


def forecast(folder=FOLDER):
    records = ledger(folder)
    now = clock()
    day = now.date().isoformat()
    results = read_json(ROOT / 'outputs/latest_results.json', [])
    decided = {str(r.get('race_id')) for r in results if normalize_outcome(r)}
    entries = pd.read_csv(ROOT / 'data/raw/today_entries.csv', dtype={'race_id': str, 'player_id': str})
    entries['date'] = pd.to_datetime(entries.date, errors='coerce').dt.strftime('%Y-%m-%d')
    today = entries[entries.date.eq(day)].copy()
    if today.empty:
        raise ValueError('today entry feed is empty or stale')
    known = {r['race_id'] for r in records if r['family'] == 'archive_models'}
    coverage, accepted = [], []
    for rid, race in today.groupby('race_id', sort=False):
        stamp = pd.to_numeric(race.close_at, errors='coerce')
        reason = ''
        if str(rid) in known:
            reason = '保存済み'
        elif str(rid) in decided:
            reason = '開始前に結果確定'
        elif stamp.isna().any() or stamp.nunique() != 1:
            reason = '締切時刻を確認できない'
        elif float(stamp.iloc[0]) <= now.timestamp():
            reason = '初回確認時点で締切後'
        else:
            reason = study.input_reason(race) or ''
        item = race.iloc[0]
        coverage.append({'race_id': str(rid), 'venue': str(item.get('venue', '')),
                         'race_no': int(item.race_no), 'reason': reason})
        if not reason:
            accepted.append(race)
    if accepted:
        clean, _ = study.eligible(features.mask_outcomes(pd.concat(accepted)))
        manifest, bundle = shadow.load_model(now.date())
        stage_model, stage_manifest, stage_bundle = load_stage(day)
        predicted = shadow.restore_race_metadata(study.base_predict(clean, bundle), clean)
        old_predicted = study.base_predict(clean, stage_bundle)
        old_by_id = {str(rid): r for rid, r in old_predicted.groupby('race_id', sort=False)}
        for rid, race in predicted.groupby('race_id', sort=False):
            item = race.iloc[0]
            methods = race_distributions(race, bundle, old_by_id[str(rid)], stage_bundle, stage_model)
            meta = {'race_id': str(rid), 'date': str(item.date), 'venue': str(item.get('venue', '')),
                    'race_no': int(item.race_no), 'close_at': float(item.close_at),
                    'input_sha256': hashlib.sha256(clean[clean.race_id.astype(str).eq(str(rid))].to_json(orient='records').encode()).hexdigest()}
            frozen = freeze(records, 'archive_models', meta, methods,
                {'refit': {'training_cutoff_exclusive': manifest['training_cutoff_exclusive'],
                           'model_sha256': manifest['model_sha256']},
                 'stage': {'training_cutoff_exclusive': stage_manifest['training_cutoff_exclusive'],
                           'model_sha256': stage_manifest['stage_model_sha256']}}, clock())
            for row in coverage:
                if row['race_id'] == str(rid):
                    row['reason'] = '保存済み' if frozen else '計算完了時点で締切後'
    capture_market(records, folder, decided)
    save(folder / 'forecasts.json', records)
    save(folder / 'coverage.json', {'date': day, 'races': coverage, 'updated_at': clock().isoformat()})
    return report(folder)


def update_results(folder, records, results):
    saved = read_json(folder / 'results.json', {})
    closes = {r['race_id']: r['close_at'] for r in records}
    for result in results:
        rid = str(result.get('race_id'))
        outcome = normalize_outcome(result)
        if rid not in closes or not outcome or closes[rid] >= clock().timestamp():
            continue
        old = saved.get(rid)
        if old and (old['winning_buys'] != outcome['winning_buys'] or any(
                not math.isclose(price, outcome['payouts'][buy], abs_tol=0.01, rel_tol=1e-10)
                for buy, price in old['payouts'].items() if buy in outcome['payouts'])):
            old['conflict'] = True
            continue
        if old:
            outcome['payouts'] = {**old['payouts'], **outcome['payouts']}
            outcome['payout_complete'] = set(outcome['payouts']) == set(outcome['winning_buys'])
            outcome['conflict'] = old.get('conflict', False)
        saved[rid] = outcome
    save(folder / 'results.json', saved)
    return saved


def metrics(rows, outcomes, name, limit):
    settled, hits, roi_races, stake, returned, payouts = 0, 0, 0, 0, 0., []
    over50 = over100 = 0
    for row in rows:
        outcome = outcomes.get(row['race_id'])
        if not outcome or outcome.get('conflict'):
            continue
        tickets = row['methods'][name]['tickets'][:limit]
        winning = set(outcome['winning_buys']) & {t['buy'] for t in tickets}
        settled += 1
        hits += bool(winning)
        if not outcome['payout_complete']:
            continue
        roi_races += 1
        stake += sum(t['stake_yen'] for t in tickets)
        cash = sum(outcome['payouts'].get(t['buy'], 0) * t['stake_yen'] / 100 for t in tickets)
        returned += cash
        payouts.append(cash)
        over50 += any(outcome['payouts'][t] > 5000 for t in winning)
        over100 += any(outcome['payouts'][t] >= 10000 for t in winning)
    return {'forecast_races': len(rows), 'settled_races': settled, 'hits': hits,
            'hit_rate': hits / settled if settled else None, 'roi_races': roi_races,
            'payout_pending_races': settled - roi_races, 'stake_yen': stake,
            'return_yen': returned, 'roi': returned / stake if stake else None,
            'hits_over_50x': over50, 'hits_at_least_100x': over100,
            'largest_return_share': max(payouts, default=0) / returned if returned else None}


def report(folder=FOLDER):
    records = ledger(folder)
    outcomes = update_results(folder, records, read_json(ROOT / 'outputs/latest_results.json', []))
    coverage = read_json(folder / 'coverage.json', {})
    stats = {}
    for name in LABELS:
        for basis in ('trained', 'provisional'):
            rows = [r for r in records if r['methods'].get(name, {}).get('basis') == basis]
            if rows:
                stats[name + ':' + basis] = {str(n): metrics(rows, outcomes, name, n) for n in (1, 6, 12)}
    value = {'version': VERSION, 'updated_at_jst': clock().isoformat(), 'equations': LABELS,
             'statistics': stats, 'coverage': coverage, 'records': len(records),
             'forecast_policy': 'first immutable pre-close top12 per family and race, 100 yen per ticket',
             'comparison_policy': 'same family, same basis, same race IDs, same timestamp, same ticket limit',
             'purchase_authorized': False, 'result_conflicts': sum(bool(o.get('conflict')) for o in outcomes.values())}
    save(folder / 'report.json', value)
    render(folder, records, outcomes, value)
    return value


def render(folder, records, outcomes, value):
    esc = lambda x: html.escape(str(x))
    pct = lambda x: '未集計' if x is None else f'{x:.1%}'
    page = ['<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>方程式ごとの独立予想｜KEIRIN NEXUS</title><style>body{background:#0b1322;color:#e8edf7;font:16px/1.7 system-ui;margin:0}main{max-width:1200px;margin:auto;padding:24px 16px}a{color:#9bd8ff}section{background:#15243a;padding:18px;margin:20px 0;border-radius:12px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #405068}td{min-width:80px}.picks{min-width:340px;max-width:650px}.badge{color:#fedf94}select{padding:12px;font-size:16px;width:100%;max-width:650px}small{color:#bdcadd}</style><main>',
        '<p><a href="../../index.html">今日の予想</a> ／ <a href="../fusion_shadow_live_report.html">38.77%案の既存成績</a> ／ <a href="../annual_equation_report.html">市場・隊列式の学習状況</a></p>',
        '<h1>方程式ごとの独立予想</h1><p>各式が自分の計算で選んだ三連単を、最大12点ずつ表示します。各100円の仮想購入です。本線・穴の購入枠や社長の選別とは別に保存しています。</p>',
        '<p>1点・6点・12点で的中率と回収率を分けて集計。回収率は公式払戻しが揃ったレースだけを対象とし、未取得件数も表示します。同着は該当する買い目すべての払戻しを合算します。</p>',
        '<p>学習済み18式は同じ入力・時刻・点数で比較します。市場を使う3式は別の共通入力群です。未学習の暫定式の成績は学習済みと混ぜません。集計対象レースが違う群同士の順位付けはできません。</p>',
        '<label for="equation">表示する方程式</label><br><select id="equation" onchange="document.querySelectorAll(\'[data-equation]\').forEach(e=>e.hidden=this.value!==\'all\'&amp;&amp;e.dataset.equation!==this.value)"><option value="all">すべての方程式</option>']
    page.extend(f'<option value="{esc(n)}"' + (' selected' if n == 'first_anchor' else '') + f'>{esc(label)}</option>' for n, label in LABELS.items())
    page.append('</select>')
    for name, label in LABELS.items():
        page.append(f'<section data-equation="{esc(name)}"' + (' hidden' if name != 'first_anchor' else '') + f'><h2>{esc(label)}</h2>')
        rows = [r for r in records if name in r['methods']]
        if not rows:
            message = ('完全な締切前市場データと学習状況を確認中です。暫定式は暫定と明記します。'
                       if name.startswith('market_') else '次の締切前予想から記録します。')
            page.append(f'<p>{message}</p>')
        for basis in ('trained', 'provisional'):
            stat = value['statistics'].get(name + ':' + basis)
            if not stat:
                continue
            page.append('<h3>' + ('学習済み' if basis == 'trained' else '未学習・固定式による暫定参考') + '</h3><div class="scroll"><table><tr><th>点数</th><th>的中 / 確定</th><th>的中率</th><th>回収率</th><th>仮想投資</th><th>払戻し</th><th>払戻未取得</th></tr>')
            for n, s in stat.items():
                page.append(f'<tr><td>上位{n}点</td><td>{s["hits"]} / {s["settled_races"]}</td><td>{pct(s["hit_rate"])}</td><td>{pct(s["roi"])}</td><td>{s["stake_yen"]:,}円</td><td>{s["return_yen"]:,.0f}円</td><td>{s["payout_pending_races"]}R</td></tr>')
            page.append('</table></div>')
            s = stat['12']
            page.append(f'<p>50倍超的中 {s["hits_over_50x"]}R ／ 100倍以上的中 {s["hits_at_least_100x"]}R ／ 最大払戻への依存 {pct(s["largest_return_share"])}</p>')
        page.append('<div class="scroll"><table><tr><th>日付・場・R</th><th>この式だけの買い目（順位順）</th><th>確定着順</th><th>12点判定</th><th>予想保存時刻</th><th>学習の終了境界</th></tr>')
        for row in sorted(rows, key=lambda r: (r['date'], r['close_at']), reverse=True):
            method = row['methods'][name]
            out = outcomes.get(row['race_id'])
            picks = [t['buy'] for t in method['tickets']]
            status = '結果待ち'
            if out:
                status = '公式結果の不一致・集計除外' if out.get('conflict') else ('的中' if set(picks) & set(out['winning_buys']) else '不的中')
            basis = '暫定参考' if method['basis'] == 'provisional' else '学習済み'
            model_key = 'market' if name.startswith('market_') else ('stage' if name in stage.VARIANTS else 'refit')
            cutoff = row['models'][model_key]['training_cutoff_exclusive']
            page.append(f'<tr><td>{esc(row["date"])}<br>{esc(row["venue"])} {esc(row["race_no"])}R<br><small>{basis}</small></td><td class="picks">{esc(" ／ ".join(picks))}</td><td>{esc(" ／ ".join(out["winning_buys"]) if out else "未確定")}</td><td>{status}</td><td>{esc(row["snapshot_at_jst"])}</td><td>{esc(cutoff)}より前</td></tr>')
        page.append('</table></div></section>')
    page.append('<section><h2>全レースの保存状況</h2><p>初回確認が締切後だったレースは後付けしません。</p><table><tr><th>場・R</th><th>状態・理由</th></tr>')
    for row in value['coverage'].get('races', []):
        page.append(f'<tr><td>{esc(row["venue"])} {esc(row["race_no"])}R</td><td>{esc(row["reason"] or "入力確認中")}</td></tr>')
    page.append('</table></section><p><a href="report.json">集計データ</a> ／ <a href="forecasts.json">締切前に固定した予想</a> ／ <a href="results.json">公式結果</a></p></main></html>')
    (folder / 'index.html').write_text(''.join(page), encoding='utf-8')


def merge(remote, folder=FOLDER):
    """Remote publication wins for an already committed race; never rewrite picks."""
    remote_rows = ledger(remote)
    remote_keys = {(r['family'], r['race_id']) for r in remote_rows}
    combined = remote_rows + [r for r in ledger(folder) if (r['family'], r['race_id']) not in remote_keys]
    save(folder / 'forecasts.json', combined)
    # Keep official outcomes from either writer, checking any overlaps.
    existing = read_json(folder / 'results.json', {})
    other = read_json(remote / 'results.json', {})
    for rid, outcome in other.items():
        if rid not in existing:
            existing[rid] = outcome
        elif digest(existing[rid]) != digest(outcome):
            left = existing[rid]
            if left['winning_buys'] != outcome['winning_buys'] or any(
                    not math.isclose(p, outcome['payouts'][b], abs_tol=.01, rel_tol=1e-10)
                    for b, p in left['payouts'].items() if b in outcome['payouts']):
                left['conflict'] = True
            else:
                left['payouts'].update(outcome['payouts'])
                left['payout_complete'] = set(left['payouts']) == set(left['winning_buys'])
                left['conflict'] = left.get('conflict', False) or outcome.get('conflict', False)
    save(folder / 'results.json', existing)
    return report(folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('forecast', 'report', 'merge'))
    parser.add_argument('--remote', type=Path)
    args = parser.parse_args()
    value = forecast() if args.command == 'forecast' else merge(args.remote) if args.command == 'merge' else report()
    print(json.dumps({'records': value['records'], 'methods_with_forecasts': len(value['statistics'])}))
