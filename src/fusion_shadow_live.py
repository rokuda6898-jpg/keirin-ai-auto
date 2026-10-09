"""Train, freeze and settle a separate prospective copy of the 38.77% formula.

This path does not alter the production seven departments, risk tickets or the
canonical forecast. Each race is frozen before its close; official outcomes
are joined later from the settlement feed.
"""
import argparse
import ast
import gzip
import hashlib
import html
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

import fixed_year_features as features
import fixed_year_models as experts
import fixed_year_study as old
import fusion_equations as fusion
import fusion_input_repair as repair
from official_outcomes import normalize_outcome

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'models/fusion_shadow_refit_first_anchor.joblib'
MANIFEST = ROOT / 'models/fusion_shadow_refit_first_anchor.json'
COMPANY = ROOT / 'outputs/company'
LEDGER = COMPANY / 'fusion_shadow_live_ledger.jsonl'
LATEST = COMPANY / 'fusion_shadow_live_predictions.json'
REPORT_JSON = COMPANY / 'fusion_shadow_live_report.json'
REPORT_HTML = COMPANY / 'fusion_shadow_live_report.html'
DROPPED = repair.ARMS['refit']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_hashes():
    names = ('fusion_input_repair.py', 'official_outcomes.py')
    return {**old.source_hashes(),
            **{name: digest(ROOT / 'src' / name) for name in names}}


def model_pipeline_hash():
    """Fingerprint fitting/scoring code while leaving report copy/UI editable."""
    source = ast.parse(Path(__file__).read_text(encoding='utf-8'))
    relevant = {'dated_phases', 'digest', 'load_model', 'rank_race_candidates',
                'source_hashes', 'train'}
    payload = '\n'.join(ast.dump(node, include_attributes=False) for node in source.body
                       if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in relevant)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def dated_phases(frame):
    days = sorted(frame.date.astype(str).unique())
    if len(days) < 100:
        raise ValueError('at least 100 distinct historical race dates are required')
    cuts = [0, int(len(days)*.6), int(len(days)*.8), int(len(days)*.9), len(days)]
    phases = [frame[frame.date.isin(days[a:b])].reset_index(drop=True)
              for a, b in zip(cuts, cuts[1:])]
    if any(part.empty for part in phases):
        raise ValueError('chronological training phase is empty')
    for before, after in zip(phases, phases[1:]):
        if max(before.date) >= min(after.date):
            raise ValueError('chronological phases overlap')
    return phases


def train(history_path, asof):
    cutoff = pd.Timestamp(asof).date().isoformat()
    history = pd.read_csv(history_path, dtype={'race_id': str, 'player_id': str}, low_memory=False)
    history['date'] = pd.to_datetime(history.date, errors='coerce').dt.strftime('%Y-%m-%d')
    history = history[history.date.notna() & history.date.lt(cutoff)].copy()
    if not len(history) or not pd.to_datetime(history.date).lt(pd.Timestamp(cutoff)).all():
        raise ValueError('training history must end strictly before the forecast date')
    clean, exclusions = old.eligible(history, labels=True)
    phases = dated_phases(clean)
    folder = MODEL.parent / 'fit'
    folder.mkdir(parents=True, exist_ok=True)
    base = repair.fit_base(phases[0], DROPPED, folder/'base', repair.boundary(phases[0]))
    equations = repair.fit_equations(phases[1], base, folder)
    phase_log = []
    for i, label in ((2, 'mixture'), (3, 'temperature')):
        past = pd.concat(phases[:i], ignore_index=True)
        base = repair.fit_base(past, DROPPED, folder/label, repair.boundary(past))
        phase_log.append({'purpose': label, 'fit_last': base['base_training_last'],
                          'predict_first': min(phases[i].date), 'predict_last': max(phases[i].date)})
        if label == 'mixture':
            equations['pool'] = fusion.fit_pool(list(repair.training_records(phases[i], base, equations)))
        else:
            losses = {tau: 0.0 for tau in fusion.CONFIG['temperatures']}
            count = 0
            for actual, available in repair.training_records(phases[i], base, equations):
                mixed = fusion.mix(equations['pool'], available)
                for tau in losses:
                    losses[tau] -= math.log(fusion.temperature(mixed, tau)[actual])
                count += 1
            if not count:
                raise ValueError('no chronological calibration races')
            losses = {tau: value/count for tau, value in losses.items()}
            equations['calibration_fit_losses'] = losses
            equations['temperature'] = min(losses, key=lambda tau: (losses[tau], abs(tau-1)))
    base = repair.fit_base(clean, DROPPED, folder/'final', cutoff)
    base['equations'] = equations
    base['training_cutoff_exclusive'] = cutoff
    base['shadow_formula'] = 'refit_first_anchor'
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(base, MODEL, compress=3)
    manifest = {
        'version': 'fusion_shadow_refit_first_anchor_v1', 'formula': 'P(anchor)=P(refit_base_first)*P(refit_fusion|first)',
        'model_sha256': digest(MODEL), 'source_hashes': source_hashes(),
        'model_pipeline_hash': model_pipeline_hash(),
        'history_sha256': digest(history_path), 'training_cutoff_exclusive': cutoff,
        'training_first': min(clean.date), 'training_last': max(clean.date),
        'training_races': int(clean.race_id.nunique()), 'eligible_rows': len(clean),
        'exclusions': exclusions, 'phases': [
            {'purpose': name, 'first': min(part.date), 'last': max(part.date), 'races': int(part.race_id.nunique())}
            for name, part in zip(('base', 'equations', 'mixture', 'temperature'), phases)],
        'expanded_base_refits': phase_log, 'temperature': equations['temperature'],
        'risk_production_changed': False, 'all_seven_departments_changed': False,
        'purchase_authorized': False,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('model_sha256', 'training_cutoff_exclusive',
          'training_races', 'training_last', 'temperature')}, ensure_ascii=False))
    return manifest


def load_model(asof):
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    recorded_sources = dict(manifest.get('source_hashes', {}))
    # Older shadow bundles included a whole-file hash. Ignore only that legacy
    # key: the stable AST fingerprint below covers model fitting and ranking.
    legacy_module_hash = recorded_sources.pop('fusion_shadow_live.py', None)
    if recorded_sources != source_hashes():
        raise ValueError('shadow model dependencies do not match their manifest')
    saved_pipeline = manifest.get('model_pipeline_hash')
    if saved_pipeline and saved_pipeline != model_pipeline_hash():
        raise ValueError('shadow fitting/scoring code does not match its manifest')
    if not saved_pipeline and not legacy_module_hash:
        raise ValueError('shadow model manifest has no verifiable pipeline fingerprint')
    if digest(MODEL) != manifest['model_sha256']:
        raise ValueError('shadow model or source does not match its manifest')
    if manifest['training_cutoff_exclusive'] > pd.Timestamp(asof).date().isoformat():
        raise ValueError('model was trained on or after the forecast date')
    return manifest, joblib.load(MODEL)


def read_ledger():
    if not LEDGER.exists():
        return []
    rows = []
    for line in LEDGER.read_text(encoding='utf-8').splitlines():
        try:
            row = json.loads(line)
            if row.get('race_id') and row.get('top12'):
                rows.append(row)
        except json.JSONDecodeError:
            continue
    return rows


def merge_remote_ledger(remote_path):
    """Union a concurrently published ledger without replacing frozen picks."""
    local = {str(row['race_id']): row for row in read_ledger()}
    remote = []
    if remote_path and Path(remote_path).exists():
        for line in Path(remote_path).read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
                if row.get('race_id') and row.get('top12'):
                    remote.append(row)
            except json.JSONDecodeError:
                continue
    for incoming in remote:
        rid = str(incoming['race_id'])
        if rid not in local:
            local[rid] = incoming
        elif incoming.get('actual') and not local[rid].get('actual'):
            local[rid].update({key: value for key, value in incoming.items()
                               if key in ('actual', 'top12_hit', 'hit_tickets', 'hit_odds',
                                          'actual_odds', 'result_source', 'result_observed_at_jst')})
    rows = list(local.values())
    results = json.loads(LATEST_RESULTS_JSON.read_text(encoding='utf-8')) if LATEST_RESULTS_JSON.exists() else []
    rows = apply_results(rows, results)
    write_ledger(rows)
    return build_report(rows, datetime.now(ZoneInfo('Asia/Tokyo')))


def write_ledger(rows):
    COMPANY.mkdir(parents=True, exist_ok=True)
    temp = LEDGER.with_suffix('.jsonl.tmp')
    with temp.open('w', encoding='utf-8') as stream:
        for row in sorted(rows, key=lambda x: (x.get('date', ''), x.get('race_id', ''))):
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
    temp.replace(LEDGER)


def apply_results(rows, results):
    by_id = {str(item.get('race_id')): item for item in results if item.get('official_result_available')}
    for row in rows:
        result = by_id.get(str(row['race_id']))
        if not result:
            continue
        outcome = normalize_outcome({**result, 'official_result_available': True})
        if not outcome:
            continue
        winners = outcome['winning_buys']
        hits = sorted(set(row['top12']) & set(winners))
        payouts = outcome['payouts']  # Official source values are yen per 100-yen ticket.
        hit_odds = [float(payouts[ticket]) / 100 for ticket in hits
                    if ticket in payouts and pd.notna(payouts[ticket])]
        row['actual'] = '|'.join(winners)
        row['top12_hit'] = bool(hits)
        row['hit_tickets'] = hits
        row['result_source'] = str(result.get('result_source') or 'official_result_feed')
        row['result_observed_at_jst'] = result.get('result_observed_at_jst')
        # A race is counted as a high-payout hit when any matching ticket in the
        # frozen candidate set has that official payout. Unknown stays unknown.
        row['hit_odds'] = hit_odds
        row['actual_odds'] = max(hit_odds) if hit_odds else None
    return rows


def build_report(rows, now):
    decided = [r for r in rows if r.get('actual')]
    hits = [r for r in decided if r.get('top12_hit')]
    known = [r for r in hits if r.get('hit_odds')]
    active = [r for r in rows if not r.get('actual')]
    latest_snapshot = {}
    training_cutoff = None
    training_first = None
    if LATEST.exists():
        try:
            latest_document = json.loads(LATEST.read_text(encoding='utf-8'))
            latest_snapshot = latest_document.get('coverage', {})
            training_cutoff = latest_document.get('model_training_cutoff_exclusive')
            training_first = latest_document.get('model_training_first')
        except (OSError, json.JSONDecodeError):
            latest_snapshot = {}
    summary = {
        'updated_at_jst': now.isoformat(timespec='seconds'),
        'status': 'ready' if rows else 'waiting_for_first_frozen_forecast',
        'forecast_races': len(rows), 'settled_races': len(decided), 'hit_races': len(hits),
        'hit_races_top1': sum(bool(set(r.get('hit_tickets', [])) & set(r['top12'][:1])) for r in decided),
        'hit_races_top6': sum(bool(set(r.get('hit_tickets', [])) & set(r['top12'][:6])) for r in decided),
        'hit_races_top12': len(hits),
        'hit_rate_top1': sum(bool(set(r.get('hit_tickets', [])) & set(r['top12'][:1])) for r in decided)/len(decided) if decided else None,
        'hit_rate_top6': sum(bool(set(r.get('hit_tickets', [])) & set(r['top12'][:6])) for r in decided)/len(decided) if decided else None,
        'hit_rate_top12': len(hits)/len(decided) if decided else None,
        'hit_rate': len(hits)/len(decided) if decided else None,
        'official_payout_known_hits': len(known),
        'hit_payout_over_50x': sum(any(odds > 50 for odds in r['hit_odds']) for r in known),
        'hit_payout_over_100x': sum(any(odds > 100 for odds in r['hit_odds']) for r in known),
        'payout_unknown_hits': len(hits)-len(known),
        'open_races': len(active), 'purchase_authorized': False,
        'latest_input_races': latest_snapshot.get('input_races', 0),
        'latest_forecasted_races': latest_snapshot.get('forecasted_races', 0),
        'latest_skipped_races': latest_snapshot.get('skipped_races', 0),
        'latest_coverage': latest_snapshot,
        'model_training_first': training_first,
        'model_training_cutoff_exclusive': training_cutoff,
        'metrics_scope': 'ranked exact-order trifecta candidates per race: top1, top6 and top12; not main/hole tickets or ROI',
        'forecast_ledger_sha256': digest(LEDGER) if LEDGER.exists() else None,
    }
    REPORT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    def pct(value): return '未集計' if value is None else f'{value:.1%}'
    body = [
        '<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>統合式の別予想・今後の成績｜KEIRIN NEXUS</title>',
        '<style>body{margin:0;background:#0b1322;color:#e8edf7;font:16px/1.8 system-ui}main{max-width:1100px;margin:auto;padding:28px 18px}a{color:#8dc8ff}.notice{background:#182d43;padding:16px;border-left:4px solid #70c8bc}.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}.stats div{background:#14253a;padding:14px;border-radius:12px}.stats b{font-size:24px;display:block}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap}td,th{text-align:left;padding:9px;border-bottom:1px solid #324157}small{color:#bac8d9}</style></head><body><main>',
        '<p><a href="../index.html">今日の予想へ戻る</a> ／ <a href="fusion_input_repair_report.html">過去一年の方程式検証</a></p>',
        '<h1>統合式の別予想・今後の成績</h1>',
        '<p class="notice">過去検証の38.77%案を、別の影予想として毎日保存します。モデルは直近までの確定履歴だけで学習し、レースごとに締切前の最大12候補を一度だけ固定します。的中判定は公式結果を取得してから更新します。</p>',
        '<p>三連単の順位上位1点・6点・12点それぞれの的中率を表示します。過去検証の38.77%は12点時の値なので、将来の12点欄と比較してください。現行の本線・穴の選別、購入点数・投資額、実購入成績、回収率とは別集計です。高配当回数は、保存候補が的中したレースの公式払戻倍率で数えます。</p>',
        '<section class="stats">',
        f'<div>締切前に固定<b>{summary["forecast_races"]:,}</b>レース</div><div>結果確定<b>{summary["settled_races"]:,}</b>レース</div>',
        f'<div>直近対象日に確認<b>{summary["latest_input_races"]:,}</b>レース</div><div>直近対象日の予想<b>{summary["latest_forecasted_races"]:,}</b>レース</div><div>予想なし・理由表示<b>{summary["latest_skipped_races"]:,}</b>レース</div>',
        f'<div>1点的中率<b>{pct(summary["hit_rate_top1"])}</b>（{summary["hit_races_top1"]:,}）</div>',
        f'<div>6点的中率<b>{pct(summary["hit_rate_top6"])}</b>（{summary["hit_races_top6"]:,}）</div>',
        f'<div>12点的中率<b>{pct(summary["hit_rate_top12"])}</b>（{summary["hit_races_top12"]:,}）</div>',
        f'<div>的中かつ50倍超<b>{summary["hit_payout_over_50x"]:,}</b>レース</div><div>的中かつ100倍超<b>{summary["hit_payout_over_100x"]:,}</b>レース</div></section>',
        f'<p>モデルの学習期間：{html.escape(str(summary["model_training_first"] or "未取得"))}〜{html.escape(str(summary["model_training_cutoff_exclusive"] or "未取得"))}の前日まで。</p>',
        f'<p>的中のうち公式倍率未取得：{summary["payout_unknown_hits"]}レース。買い目候補は各レースの確率上位12通りです。</p>',
        '<h2>最近の確定結果と、締切前に保存した候補</h2><div class="scroll"><table><thead><tr><th>日付</th><th>場・R</th><th>締切前候補（上位12）</th><th>実際の3連単</th><th>結果（1点・6点・12点）</th><th>公式払戻倍率</th><th>予想時刻</th></tr></thead><tbody>']
    for row in sorted(rows, key=lambda r: (r.get('date',''), r.get('race_id','')), reverse=True)[:250]:
        venue = html.escape(str(row.get('venue') or ''))
        actual = html.escape(str(row.get('actual') or '未確定'))
        status = ('／'.join('的中' if set(row.get('hit_tickets', [])) & set(row['top12'][:n]) else '不的中' for n in (1,6,12))) if row.get('actual') else '結果待ち'
        odds = f'{row["actual_odds"]:.1f}倍（候補的中）' if row.get('actual_odds') is not None else ('未取得' if row.get('actual') else '—')
        picks = html.escape(' ／ '.join(row['top12']))
        created = html.escape(str(row.get('snapshot_at_jst') or ''))
        body.append(f'<tr><td>{html.escape(str(row.get("date", "")))}</td><td>{venue} {row.get("race_no", "")}R</td><td>{picks}</td><td>{actual}</td><td>{status}</td><td>{odds}</td><td>{created}</td></tr>')
    if not rows:
        body.append('<tr><td colspan="7">最初の締切前予想を保存する準備中です。</td></tr>')
    coverage = summary['latest_coverage']
    body.extend(['</tbody></table></div>', '<h2>直近対象日の全レース確認</h2>',
        '<p>開催情報に載ったレースを、買い目を固定したものと予想できなかったものに分けて表示します。締切後に初めて検出したレースは予想を捏造せず、理由を記録します。</p>',
        '<div class="scroll"><table><thead><tr><th>日付</th><th>場・R</th><th>状態</th><th>理由</th></tr></thead><tbody>'])
    for item in coverage.get('races', []):
        body.append('<tr><td>{}</td><td>{} {}R</td><td>{}</td><td>{}</td></tr>'.format(
            html.escape(str(item.get('date', ''))), html.escape(str(item.get('venue', ''))),
            html.escape(str(item.get('race_no', ''))), html.escape(str(item.get('status', ''))),
            html.escape(str(item.get('reason', '')))))
    if not coverage.get('races'):
        body.append('<tr><td colspan="4">当日の開催情報がまだありません。</td></tr>')
    body.extend(['</tbody></table></div>', '<h2>この集計の対象</h2><p>同じレースで候補を後から作り直しません。締切後に初めて登場したレースは追加しません。結果確定後に実際の三連単と公式払戻を照合します。同着など順序が一意でないレースは的中率から除外します。</p>',
        '<p>学習・候補生成・公式結果の照合は本番リスク部と分離しています。候補は購入を指示するものではありません。学習日は結果レポートに表示します。</p>',
        '<p><a href="fusion_shadow_live_report.json">集計JSON</a> ／ <a href="fusion_shadow_live_ledger.jsonl">締切前予想台帳</a></p></main></body></html>'])
    REPORT_HTML.write_text(''.join(body), encoding='utf-8')
    return summary


def rank_race_candidates(race, bundle):
    """Apply the frozen formula independently to this race and rank exact tickets."""
    packet = experts.make_packet(race, bundle['profiles'])
    available = experts.components(bundle['equations'], packet)
    distribution = fusion.temperature(
        fusion.mix(bundle['equations']['pool'], available), bundle['equations']['temperature'])
    base = available['original:model_positions']
    anchored = repair.anchor_first(distribution, base)
    keys = fusion.keys(packet)
    return sorted(keys, key=lambda key: (-anchored[key], tuple(map(int, key.split('-'))))), anchored


def forecast(entries_path, results_path, asof=None):
    now = datetime.now(ZoneInfo('Asia/Tokyo'))
    if asof:
        now = datetime.fromisoformat(asof).astimezone(ZoneInfo('Asia/Tokyo'))
    manifest, bundle = load_model(now.date())
    entries = pd.read_csv(entries_path, dtype={'race_id': str, 'player_id': str}, low_memory=False)
    if not {'race_id', 'date', 'car_no', 'player_id', 'close_at'}.issubset(entries.columns):
        raise ValueError('entry feed is missing race identity or closing time')
    entries['date'] = pd.to_datetime(entries.date, errors='coerce').dt.strftime('%Y-%m-%d')
    close = pd.to_numeric(entries.close_at, errors='coerce')
    today_entries = entries[entries.date.eq(now.strftime('%Y-%m-%d'))].copy()
    if today_entries.empty:
        raise ValueError('today race snapshot has no racecards; refusing to publish empty all-race coverage')
    entries = entries[close.gt(now.timestamp())].copy()
    results = json.loads(Path(results_path).read_text(encoding='utf-8')) if Path(results_path).exists() else []
    decided_ids = {str(item.get('race_id')) for item in results
                   if item.get('official_result_available') and item.get('actual_trifecta')}
    entries = entries[~entries.race_id.astype(str).isin(decided_ids)].copy()
    valid_rows = []
    eligible_ids = set()
    skipped = []
    frozen_ids = {str(row['race_id']) for row in read_ledger()}
    entry_ids = set(today_entries.race_id.dropna().astype(str))
    for rid, race in today_entries.groupby('race_id', sort=False):
        rid = str(rid)
        if rid in frozen_ids:
            valid_rows.append((rid, race, '買い目固定済み', ''))
            continue
        if rid in decided_ids:
            valid_rows.append((rid, race, '予想なし', '予想保存前に公式結果が確定'))
            continue
        close_at = pd.to_numeric(race.close_at, errors='coerce').dropna()
        if close_at.empty or float(close_at.iloc[0]) <= now.timestamp():
            valid_rows.append((rid, race, '予想なし', '初回確認時点ですでに締切後'))
            continue
        eligible_ids.add(rid)
    eligible_input = entries[entries.race_id.astype(str).isin(eligible_ids)].copy()
    eligible, _ = old.eligible(features.mask_outcomes(eligible_input))
    accepted_ids = set(eligible.race_id.astype(str))
    for rid in eligible_ids - accepted_ids:
        race = today_entries[today_entries.race_id.astype(str).eq(rid)]
        reason = old.input_reason(race) or '候補生成対象外'
        valid_rows.append((rid, race, '予想なし', reason))
    if not eligible.empty and not pd.to_datetime(eligible.date).ge(pd.Timestamp(manifest['training_cutoff_exclusive'])).all():
        raise ValueError('a forecast date is not after the model training window')
    predicted = old.base_predict(eligible, bundle) if len(eligible) else eligible
    rows = read_ledger()
    known = {str(r['race_id']) for r in rows}
    new = []
    for rid, race in predicted.groupby('race_id', sort=False):
        rid = str(rid)
        close_at = float(pd.to_numeric(race.close_at, errors='coerce').iloc[0])
        if rid in known or not math.isfinite(close_at) or now.timestamp() >= close_at:
            continue
        try:
            ranked, anchored = rank_race_candidates(race, bundle)
        except Exception as error:
            valid_rows.append((rid, race, '予想なし', f'候補計算エラー: {type(error).__name__}'))
            continue
        if not ranked:
            valid_rows.append((rid, race, '予想なし', '有効な三連単候補なし'))
            continue
        item = race.iloc[0]
        race_no = pd.to_numeric(item.get('race_no'), errors='coerce')
        new.append({
            'race_id': rid, 'date': str(item.date), 'venue': str(item.get('venue') or ''),
            'race_no': int(race_no) if pd.notna(race_no) else 0, 'close_at': close_at,
            'snapshot_at_jst': now.isoformat(timespec='seconds'), 'top12': ranked[:12],
            'probabilities': [float(anchored[key]) for key in ranked[:12]],
            'training_cutoff_exclusive': manifest['training_cutoff_exclusive'],
            'model_sha256': manifest['model_sha256'], 'actual': None,
            'top12_hit': None, 'actual_odds': None, 'purchase_authorized': False,
        })
        valid_rows.append((rid, race, '買い目固定済み', ''))
    rows.extend(new)
    rows = apply_results(rows, results)
    write_ledger(rows)
    active = [r for r in rows if r['race_id'] in {x['race_id'] for x in new}]
    coverage_races = []
    status_by_id = {rid: (status, reason) for rid, _, status, reason in valid_rows}
    for rid, race in today_entries.groupby('race_id', sort=False):
        status, reason = status_by_id.get(str(rid), ('予想なし', '候補を固定できなかった'))
        item = race.iloc[0]
        race_no = pd.to_numeric(item.get('race_no'), errors='coerce')
        coverage_races.append({'race_id': str(rid), 'date': str(item.get('date', '')),
            'venue': str(item.get('venue') or ''), 'race_no': int(race_no) if pd.notna(race_no) else 0,
            'status': status, 'reason': reason})
    coverage = {'date': now.strftime('%Y-%m-%d'), 'input_races': int(len(entry_ids)),
        'forecasted_races': sum(x['status'] == '買い目固定済み' for x in coverage_races),
        'skipped_races': sum(x['status'] == '予想なし' for x in coverage_races),
        'races': coverage_races}
    LATEST.write_text(json.dumps({'updated_at_jst': now.isoformat(timespec='seconds'),
        'model_training_first': manifest['training_first'],
        'model_training_cutoff_exclusive': manifest['training_cutoff_exclusive'],
        'new_forecasts': active, 'coverage': coverage}, ensure_ascii=False, indent=2), encoding='utf-8')
    return build_report(rows, now)


def settle(results_path):
    rows = read_ledger()
    if Path(results_path).exists():
        results = json.loads(Path(results_path).read_text(encoding='utf-8'))
        rows = apply_results(rows, results)
        write_ledger(rows)
    return build_report(rows, datetime.now(ZoneInfo('Asia/Tokyo')))


def has_overdue_unsettled():
    now = datetime.now(ZoneInfo('Asia/Tokyo')).timestamp()
    return any(not row.get('actual') and pd.to_numeric(row.get('close_at'), errors='coerce') < now
               for row in read_ledger())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('train', 'forecast', 'settle', 'merge'), required=True)
    parser.add_argument('--history', type=Path, default=ROOT/'data/raw/history.csv')
    parser.add_argument('--entries', type=Path, default=ROOT/'data/raw/today_entries.csv')
    parser.add_argument('--results', type=Path, default=ROOT/'outputs/latest_results.json')
    parser.add_argument('--remote-ledger', type=Path)
    parser.add_argument('--asof')
    args = parser.parse_args()
    if args.stage == 'train':
        train(args.history, args.asof or datetime.now(ZoneInfo('Asia/Tokyo')).date().isoformat())
    elif args.stage == 'forecast':
        report = forecast(args.entries, args.results, args.asof)
        print(json.dumps(report, ensure_ascii=False))
    elif args.stage == 'settle':
        print(json.dumps(settle(args.results), ensure_ascii=False))
    else:
        print(json.dumps(merge_remote_ledger(args.remote_ledger), ensure_ascii=False))


if __name__ == '__main__':
    main()

