"""Outcome-only assessment, after every repair forecast has been saved."""
import gzip
import html
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pandas as pd

import fixed_year_study as old
import fusion_equations as fusion
from fusion_input_repair import ARMS, LABELS, VERSION, read_json, source_hashes


def diagnostics(distribution, actual):
    ranked = sorted(distribution, key=lambda k: (-distribution[k], tuple(map(int, k.split('-')))))
    chosen = [k.split('-') for k in ranked[:12]]
    truth = actual.split('-')
    hit = actual in ranked[:12]
    heads = {t[0] for t in chosen}
    pairs = {tuple(t[:2]) for t in chosen}
    if hit:
        reason = 'hit'
    elif truth[0] not in heads:
        reason = 'first_absent_from_top12'
    elif tuple(truth[:2]) not in pairs:
        reason = 'true_pair_absent_from_top12'
    else:
        reason = 'third_missing_for_true_pair'
    marginal = Counter()
    for key, p in distribution.items():
        marginal[key.split('-')[0]] += p
    winner = min(marginal, key=lambda k: (-marginal[k], int(k)))
    rank = ranked.index(actual)+1
    return {'reason': reason, 'actual_rank': rank,
            'marginal_first_hit': int(winner == truth[0]),
            'podium_set_covered': int(any(set(t) == set(truth) for t in chosen)),
            'correct_set_wrong_order': int(not hit and any(set(t) == set(truth) for t in chosen)),
            'actual_rank_13_to_24': int(13 <= rank <= 24),
            'selected_probability_mass': sum(distribution[k] for k in ranked[:12])}


def compare(daily, method, reference, bootstrap_indices):
    delta = np.array([d[method]-d[reference] for d in daily], dtype=float)
    counts = np.array([d['races'] for d in daily], dtype=float)
    total = counts.sum()
    resampled_n = counts[bootstrap_indices].sum(axis=1)
    valid = resampled_n > 0
    values = delta[bootstrap_indices].sum(axis=1)[valid]/resampled_n[valid]
    return {'reference': reference, 'hit_difference': int(delta.sum()),
            'top12_hit_rate_difference': float(delta.sum()/total),
            'seven_day_block_bootstrap_95_percent_interval': np.quantile(values, [.025, .975]).tolist(),
            'scope': 'descriptive_exploratory_multiple_comparisons_not_promotion'}


def assess(target, source, folder):
    original = read_json(source/'prediction_manifest.json')
    if old.digest_file(source/'holdout_predictions.jsonl.gz') != original['predictions_sha256']:
        raise ValueError('original forecasts changed')
    original_results = read_json(source/'results.json')
    manifests = {}
    for arm in ARMS:
        model = read_json(folder/arm/'manifest.json')
        pred = read_json(folder/arm/'prediction_manifest.json')
        if model['source_hashes'] != source_hashes():
            raise ValueError('repair source changed')
        for name, expected in (('model.joblib', model['model_sha256']),
                               ('manifest.json', pred['model_manifest_sha256']),
                               ('predictions.jsonl.gz', pred['predictions_sha256'])):
            if old.digest_file(folder/arm/name) != expected:
                raise ValueError('repair evidence changed: ' + arm + '/' + name)
        if pred['source_predictions_sha256'] != original['predictions_sha256']:
            raise ValueError('different original comparator')
        manifests[arm] = {'model': model, 'prediction': pred}
    answers = {str(rid): old.outcome(r) for rid, r in target.groupby('race_id', sort=False)}
    grouped = defaultdict(lambda: defaultdict(Counter))
    diag_counts = defaultdict(Counter)
    counts, exclusions = Counter(), Counter()
    daily = defaultdict(Counter)
    rescue = defaultdict(Counter)
    seen = set()
    ledger = folder/'assessed_races.jsonl.gz'
    with ExitStack() as stack:
        baseline = stack.enter_context(gzip.open(source/'holdout_predictions.jsonl.gz', 'rt', encoding='utf-8'))
        handles = {arm: stack.enter_context(gzip.open(folder/arm/'predictions.jsonl.gz', 'rt', encoding='utf-8')) for arm in ARMS}
        evidence = stack.enter_context(gzip.open(ledger, 'wt', encoding='utf-8'))
        for line in baseline:
            saved = json.loads(line)
            rid = saved['race_id']
            if rid in seen:
                raise ValueError('duplicate prediction race')
            seen.add(rid)
            keys = saved['keys']
            distributions = {name: dict(zip(keys, saved['distributions'][name])) for name in ('risk', 'fusion', 'original')}
            for arm, handle in handles.items():
                row = json.loads(next(handle))
                if any(row[k] != saved[k] for k in ('race_id', 'date', 'keys', 'field', 'race_class')):
                    raise ValueError('unmatched prediction input cohort')
                distributions[arm] = dict(zip(keys, row['distributions']['fusion']))
                distributions[arm+'_base'] = dict(zip(keys, row['distributions']['base']))
                if arm == 'refit':
                    distributions['refit_first_anchor'] = dict(zip(keys, row['distributions']['first_anchor']))
            if any(not fusion.valid_distribution(p, keys) for p in distributions.values()):
                raise ValueError('invalid saved probabilities')
            actual = answers.get(rid)
            if actual is None:
                exclusions['ambiguous_or_missing_podium'] += 1
                evidence.write(json.dumps({'race_id': rid, 'exclusion': 'ambiguous_or_missing_podium'})+'\n')
                continue
            ms = {name: old.metrics(p, actual) for name, p in distributions.items()}
            diagnostic = {name: diagnostics(p, actual) for name, p in distributions.items()}
            groups = ('all', 'month:'+saved['date'][:7], 'field:'+str(saved['field']), 'class:'+saved['race_class'])
            for group in groups:
                counts[group] += 1
                for name, m in ms.items():
                    grouped[group][name].update(m)
            day = daily[saved['date']]
            day['races'] += 1
            for name, m in ms.items():
                day[name] += m['top12_hit']
                diag_counts[name][diagnostic[name]['reason']] += 1
                for k in ('marginal_first_hit', 'podium_set_covered', 'correct_set_wrong_order', 'actual_rank_13_to_24', 'selected_probability_mass'):
                    diag_counts[name][k] += diagnostic[name][k]
                for ref in ('fusion', 'risk'):
                    rescue[name]['rescued_vs_'+ref] += int(m['top12_hit'] and not ms[ref]['top12_hit'])
                    rescue[name]['lost_vs_'+ref] += int(ms[ref]['top12_hit'] and not m['top12_hit'])
            evidence.write(json.dumps({'race_id': rid, 'date': saved['date'], 'field': saved['field'],
                'race_class': saved['race_class'], 'actual': actual,
                'methods': {name: {'top12_hit': m['top12_hit'], **diagnostic[name]} for name, m in ms.items()}},
                ensure_ascii=False, allow_nan=False, separators=(',', ':'))+'\n')
            if counts['all'] % 3000 == 0:
                print('REPAIR_ASSESSED ' + str(counts['all']), flush=True)
        if any(next(handle, None) is not None for handle in handles.values()):
            raise ValueError('additional unmatched repaired forecasts')
    n = counts['all']
    if len(seen) != original['predicted_races'] or n != original_results['evaluated_races'] or dict(exclusions) != original_results['outcome_exclusions']:
        raise ValueError('evaluation cohort differs from the original study')
    summary = {g: {'races': counts[g], 'methods': {name: {k: float(v/counts[g]) for k, v in sums.items()}
                for name, sums in methods.items()}} for g, methods in grouped.items()}
    for name in ('risk', 'fusion', 'original'):
        for k, value in summary['all']['methods'][name].items():
            if abs(value-original_results['summary']['all']['methods'][name][k]) > 1e-12:
                raise ValueError('immutable baseline result did not reproduce: '+name+'/'+k)
    calendar = pd.date_range(min(daily), max(daily))
    days = [daily[str(d.date())] for d in calendar]
    width = min(7, len(days))
    rng = np.random.default_rng(20261009)
    starts = rng.integers(0, len(days)-width+1, size=(4000, math_ceil(len(days)/width)))
    indices = (starts[:, :, None]+np.arange(width)).reshape(4000, -1)[:, :len(days)]
    comparisons = {name: compare(days, name, 'fusion', indices) for name in summary['all']['methods']}
    incremental = {name+'_vs_'+ref: compare(days, name, ref, indices)
                   for name, ref in (('both', 'gap'), ('both', 'identity'), ('refit', 'both'), ('refit_first_anchor', 'refit'))}
    result = {'version': VERSION, 'status': 'exploratory_repaired_forecasts_scored',
              'predicted_races': len(seen), 'evaluated_races': n, 'outcome_exclusions': dict(exclusions),
              'summary': summary, 'diagnostics_top12': dict(diag_counts), 'rescued_and_lost': dict(rescue),
              'comparisons_to_original_fusion': comparisons, 'incremental_comparisons': incremental,
              'training_cutoff_exclusive': read_json(source/'inventory.json')['training_cutoff_exclusive'],
              'baseline_predictions_sha256': original['predictions_sha256'],
              'assessed_races_sha256': old.digest_file(ledger), 'manifests': manifests,
              'baseline_results_reproduced': True, 'already_inspected_target_year': True,
              'same_races_dates_candidate_counts': True,
              'candidate_points': [1, 6, 12], 'odds_and_actual_wagers_available': False,
              'roi': None, 'high_payout_dependency': None, 'main_hole_comparison_available': False,
              'automatic_promotion': False, 'ceo_integration': False, 'risk_logic_changed': False,
              'scope': 'retrospective_unpriced_full_field_trifecta_candidates_not_backdated_wagers'}
    old.write_json(folder/'results.json', result)
    render(result, folder/'fusion_input_repair_report.html')
    print('REPAIR_RESULTS ' + json.dumps({'races': n, 'methods': summary['all']['methods'],
        'comparisons': comparisons, 'incremental': incremental}, ensure_ascii=False), flush=True)
    return result


def math_ceil(value):
    return int(np.ceil(value))


def render(result, destination):
    metrics = result['summary']['all']['methods']
    n = result['evaluated_races']
    rows = []
    for name, label in LABELS.items():
        m = metrics[name]
        comparison = result['comparisons_to_original_fusion'][name]
        lo, hi = comparison['seven_day_block_bootstrap_95_percent_interval']
        rows.append(f'<tr><th>{html.escape(label)}</th><td>{m["top1_hit"]:.2%}</td><td>{m["top6_hit"]:.2%}</td>'
            f'<td>{m["top12_hit"]:.2%}</td><td>{round(m["top12_hit"]*n):,}</td>'
            f'<td>{comparison["hit_difference"]:+,}</td><td>{lo*100:+.3f}〜{hi*100:+.3f}</td><td>{m["nll"]:.4f}</td></tr>')
    diagnosis = []
    for name, label in LABELS.items():
        d = result['diagnostics_top12'][name]
        diagnosis.append(f'<tr><th>{html.escape(label)}</th>'+''.join(f'<td>{int(d.get(k,0)):,}</td>' for k in
            ('hit', 'first_absent_from_top12', 'true_pair_absent_from_top12', 'third_missing_for_true_pair', 'actual_rank_13_to_24', 'correct_set_wrong_order'))+'</tr>')
    monthly = []
    for group, values in sorted(result['summary'].items()):
        if group.startswith('month:'):
            monthly.append(f'<tr><th>{group[6:]}</th><td>{values["races"]:,}</td>'+''.join(
                f'<td>{values["methods"][name]["top12_hit"]:.2%}</td>' for name in ('fusion', 'gap', 'identity', 'both', 'refit'))+'</tr>')
    base_rows = ''.join(f'<tr><th>{html.escape(LABELS[arm])}</th><td>{metrics[arm+"_base"]["top12_hit"]:.2%}</td>'
        f'<td>{result["manifests"][arm]["model"]["final_base_training_last"]}</td></tr>' for arm in ARMS)
    document = '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KEIRIN NEXUS｜入力修正と再学習の一年検証</title><style>
body{margin:0;background:#0b1322;color:#e8edf7;font:16px/1.8 system-ui,sans-serif}main{max-width:1160px;margin:auto;padding:30px 20px}
h1{font-size:clamp(24px,4vw,36px)}h2{margin-top:36px}a{color:#8dc8ff}.notice{padding:18px;background:#182d43;border-left:4px solid #70c8bc}
.scroll{overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:14px}th,td{padding:10px;border-bottom:1px solid #324157;text-align:right}th:first-child{text-align:left}th{color:#b8dafa}small{color:#bbc8da}li{margin-bottom:8px}</style><main>
<p><a href="fixed_year_report.html">従来の一年検証</a> ／ <a href="fusion_stage_report.html">配合変更の検証</a></p>
<h1>入力修正と再学習の一年検証</h1>
<p class="notice">予想保存後に答え合わせを実施。リスク部・本番の買い目・社長への組み込みは変更していません。<br>
学習：2025年10月8日まで ／ 予想対象：2025年10月9日〜2026年10月8日。対象年の結果で再学習していません。</p>
<p>予想 __PREDICTED__ レース、同着・上位着順不明を除く共通 __SCORED__ レースで比較。保存済みの従来成績を再現しました。
締切前オッズが揃っていないため、これは三連単の上位候補1・6・12点の成績です。本線・100倍以上の穴・回収率の検証ではありません。</p>
<h2>修正を分けて比較</h2><ol>
<li>日数項目の修正：「前走間隔」と意味が異なる凍結記録からの経過日数を、学習・予想の両方から除外。別途保存された前走間隔の項目は維持。</li>
<li>選手ID項目の修正：別人の値が衝突し、数値の大小として扱われていたID特徴を除外。選手別の過去成績は維持。</li>
<li>両方を修正：①と②を同時に適用。その他は従来と同じ学習区間・係数設定。</li>
<li>両方修正＋再学習：統合学習用の予想は必ず過去だけで学習したモデルから作成。基礎モデルとプロフィールを段階的に更新し、最終的に古い期間全体で基礎モデルを学習。</li></ol>
<p>参考案は④の統合分布の1着確率だけを④の基礎モデルに合わせ、1着が決まった場合の2・3着評価を保持したものです。対象年の成績を見て係数を調整していません。
組み合わせ式と先着関係式の学習区間は全案で従来と同じです。④では基礎モデル・プロフィール・配合・確率調整を更新しています。</p>
<h2>的中率と従来統合式との差</h2><div class="scroll"><table><thead><tr><th>方式</th><th>1点</th><th>6点</th><th>12点</th><th>12点的中数</th><th>的中数差</th><th>差の95%範囲（pt）</th><th>確率の損失↓</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p><small>同じ日付の比較を保ち、連続7日単位で再標本化した記述的な範囲。対象年は以前にも分析済みのため、新しい未使用テストではありません。複数案の探索結果から自動採用しません。</small></p>
<h2>12点で外した場所</h2><div class="scroll"><table><thead><tr><th>方式</th><th>的中</th><th>正解1着なし</th><th>1着あり・正解1/2着なし</th><th>正解1/2着あり・3着なし</th><th>正解13〜24位</th><th>3人同じ・順番違い</th></tr></thead><tbody>__DIAG__</tbody></table></div>
<p>最初の4区分は重複なく全評価レースを分解。最後の2列は別の観点で重複あり。今回は全組み合わせを順位付けしており、実際の買い目条件で落ちた原因は評価していません。</p>
<h2>基礎モデルと学習範囲</h2><div class="scroll"><table><thead><tr><th>方式</th><th>基礎モデルの12点的中率</th><th>基礎学習の最終日</th></tr></thead><tbody>__BASE__</tbody></table></div>
<h2>月別・12点</h2><div class="scroll"><table><thead><tr><th>月</th><th>レース数</th><th>従来</th><th>日数修正</th><th>ID修正</th><th>両方</th><th>再学習</th></tr></thead><tbody>__MONTHS__</tbody></table></div>
<h2>今回の限界と保存記録</h2><p>保存済みアーカイブのレースであり全国全レースではありません。過去ページの特徴量が締切前に取得されたことは確認できません。
取得されていない過去の詳細展開・並び確認・オッズを作り足していません。予想ファイルに答えを混ぜず、答え合わせの記録を別保存しています。</p>
<p><a href="fusion_input_repair_results.json">成績・比較・学習範囲</a> ／ <a href="fusion_input_repair_provenance.json">実行証跡</a></p></main></html>'''
    replacements = {'PREDICTED': f'{result["predicted_races"]:,}', 'SCORED': f'{n:,}', 'ROWS': ''.join(rows),
                    'DIAG': ''.join(diagnosis), 'MONTHS': ''.join(monthly), 'BASE': base_rows}
    for key, value in replacements.items():
        document = document.replace('__'+key+'__', value)
    Path(destination).write_text(document, encoding='utf-8')
