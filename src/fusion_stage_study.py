"""Separate exploratory extension of the already inspected fixed-year study.

Reuses the exact frozen experts and archive. No refit of the base classifiers,
department profiles, risk formula, or live forecasts. Train/predict/assess are
separate commands; holdout predictions contain no outcomes.
"""
import argparse
import gzip
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import fixed_year_features as features
import fixed_year_models as experts
import fixed_year_study as old
import fusion_equations as fusion
import fusion_stage_context as stage
from fixed_year_research import split_history

LABELS = {'risk': 'リスク部の現行評価式', 'fusion': '従来の統合式',
          'context_joint': '① 出走人数・クラスで配分変更', 'stage_global': '② 着順ごとに配分変更',
          'stage_context': '③ 条件と着順の両方で配分変更'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def source_hashes():
    result = old.source_hashes()
    for name in ('fusion_stage_context.py', 'fusion_stage_study.py'):
        result[name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    return result


def verify_source(source, history_path):
    manifest = read_json(source / 'frozen_manifest.json')
    inventory = read_json(source / 'inventory.json')
    if old.digest_file(history_path) != inventory['history_sha256']:
        raise ValueError('archive changed from the original fixed-year study')
    if old.source_hashes() != manifest['source_hashes']:
        raise ValueError('original expert source changed')
    if old.digest_file(source / 'frozen_model.joblib') != manifest['model_sha256']:
        raise ValueError('original frozen model changed')
    if inventory['training_cutoff_exclusive'] != manifest['training_cutoff_exclusive']:
        raise ValueError('original cutoff mismatch')
    return manifest, inventory


def records(frame, bundle, labels=False):
    answers = {str(rid): old.outcome(r) for rid, r in frame.groupby('race_id', sort=False)} if labels else None
    clean, excluded = old.eligible(features.mask_outcomes(frame))
    if labels and excluded:
        raise ValueError('training input exclusions changed')
    rows = old.base_predict(clean, bundle)
    for count, (rid, race) in enumerate(rows.groupby('race_id', sort=False), 1):
        packet = experts.make_packet(race, bundle['profiles'])
        available = experts.components(bundle['equations'], packet)
        keys, matrix = stage.expert_matrix(bundle['equations']['pool'], available)
        baseline = fusion.temperature(fusion.mix(bundle['equations']['pool'], available), bundle['equations']['temperature'])
        row = {'race_id': str(rid), 'date': str(race.iloc[0].date), 'field': len(race),
               'race_class': str(race.iloc[0].get('race_class', 'missing')), 'keys': keys,
               'matrix': matrix,
               'baseline': np.array([baseline[k] for k in keys])}
        row['context'] = stage.context_key(row['field'], row['race_class'])
        # Only the training caller requests labels. Prediction never does.
        if labels:
            row['actual_index'] = keys.index(answers[str(rid)])
        row['risk'] = np.array([available['risk:current_logic'][k] for k in keys])
        yield row
        if count % 1000 == 0:
            print(f'STAGE_CONTEXT_PROGRESS processed {count}/{clean.race_id.nunique()}', flush=True)


def train(older, source, folder):
    frozen = read_json(source / 'frozen_manifest.json')
    cutoff = frozen['training_cutoff_exclusive']
    if not pd.to_datetime(older.date).lt(pd.Timestamp(cutoff)).all():
        raise ValueError('older training contains held-out dates')
    eligible, excluded = old.eligible(older, labels=True)
    phases = old.phase_split(eligible)
    for frame, phase in zip(phases, frozen['phases']):
        if min(frame.date) != phase['first'] or max(frame.date) != phase['last'] or frame.race_id.nunique() != phase['races']:
            raise ValueError('frozen training phase mismatch')
    bundle = joblib.load(source / 'frozen_model.joblib')
    joint, conditional, contexts, dates, ids = [], [], [], [], []
    for row in records(phases[2], bundle, labels=True):
        joint.append(row['matrix'][:, row['actual_index']])
        conditional.append(stage.decompose(row['keys'], row['matrix'])[:, :, row['actual_index']])
        contexts.append(row['context']); dates.append(row['date']); ids.append(row['race_id'])
    model = stage.fit(joint, conditional, contexts, dates, bundle['equations']['pool'], cutoff)
    np.savez_compressed(folder / 'older_fit_evidence.npz', joint=joint, conditional=conditional,
                        contexts=contexts, dates=dates, race_ids=ids)
    print('STAGE_CONTEXT_PROGRESS stage_and_context_weights_fitted', flush=True)
    def calibration_records():
        for row in records(phases[3], bundle, labels=True):
            row['raw'] = stage.raw_predictions(model, row['keys'], row['matrix'], row['context'])
            yield row
    model['calibration'] = stage.calibrate(calibration_records(), model)
    old.write_json(folder / 'stage_model.json', model)
    manifest = {'version': stage.VERSION, 'source_model_sha256': frozen['model_sha256'],
                'stage_model_sha256': old.digest_file(folder / 'stage_model.json'),
                'older_fit_evidence_sha256': old.digest_file(folder / 'older_fit_evidence.npz'),
                'source_hashes': source_hashes(), 'phases': frozen['phases'],
                'training_exclusions': excluded, 'training_cutoff_exclusive': cutoff,
                'model_update_during_target': False, 'already_inspected_target_year': True,
                'evaluation_scope': 'exploratory_reuse_not_new_untouched_holdout',
                'selection_uses_target_outcomes': False, 'ceo_integration': False,
                'risk_logic_changed': False, 'purchase_authorized': False}
    old.write_json(folder / 'stage_manifest.json', manifest)
    print('STAGE_CONTEXT_FROZEN ' + json.dumps({'calibration': {k: {f: v[f] for f in ('blend', 'temperature', 'nll')}
          for k, v in model['calibration']['variants'].items()}, 'baseline_nll': model['calibration']['baseline_nll']}), flush=True)
    return model


def load_stage(folder):
    manifest = read_json(folder / 'stage_manifest.json')
    if old.digest_file(folder / 'stage_model.json') != manifest['stage_model_sha256']:
        raise ValueError('stage model changed')
    if source_hashes() != manifest['source_hashes']:
        raise ValueError('source changed after fitting')
    return read_json(folder / 'stage_model.json'), manifest


def predict(target, source, folder):
    model, manifest = load_stage(folder)
    frozen = read_json(source / 'frozen_manifest.json')
    if frozen['model_sha256'] != manifest['source_model_sha256'] or old.digest_file(source / 'frozen_model.joblib') != frozen['model_sha256']:
        raise ValueError('original model changed')
    old_prediction = read_json(source / 'prediction_manifest.json')
    if old.digest_file(source / 'holdout_predictions.jsonl.gz') != old_prediction['predictions_sha256']:
        raise ValueError('original predictions changed')
    bundle = joblib.load(source / 'frozen_model.joblib')
    clean, exclusions = old.eligible(features.mask_outcomes(target))
    count, seen = 0, set()
    path = folder / 'stage_predictions.jsonl.gz'
    with gzip.open(source / 'holdout_predictions.jsonl.gz', 'rt', encoding='utf-8') as previous, gzip.open(path, 'wt', encoding='utf-8') as out:
        for row in records(clean, bundle):
            prior = json.loads(next(previous))
            if (row['race_id'] != prior['race_id'] or row['keys'] != prior['keys'] or row['date'] != prior['date']
                    or row['field'] != prior['field'] or row['race_class'] != prior['race_class']):
                raise ValueError('cohort or race order differs from original study')
            if row['race_id'] in seen:
                raise ValueError('duplicate predicted race')
            seen.add(row['race_id'])
            for name, values in (('fusion', row['baseline']), ('risk', row['risk'])):
                if not np.allclose(values, prior['distributions'][name], rtol=1e-11, atol=1e-14):
                    raise ValueError('frozen baseline forecasts cannot be reproduced')
            distributions = {'fusion': row['baseline'], 'risk': row['risk'],
                             **stage.predict(model, row['keys'], row['matrix'], row['context'], row['baseline'])}
            saved = {k: row[k] for k in ('race_id', 'date', 'field', 'race_class', 'keys', 'context')}
            saved['distributions'] = {k: p.tolist() for k, p in distributions.items()}
            saved['context_available'] = row['context'] in model['groups']
            out.write(json.dumps(saved, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
            count += 1
        if previous.readline():
            raise ValueError('original study contains unmatched races')
    if count != old_prediction['predicted_races']:
        raise ValueError('predicted cohort differs')
    load_stage(folder)
    result = {'predicted_races': count, 'input_exclusions': exclusions,
              'predictions_sha256': old.digest_file(path), 'stage_model_sha256': manifest['stage_model_sha256'],
              'original_predictions_sha256': old_prediction['predictions_sha256'],
              'outcomes_accessed': False, 'model_updated': False, 'baseline_reproduced': True}
    old.write_json(folder / 'stage_prediction_manifest.json', result)
    print('STAGE_CONTEXT_PREDICTED ' + json.dumps(result), flush=True)


def assess(target, source, folder):
    model, frozen = load_stage(folder)
    manifest = read_json(folder / 'stage_prediction_manifest.json')
    if manifest['stage_model_sha256'] != frozen['stage_model_sha256']:
        raise ValueError('prediction model mismatch')
    if old.digest_file(folder / 'stage_predictions.jsonl.gz') != manifest['predictions_sha256']:
        raise ValueError('prediction evidence changed')
    answers = {str(rid): old.outcome(r) for rid, r in target.groupby('race_id', sort=False)}
    grouped = defaultdict(lambda: defaultdict(Counter)); counts = Counter(); exclusions = Counter()
    daily = defaultdict(lambda: defaultdict(Counter)); seen = set(); context_coverage = Counter()
    with gzip.open(folder / 'stage_predictions.jsonl.gz', 'rt', encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line); rid = row['race_id']
            if rid in seen or rid not in answers:
                raise ValueError('duplicate or foreign race')
            seen.add(rid)
            actual = answers[rid]
            if not actual:
                exclusions['ambiguous_or_missing_podium'] += 1; continue
            scores = {name: old.metrics(dict(zip(row['keys'], values)), actual)
                      for name, values in row['distributions'].items()}
            context_coverage[str(row['context_available'])] += 1
            for group in ('all', 'month:' + row['date'][:7], 'field:' + str(row['field']), 'class:' + row['race_class']):
                counts[group] += 1
                for name, metrics in scores.items():
                    grouped[group][name].update(metrics)
            daily[row['date']]['count']['races'] += 1
            for name, metrics in scores.items():
                daily[row['date']][name].update(metrics)
                for k in (1, 6, 12):
                    current, baseline = metrics[f'top{k}_hit'], scores['fusion'][f'top{k}_hit']
                    grouped['all'][name][f'rescued_top{k}'] += int(current and not baseline)
                    grouped['all'][name][f'lost_top{k}'] += int(baseline and not current)
    if len(seen) != manifest['predicted_races'] or not counts['all']:
        raise ValueError('incomplete prediction cohort')
    previous = read_json(source / 'results.json')
    if counts['all'] != previous['evaluated_races'] or dict(exclusions) != previous['outcome_exclusions']:
        raise ValueError('evaluated cohort differs from original')
    summary = {g: {'races': counts[g], 'methods': {name: {k: float(v / counts[g]) for k, v in values.items()}
               for name, values in methods.items()}} for g, methods in grouped.items()}
    days = sorted(daily); sizes = np.array([daily[d]['count']['races'] for d in days])
    rng = np.random.default_rng(6908); ix = rng.integers(0, len(days), size=(2000, len(days)))
    comparisons = {}
    for name in stage.VARIANTS:
        comparisons[name] = {}
        for k in (1, 6, 12):
            delta = np.array([daily[d][name][f'top{k}_hit'] - daily[d]['fusion'][f'top{k}_hit'] for d in days])
            draws = delta[ix].sum(axis=1) / sizes[ix].sum(axis=1)
            comparisons[name][f'top{k}'] = {'extra_hits': int(delta.sum()), 'difference': float(delta.sum() / counts['all']),
                'day_bootstrap_95_percent_interval': np.quantile(draws, [.025, .975]).tolist()}
    result = {'version': stage.VERSION, 'status': 'exploratory_reused_year_scored',
              'predicted_races': len(seen), 'evaluated_races': counts['all'], 'outcome_exclusions': dict(exclusions),
              'summary': summary, 'comparisons_to_original_fusion': comparisons,
              'calibration': model['calibration'], 'context_coverage': dict(context_coverage),
              'prediction_manifest': manifest, 'roi': None, 'main_hole_comparison_available': False,
              'ceo_integration': False, 'risk_logic_changed': False, 'purchase_authorized': False,
              'limitations': previous['limitations'] + [
                  '前回の１年の結果を見た後に考案した追加式。同じ１年での探索比較であり、新たな未使用期間の検証ではない。',
                  '追加式の学習は従来のmixture期間、配合率・温度の選択は従来のtemperature期間。どちらも１年前より古い。',
                  '基礎モデルと部署の知識は前回の2023-10-17時点の固定物を再利用。全旧データでの再学習ではない。',
                  '区間は日単位再抽出による参考値。複数案の探索後の採用合格判定には使わない。']}
    old.write_json(folder / 'stage_results.json', result)
    render(result, folder)
    print('STAGE_CONTEXT_RESULTS ' + json.dumps({'overall': summary['all'], 'comparisons': comparisons}, ensure_ascii=False), flush=True)
    return result


def render(result, folder):
    metrics = result['summary']['all']['methods']
    rows = ''.join(f'<tr><th>{html.escape(LABELS[n])}</th><td>{m["top1_hit"]:.2%}</td><td>{m["top6_hit"]:.2%}</td>'
                   f'<td>{m["top12_hit"]:.2%}</td><td>{m["nll"]:.4f}</td></tr>' for n, m in metrics.items())
    calibration = ''.join(f'<li>{LABELS[n]}：追加案の配合 {c["blend"]:.0%}、温度 {c["temperature"]}、古い選択期間の損失 {c["nll"]:.4f}</li>'
                          for n, c in result['calibration']['variants'].items())
    limitations = ''.join(f'<li>{html.escape(x)}</li>' for x in result['limitations'])
    doc = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>統合式への追加案・探索比較</title><style>body{{max-width:1000px;margin:auto;padding:28px;background:#f4f7fb;color:#15253a;font:16px/1.8 system-ui}}h1{{line-height:1.4}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:12px;border-bottom:1px solid #dce3eb;text-align:right}}th:first-child{{text-align:left}}.note{{background:#fff1cc;padding:20px;border-radius:12px}}.scroll{{overflow:auto}}</style>
<h1>統合式への追加案<br>着順ごとの得意分野 × レース条件</h1>
<p class="note">結果を一度見た１年での<b>探索比較</b>です。新たな未使用期間の検証ではありません。追加式の係数・配合率には、この１年の結果を使っていません。</p>
<p>予想 {result['predicted_races']:,} レース／同じ {result['evaluated_races']:,} レースで答え合わせ。７部署・現行リスク部・社長の本番判断は維持。</p>
<h2>同じ候補数での的中率</h2><div class="scroll"><table><tr><th>方式</th><th>１候補</th><th>６候補</th><th>12候補</th><th>確率の損失↓</th></tr>{rows}</table></div>
<p>事前オッズを持たない予想候補の比較です。本線・100倍以上の穴・回収率の検証ではありません。</p>
<h2>追加した考え方</h2><p>①は出走人数とクラスに応じた配分、②は１着・２着｜１着・３着｜１／２着の３段階ごとの配分、③はその両方。少数条件は全体へ戻し、同じ予想の複製は１枠にまとめます。</p>
<p>P(a,b,c) = Σw₁p(a) × Σw₂p(b｜a) × Σw₃p(c｜a,b)</p>
<p>正解の１・２着を予想に渡しません。すべての１・２着候補について計算し、全３連単の合計を１にします。</p>
<h2>１年前より古い期間で決めた配合</h2><ul>{calibration}</ul><p>損失を基準に配合率と温度を選択。配合０％も選べます。的中率が上がったと事前に保証する選択ではありません。</p>
<h2>解釈上の制限</h2><ul>{limitations}</ul>
<p>関連研究：<a href="https://arxiv.org/abs/2101.08954">条件に応じて配分する階層的スタッキング</a>。本実装は縮約付きの離散条件配分であり、論文のベイズモデルの再現ではありません。</p>
<p><a href="fixed_year_report.html">前回の固定１年検証</a> · <a href="fusion_stage_results.json">全結果</a> · <a href="fusion_stage_manifest.json">学習の証跡</a> · <a href="fusion_stage_model.json">配分と選択結果</a></p></html>'''
    (folder / 'fusion_stage_report.html').write_text(doc, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, default=Path('data/raw/history.csv'))
    parser.add_argument('--source', type=Path, default=Path('research/frozen_original'))
    parser.add_argument('--output', type=Path, default=Path('research/fusion_stage_context'))
    parser.add_argument('--stage', choices=('train', 'predict', 'assess'), required=True)
    args = parser.parse_args()
    _, inventory = verify_source(args.source, args.history)
    history = pd.read_csv(args.history, dtype={'race_id': str, 'player_id': str}, low_memory=False)
    older, target = split_history(history, inventory['asof_exclusive'])
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == 'train': train(older, args.source, args.output)
    elif args.stage == 'predict': predict(target, args.source, args.output)
    else: assess(target, args.source, args.output)


if __name__ == '__main__':
    main()
