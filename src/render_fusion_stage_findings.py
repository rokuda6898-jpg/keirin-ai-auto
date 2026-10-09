"""Publish verified stage-study evidence; presentation only, never fit or predict."""
import argparse
import hashlib
import html
import json
import shutil
from pathlib import Path


NAMES = {'risk': 'リスク部の現行評価式', 'fusion': '従来の統合式',
         'context_joint': '① レース条件で配分変更', 'stage_global': '② 着順ごとに配分変更',
         'stage_context': '③ 条件と着順の両方'}
EXPERTS = {'original:model_positions': '順位別モデル', 'pairwise:pairwise_order': '先着関係式',
           'joint:prefix_context': '組み合わせ式', 'risk:current_logic': 'リスク部の現行式',
           'positions:data_department': 'データ部・順位保持', 'positions:pace_department': '展開部・順位保持',
           'positions:line_department': 'ライン部・順位保持', 'positions:risk_department': 'リスク部・順位保持',
           'legacy_positions:data_department:baseline': 'データ部・旧式',
           'legacy_positions:pace_department:baseline': '展開部・旧式',
           'legacy_positions:line_department:baseline': 'ライン部・旧式'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(source, destination):
    def read(name): return json.loads((source / name).read_text(encoding='utf-8'))
    result, model, frozen, predicted = (read(n) for n in ('stage_results.json', 'stage_model.json', 'stage_manifest.json', 'stage_prediction_manifest.json'))
    if digest(source / 'stage_model.json') != frozen['stage_model_sha256']:
        raise ValueError('model evidence mismatch')
    if result['prediction_manifest'] != predicted or predicted['stage_model_sha256'] != frozen['stage_model_sha256']:
        raise ValueError('result and prediction manifest mismatch')
    if not frozen['already_inspected_target_year'] or result['ceo_integration'] or result['risk_logic_changed']:
        raise ValueError('unexpected study scope')
    destination.mkdir(parents=True, exist_ok=True)
    for source_name, public_name in (
        ('stage_results.json', 'fusion_stage_results.json'), ('stage_model.json', 'fusion_stage_model.json'),
        ('stage_manifest.json', 'fusion_stage_manifest.json'),
        ('stage_prediction_manifest.json', 'fusion_stage_prediction_manifest.json'),
        ('execution_provenance.json', 'fusion_stage_execution_provenance.json')):
        shutil.copyfile(source / source_name, destination / public_name)
    overall = result['summary']['all']; methods = overall['methods']; n = overall['races']
    best = max(('context_joint', 'stage_global', 'stage_context'), key=lambda k: methods[k]['top12_hit'])
    diff = result['comparisons_to_original_fusion'][best]['top12']
    if diff['extra_hits'] > 0:
        headline = f'{NAMES[best]}が12候補で最多。従来の統合式より{diff["extra_hits"]:,}レース多く的中。'
    else:
        headline = '今回の３案では、12候補の的中数で従来の統合式を上回る案はありませんでした。'
    rows = ''
    for name in NAMES:
        m = methods[name]
        hits = round(m['top12_hit'] * n)
        difference = '—' if name == 'fusion' else f'{(m["top12_hit"] - methods["fusion"]["top12_hit"]) * 100:+.2f} pt'
        rows += f'<tr><th>{NAMES[name]}</th><td>{m["top1_hit"]:.2%}</td><td>{m["top6_hit"]:.2%}</td><td>{m["top12_hit"]:.2%}</td><td>{hits:,}</td><td>{difference}</td><td>{m["nll"]:.4f}</td></tr>'
    changes = ''
    for name in ('context_joint', 'stage_global', 'stage_context'):
        m = methods[name]; c = result['comparisons_to_original_fusion'][name]['top12']
        lo, hi = c['day_bootstrap_95_percent_interval']
        changes += f'<tr><th>{NAMES[name]}</th><td>{round(m["rescued_top12"] * n):,}</td><td>{round(m["lost_top12"] * n):,}</td><td>{c["extra_hits"]:+,}</td><td>{lo * 100:+.2f}〜{hi * 100:+.2f} pt</td></tr>'
    monthly = ''
    for group, data in sorted(result['summary'].items()):
        if group.startswith('month:'):
            monthly += f'<tr><th>{group[6:]}</th><td>{data["races"]:,}</td>' + ''.join(f'<td>{data["methods"][name]["top12_hit"]:.2%}</td>' for name in ('fusion', 'context_joint', 'stage_global', 'stage_context')) + '</tr>'
    stages = ''
    for name in NAMES:
        m = methods[name]
        stages += f'<tr><th>{NAMES[name]}</th>' + ''.join(f'<td>{m[k]:.4f}</td>' for k in ('first_nll', 'second_given_first_nll', 'third_given_pair_nll')) + '</tr>'
    weights = ''
    for i, name in enumerate(model['names']):
        weights += f'<tr><th>{html.escape(EXPERTS.get(name, name))}</th><td>{model["global_joint"][i]:.1%}</td>' + ''.join(f'<td>{w[i]:.1%}</td>' for w in model['global_stages']) + '</tr>'
    calibration = ''.join(f'<li>{NAMES[name]}：追加案の配合 <b>{c["blend"]:.0%}</b>、温度 {c["temperature"]}、古い選択期間の損失 {c["nll"]:.4f}</li>' for name, c in model['calibration']['variants'].items())
    limits = ''.join(f'<li>{html.escape(s)}</li>' for s in result['limitations'])
    document = f'''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KEIRIN NEXUS｜統合式の追加比較</title><style>body{{max-width:1120px;margin:auto;padding:28px;background:#f3f6fb;color:#17283e;font:16px/1.8 system-ui}}h1{{line-height:1.35}}h2{{margin-top:36px}}table{{width:100%;border-collapse:collapse;background:#fff}}th,td{{padding:10px 12px;border-bottom:1px solid #dde5ef;text-align:right;white-space:nowrap}}th:first-child{{text-align:left}}.scroll{{overflow:auto}}.lead{{background:#e0edf9;padding:22px;border-radius:12px;font-size:19px}}.note{{background:#fff0cc;padding:20px;border-radius:12px}}a{{color:#175aa5}}small{{overflow-wrap:anywhere}}</style>
<p>KEIRIN NEXUS · 追加式の探索研究</p><h1>各式の得意な着順を使い分ける</h1><p class="lead">{headline}</p>
<p class="note"><b>結果を一度見た１年での探索比較です。</b>配分と混合率は2025-10-09より前だけで学習・選択しました。ただし、このアイデア自体は前回の１年の結果を見た後に考えたため、新たな未使用期間の検証とは呼びません。本番の社長・７部署・リスク部のロジックと買い目は変更していません。</p>
<p>予想対象：2025-10-09〜2026-10-08。予想 {result['predicted_races']:,} レース、同じ {n:,} レースで比較。判定できない着順 {sum(result['outcome_exclusions'].values()):,} レースは全方式共通で集計から除外。</p>
<h2>同じ候補数で比べた的中率</h2><div class="scroll"><table><tr><th>方式</th><th>１候補</th><th>６候補</th><th>12候補</th><th>12候補の的中数</th><th>従来統合式との差</th><th>確率の損失↓</th></tr>{rows}</table></div>
<p>無価格の３連単候補の比較です。本線・100倍以上の穴・同金額の買い目比較、回収率は未検証です。候補数は追加していません。</p>
<h2>何を追加したか</h2><ol><li>出走人数×クラスで、各式の配分を変更。</li><li>１着、２着｜１着、３着｜１・２着の３段階で、別々の配分を学習。</li><li>①と②を組み合わせる。</li></ol>
<p><b>P(a,b,c) = Σw₁p(a) × Σw₂p(b｜a) × Σw₃p(c｜a,b)</b></p><p>予想時にはすべての１・２着候補を評価します。正解の１・２着は渡しません。少数条件は全体の配分へ戻し、同一予想の複製は１枠にまとめます。</p>
<h2>増えた的中と失った的中（12候補）</h2><div class="scroll"><table><tr><th>追加案</th><th>旧式の外れを救済</th><th>旧式の的中を失う</th><th>差し引き</th><th>差の参考95％区間</th></tr>{changes}</table></div><p>区間は日単位の再抽出。探索後の採用合格判定には使いません。</p>
<h2>どの段階の確率が改善したか</h2><div class="scroll"><table><tr><th>方式</th><th>１着の損失↓</th><th>２着｜１着の損失↓</th><th>３着｜１・２着の損失↓</th></tr>{stages}</table></div><p>合計が全体の損失。２・３着の診断では正しい前段が与えられた条件で確率を採点しますが、実際の予想は正解を使いません。</p>
<h2>月ごとの12候補的中率</h2><div class="scroll"><table><tr><th>月</th><th>レース</th><th>従来統合</th><th>① 条件別</th><th>② 着順別</th><th>③ 両方</th></tr>{monthly}</table></div><p>最初と最後の月は一部期間です。</p>
<h2>古いデータで決めた配合</h2><ul>{calibration}</ul><p>配分の学習：2024-10-13〜2025-04-10。配合率・温度の選択：2025-04-11〜2025-10-08。対数損失で選び、０％も候補に含めました。的中率が上がる保証ではありません。</p>
<details><summary>着順別に学習した全体の配分を見る</summary><div class="scroll"><table><tr><th>成分</th><th>従来の全着順</th><th>１着</th><th>２着｜１着</th><th>３着｜１・２着</th></tr>{weights}</table></div><p>温度補正と旧統合式への混合より前の配分。①と③には別途、人数×クラス別の配分があります。７部署は維持され、軍師などのシナリオは組み合わせ式の入力にも残っています。</p></details>
<h2>検証の限界</h2><ul>{limits}</ul><p>関連研究：<a href="https://arxiv.org/abs/2101.08954">条件に応じて配分する階層的スタッキング</a>。本実装は論文のベイズモデルの再現ではありません。</p>
<p><a href="fixed_year_report.html">前回の固定１年検証</a> · <a href="fusion_stage_results.json">全結果と条件別内訳</a> · <a href="fusion_stage_model.json">配分と選択結果</a> · <a href="fusion_stage_manifest.json">学習の証跡</a> · <a href="fusion_stage_prediction_manifest.json">予想保存の証跡</a> · <a href="fusion_stage_execution_provenance.json">実行の証跡</a></p>
<small>予想保存物の確認値：{predicted['predictions_sha256']}</small></html>'''
    (destination / 'fusion_stage_report.html').write_text(document, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.destination)
